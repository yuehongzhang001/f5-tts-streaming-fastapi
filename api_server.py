import os
import logging
import re
import time
import json
import struct
from importlib.resources import files

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from hydra.utils import get_class
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

from f5_tts.infer.utils_infer import (
    infer_batch_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)
from f5_tts.api import F5TTS

logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

app = FastAPI(title="F5-TTS Unified API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

MODEL_NAME = "F5TTS_v1_Base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ⚡ 优化的推理配置
INFERENCE_CONFIG = {
    "nfe_step": 32,
    "cfg_strength": 2.5,
    "sway_sampling_coef": -1.0,
    "chunk_size": 512,
}

# 文件路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHARACTERS_FILE = os.path.join(BASE_DIR, "characters.json")
OUTPUT_WAV = os.path.join(BASE_DIR, "output.wav")


def create_wav_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16):
    """创建 WAV 文件头"""
    datasize = 0xFFFFFFFF - 36
    
    header = struct.pack('<4sI4s', b'RIFF', datasize + 36, b'WAVE')
    
    fmt_chunk = struct.pack(
        '<4sIHHIIHH',
        b'fmt ',
        16,
        1,
        num_channels,
        sample_rate,
        sample_rate * num_channels * bits_per_sample // 8,
        num_channels * bits_per_sample // 8,
        bits_per_sample
    )
    
    data_header = struct.pack('<4sI', b'data', datasize)
    
    return header + fmt_chunk + data_header


class TTSModel:
    def __init__(self):
        self.model = None
        self.vocoder = None
        self.sampling_rate = None
        self.f5tts_api = None
        self.characters = {}
        self.warmup_audio = None
        self.warmup_sr = None

    def load(self):
        logger.info(f"Loading on {DEVICE} ({DTYPE})...")
        start = time.time()

        model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{MODEL_NAME}.yaml")))
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        model_arc = model_cfg.model.arch
        mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
        self.sampling_rate = model_cfg.model.mel_spec.target_sample_rate

        ckpt_file = str(hf_hub_download(repo_id="SWivid/F5-TTS", filename="F5TTS_v1_Base/model_1250000.safetensors"))

        self.model = load_model(
            model_cls, model_arc, ckpt_path=ckpt_file, mel_spec_type=mel_spec_type,
            vocab_file="", ode_method="euler", use_ema=True, device=DEVICE,
        ).to(DEVICE, dtype=DTYPE)

        self.model.eval()
        torch.set_grad_enabled(False)

        if DEVICE == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("✅ Model compiled")
            except:
                pass

        self.vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=False, local_path=None, device=DEVICE)
        self.vocoder.eval()

        logger.info("Loading F5TTS API for file generation...")
        self.f5tts_api = F5TTS()

        # 加载角色配置
        self.load_characters()
        
        # 使用第一个可用角色进行预热
        self._warmup_model()

        logger.info(f"✅ Ready in {time.time()-start:.1f}s!")

    def _warmup_model(self):
        """使用第一个可用角色进行模型预热"""
        if not self.characters:
            logger.warning("⚠️  No characters configured, skipping warmup")
            return
        
        # 获取第一个角色配置
        first_character = list(self.characters.keys())[0]
        char_config = self.characters[first_character]
        
        try:
            logger.info(f"Warming up with character '{first_character}'...")
            ref_audio, ref_text = preprocess_ref_audio_text(
                char_config["ref_file"], 
                char_config["ref_text"]
            )
            self.warmup_audio, self.warmup_sr = torchaudio.load(ref_audio)
            
            # 预热推理
            for text in ["Hi.", "Test warmup.", "Hello world!"]:
                for _ in infer_batch_process(
                    (self.warmup_audio, self.warmup_sr), 
                    char_config["ref_text"], 
                    [text], 
                    self.model, 
                    self.vocoder,
                    progress=None, 
                    device=DEVICE, 
                    streaming=True,
                    chunk_size=INFERENCE_CONFIG["chunk_size"], 
                    nfe_step=INFERENCE_CONFIG["nfe_step"],
                ):
                    pass
            
            logger.info("✅ Warmup completed")
        except Exception as e:
            logger.warning(f"⚠️  Warmup failed: {e}")

    def load_characters(self):
        """加载角色配置"""
        if not os.path.exists(CHARACTERS_FILE):
            logger.error(f"❌ Characters file not found: {CHARACTERS_FILE}")
            logger.error("Please create a characters.json file with at least one character configuration")
            raise FileNotFoundError(f"Required file not found: {CHARACTERS_FILE}")
        
        try:
            with open(CHARACTERS_FILE, 'r', encoding='utf-8') as f:
                self.characters = json.load(f)
            
            if not self.characters:
                logger.error("❌ characters.json is empty")
                raise ValueError("characters.json must contain at least one character")
            
            # 验证所有角色的文件是否存在
            valid_characters = {}
            for name, config in self.characters.items():
                if "ref_file" not in config or "ref_text" not in config:
                    logger.warning(f"⚠️  Character '{name}' missing ref_file or ref_text, skipping")
                    continue
                
                if not os.path.exists(config["ref_file"]):
                    logger.warning(f"⚠️  Reference file not found for character '{name}': {config['ref_file']}")
                    continue
                
                valid_characters[name] = config
            
            if not valid_characters:
                logger.error("❌ No valid characters found in characters.json")
                raise ValueError("No valid characters with existing audio files")
            
            self.characters = valid_characters
            logger.info(f"✅ Loaded {len(self.characters)} valid characters: {list(self.characters.keys())}")
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse characters.json: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load characters: {e}")
            raise

    def get_character_config(self, character: str = None):
        """获取角色配置"""
        # 如果没有指定角色或指定为 default，使用第一个角色
        if not character or character == "default":
            if not self.characters:
                raise ValueError("No characters configured")
            first_character = list(self.characters.keys())[0]
            logger.info(f"Using default character: {first_character}")
            return self.characters[first_character]
        
        # 重新加载配置（支持热更新）
        if os.path.exists(CHARACTERS_FILE):
            try:
                with open(CHARACTERS_FILE, 'r', encoding='utf-8') as f:
                    updated_characters = json.load(f)
                    if character in updated_characters:
                        config = updated_characters[character]
                        if os.path.exists(config["ref_file"]):
                            self.characters[character] = config
            except Exception as e:
                logger.warning(f"Failed to reload character config: {e}")
        
        if character in self.characters:
            return self.characters[character]
        else:
            available = list(self.characters.keys())
            logger.warning(f"Character '{character}' not found. Available: {available}")
            raise HTTPException(
                status_code=400, 
                detail=f"Character '{character}' not found. Available characters: {available}"
            )


tts_model = TTSModel()


@app.on_event("startup")
async def startup():
    tts_model.load()


class TTSRequest(BaseModel):
    text: str
    character: str = "default"
    nfe_step: int = None
    cfg_strength: float = None
    speed: float = 1.0


def smart_split(text: str, first_chunk_max: int = 20, regular_chunk_max: int = 50):
    """智能分句"""
    text = re.sub(r'\s+', ' ', text.strip())
    
    strong_delimiters = r'([。！？\n.!?]+)'
    weak_delimiters = r'([，；：、,;:]+)'
    
    sentences = []
    remaining = text
    is_first = True
    
    while remaining:
        max_len = first_chunk_max if is_first else regular_chunk_max
        
        if len(remaining) <= max_len:
            if remaining.strip():
                sentences.append(remaining.strip())
            break
        
        search_text = remaining[:max_len + 20]
        
        strong_matches = list(re.finditer(strong_delimiters, search_text))
        if strong_matches:
            last_match = strong_matches[-1]
            split_pos = last_match.end()
            chunk = remaining[:split_pos].strip()
            if chunk:
                sentences.append(chunk)
            remaining = remaining[split_pos:].strip()
            is_first = False
            continue
        
        weak_matches = list(re.finditer(weak_delimiters, search_text[:max_len]))
        if weak_matches:
            last_match = weak_matches[-1]
            split_pos = last_match.end()
            chunk = remaining[:split_pos].strip()
            if chunk:
                sentences.append(chunk)
            remaining = remaining[split_pos:].strip()
            is_first = False
            continue
        
        space_pos = search_text[:max_len].rfind(' ')
        if space_pos > max_len // 2:
            chunk = remaining[:space_pos].strip()
            if chunk:
                sentences.append(chunk)
            remaining = remaining[space_pos:].strip()
            is_first = False
            continue
        
        chunk = remaining[:max_len].strip()
        if chunk:
            sentences.append(chunk)
        remaining = remaining[max_len:].strip()
        is_first = False
    
    sentences = [s for s in sentences if len(s) >= 2]
    return sentences


def generate_audio_for_sentence(sentence: str, ref_audio_tuple, ref_text: str, nfe: int, cfg_strength: float):
    """单句推理 - 改进的音频生成"""
    audio_stream = infer_batch_process(
        ref_audio_tuple, ref_text, [sentence],
        tts_model.model, tts_model.vocoder, progress=None, device=DEVICE,
        streaming=True, chunk_size=INFERENCE_CONFIG["chunk_size"], nfe_step=nfe,
        cfg_strength=cfg_strength,
        sway_sampling_coef=INFERENCE_CONFIG["sway_sampling_coef"],
    )

    sentence_audio = []
    for audio_chunk, _ in audio_stream:
        if audio_chunk is not None and len(audio_chunk) > 0:
            sentence_audio.append(audio_chunk)
    
    if sentence_audio:
        full_audio = np.concatenate(sentence_audio)
        
        # 更精细的归一化，避免削波失真
        peak = np.abs(full_audio).max()
        if peak > 0:
            full_audio = full_audio * (0.95 / peak)
        
        # 添加轻微的淡入淡出，减少爆音
        fade_samples = min(100, len(full_audio) // 10)
        if fade_samples > 0:
            fade_in = np.linspace(0, 1, fade_samples)
            full_audio[:fade_samples] *= fade_in
            
            fade_out = np.linspace(1, 0, fade_samples)
            full_audio[-fade_samples:] *= fade_out
        
        # 使用 round 减少量化误差
        pcm16 = np.clip(full_audio, -1.0, 1.0)
        pcm16 = np.round(pcm16 * 32767).astype(np.int16)
        
        yield pcm16.tobytes()


def audio_stream_generator(text: str, character: str = "default", nfe_step: int = None, cfg_strength: float = None):
    """流式生成器"""
    gen_start = time.time()
    req_id = f"{int(gen_start * 1000) % 100000}"

    logger.info(f"[{req_id}] Stream Request: '{text[:80]}{'...' if len(text) > 80 else ''}' (character: {character})")

    try:
        wav_header = create_wav_header(tts_model.sampling_rate)
        yield wav_header
        
        char_config = tts_model.get_character_config(character)
        ref_file = char_config["ref_file"]
        ref_text = char_config["ref_text"]

        ref_audio, processed_ref_text = preprocess_ref_audio_text(ref_file, ref_text)
        audio, sr = torchaudio.load(ref_audio)

        nfe = nfe_step if nfe_step is not None else INFERENCE_CONFIG["nfe_step"]
        cfg = cfg_strength if cfg_strength is not None else INFERENCE_CONFIG["cfg_strength"]
        
        logger.info(f"[{req_id}] Using character='{character}', nfe_step={nfe}, cfg_strength={cfg}")
        
        sentences = smart_split(text, first_chunk_max=20, regular_chunk_max=50)

        logger.info(f"[{req_id}] Split into {len(sentences)} chunks")
        for i, s in enumerate(sentences):
            logger.info(f"[{req_id}]   Chunk[{i+1}]: '{s[:60]}{'...' if len(s) > 60 else ''}' ({len(s)} chars)")

        chunk_count = 0
        first_logged = False

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            sent_start = time.time()

            for audio_bytes in generate_audio_for_sentence(sentence, (audio, sr), processed_ref_text, nfe, cfg):
                chunk_count += 1

                if not first_logged:
                    ttfa = (time.time() - gen_start) * 1000
                    logger.info(f"[{req_id}] ⚡ First audio at {ttfa:.0f}ms")
                    first_logged = True

                yield audio_bytes

            logger.info(f"[{req_id}] Chunk[{i+1}/{len(sentences)}] done in {(time.time()-sent_start)*1000:.0f}ms")

        total_time = (time.time() - gen_start) * 1000
        logger.info(f"[{req_id}] ✅ Complete: {chunk_count} audio chunks in {total_time:.0f}ms")

    except Exception as e:
        logger.error(f"[{req_id}] ❌ Error: {e}", exc_info=True)
        raise


@app.post("/tts/stream")
def tts_stream(req: TTSRequest):
    """流式TTS API"""
    if not req.text.strip():
        raise HTTPException(400, "Empty text")

    return StreamingResponse(
        audio_stream_generator(req.text, req.character, req.nfe_step, req.cfg_strength),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "inline; filename=stream.wav",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/tts")
def tts_file(req: TTSRequest):
    """完整文件TTS API"""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    char_config = tts_model.get_character_config(req.character)
    ref_file = char_config["ref_file"]
    ref_text = char_config["ref_text"]
    
    if not os.path.exists(ref_file):
        raise HTTPException(status_code=400, detail=f"Reference audio file does not exist: {ref_file}")
    
    start_time = time.time()
    
    nfe = req.nfe_step or INFERENCE_CONFIG["nfe_step"]
    cfg = req.cfg_strength or INFERENCE_CONFIG["cfg_strength"]
    speed = req.speed or 1.0
    
    logger.info(f"File generation: text='{req.text[:80]}...', character={req.character}, nfe={nfe}, cfg={cfg}, speed={speed}")
    
    wav, sr, spec = tts_model.f5tts_api.infer(
        ref_file=ref_file,
        ref_text=ref_text,
        gen_text=req.text,
        cfg_strength=cfg,
        nfe_step=nfe,
        speed=speed,
        file_wave=OUTPUT_WAV,
        file_spec=None
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    logger.info(f"Generated WAV file in {processing_time:.2f}s using character '{req.character}'")

    return FileResponse(OUTPUT_WAV, media_type="audio/wav", filename="output.wav")


@app.get("/characters")
def list_characters():
    """列出所有可用角色"""
    tts_model.load_characters()
    return {
        "characters": list(tts_model.characters.keys()),
        "default": list(tts_model.characters.keys())[0] if tts_model.characters else None
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "config": INFERENCE_CONFIG,
        "characters_loaded": len(tts_model.characters),
        "available_characters": list(tts_model.characters.keys())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0",
        port=8000, 
        log_level="info",
        access_log=False,
    )