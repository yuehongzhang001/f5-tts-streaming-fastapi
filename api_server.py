import os
import logging
import re
import time
import json
import struct
import io
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

INFERENCE_CONFIG = {
    "nfe_step": 32,
    "cfg_strength": 4.0,
    "sway_sampling_coef": -1.0,
    "chunk_size": 512,
}

# 文件路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHARACTERS_FILE = os.path.join(BASE_DIR, "characters.json")
OUTPUT_WAV = os.path.join(BASE_DIR, "output.wav")
DEFAULT_REF_AUDIO_PATH = os.path.join(BASE_DIR, "audio/female_1.mp3")
DEFAULT_REF_TEXT = "光线穿过百叶窗在桌面投下均匀条纹"


def create_wav_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16):
    """创建 WAV 文件头"""
    # 注意：这里 data_size 设为最大值，因为流式传输时我们不知道总大小
    datasize = 0xFFFFFFFF - 36  # 最大可能值
    
    header = struct.pack('<4sI4s', b'RIFF', datasize + 36, b'WAVE')
    
    # fmt chunk
    fmt_chunk = struct.pack(
        '<4sIHHIIHH',
        b'fmt ',
        16,  # fmt chunk size
        1,   # PCM format
        num_channels,
        sample_rate,
        sample_rate * num_channels * bits_per_sample // 8,  # byte rate
        num_channels * bits_per_sample // 8,  # block align
        bits_per_sample
    )
    
    # data chunk header
    data_header = struct.pack('<4sI', b'data', datasize)
    
    return header + fmt_chunk + data_header


class TTSModel:
    def __init__(self):
        self.model = None
        self.vocoder = None
        self.audio = None
        self.sr = None
        self.sampling_rate = None
        self.f5tts_api = None
        self.characters = {}

    def load(self):
        logger.info(f"Loading on {DEVICE} ({DTYPE})...")
        start = time.time()

        # 加载流式模型
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
            except: pass

        self.vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=False, local_path=None, device=DEVICE)
        self.vocoder.eval()

        # 加载默认参考音频
        ref_audio, ref_text = preprocess_ref_audio_text(DEFAULT_REF_AUDIO_PATH, DEFAULT_REF_TEXT)
        self.audio, self.sr = torchaudio.load(ref_audio)

        # 加载 F5TTS API（用于完整文件生成）
        logger.info("Loading F5TTS API for file generation...")
        self.f5tts_api = F5TTS()

        # 加载角色配置
        self.load_characters()

        logger.info("Warming up...")
        for text in ["Hi.", "Test warmup.", "Hello world!"]:
            for _ in infer_batch_process(
                (self.audio, self.sr), DEFAULT_REF_TEXT, [text], self.model, self.vocoder,
                progress=None, device=DEVICE, streaming=True,
                chunk_size=INFERENCE_CONFIG["chunk_size"], nfe_step=INFERENCE_CONFIG["nfe_step"],
            ):
                pass

        logger.info(f"✅ Ready in {time.time()-start:.1f}s!")

    def load_characters(self):
        """加载角色配置"""
        if os.path.exists(CHARACTERS_FILE):
            try:
                with open(CHARACTERS_FILE, 'r', encoding='utf-8') as f:
                    self.characters = json.load(f)
                logger.info(f"Loaded {len(self.characters)} characters from config")
            except Exception as e:
                logger.warning(f"Failed to load characters: {e}")
                self.characters = {}
        else:
            logger.warning(f"Characters file not found: {CHARACTERS_FILE}")
            self.characters = {}

    def get_character_config(self, character: str = None):
        """获取角色配置，如果未指定或不存在则使用默认"""
        if not character or character == "default":
            return {
                "ref_file": DEFAULT_REF_AUDIO_PATH,
                "ref_text": DEFAULT_REF_TEXT
            }
        
        # 重新加载配置（支持热更新）
        self.load_characters()
        
        if character in self.characters:
            config = self.characters[character]
            if not os.path.exists(config["ref_file"]):
                logger.warning(f"Reference file not found for character '{character}', using default")
                return {
                    "ref_file": DEFAULT_REF_AUDIO_PATH,
                    "ref_text": DEFAULT_REF_TEXT
                }
            return config
        else:
            logger.warning(f"Character '{character}' not found, using default. Available: {list(self.characters.keys())}")
            return {
                "ref_file": DEFAULT_REF_AUDIO_PATH,
                "ref_text": DEFAULT_REF_TEXT
            }


tts_model = TTSModel()


@app.on_event("startup")
async def startup():
    tts_model.load()


class TTSRequest(BaseModel):
    text: str
    character: str = "default"  # 角色名称
    nfe_step: int = None  # 推理步数，默认使用配置值
    cfg_strength: float = None  # CFG强度，默认使用配置值
    speed: float = 1.0  # 语速（仅用于文件生成）


def smart_split(text: str, first_chunk_max: int = 20, regular_chunk_max: int = 50):
    """
    智能分句策略：
    1. 首句控制在较短长度以快速返回
    2. 后续句子可以更长，保证完整性
    3. 优先在强标点处分割（。！？\n）
    4. 次选在弱标点处分割（，；：、）
    5. 避免在句子中间切断
    """
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
        
        # 优先查找强分隔符
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
        
        # 查找弱分隔符
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
        
        # 在空格处分割
        space_pos = search_text[:max_len].rfind(' ')
        if space_pos > max_len // 2:
            chunk = remaining[:space_pos].strip()
            if chunk:
                sentences.append(chunk)
            remaining = remaining[space_pos:].strip()
            is_first = False
            continue
        
        # 硬切
        chunk = remaining[:max_len].strip()
        if chunk:
            sentences.append(chunk)
        remaining = remaining[max_len:].strip()
        is_first = False
    
    sentences = [s for s in sentences if len(s) >= 2]
    return sentences


def generate_audio_for_sentence(sentence: str, ref_audio_tuple, ref_text: str, nfe: int, cfg_strength: float):
    """单句推理 - 完整处理整个句子，不再内部分块"""
    audio_stream = infer_batch_process(
        ref_audio_tuple, ref_text, [sentence],
        tts_model.model, tts_model.vocoder, progress=None, device=DEVICE,
        streaming=True, chunk_size=INFERENCE_CONFIG["chunk_size"], nfe_step=nfe,
        cfg_strength=cfg_strength,
        sway_sampling_coef=INFERENCE_CONFIG["sway_sampling_coef"],
    )

    # 收集整个句子的音频
    sentence_audio = []
    for audio_chunk, _ in audio_stream:
        if audio_chunk is not None and len(audio_chunk) > 0:
            sentence_audio.append(audio_chunk)
    
    # 一次性返回完整句子的音频
    if sentence_audio:
        full_audio = np.concatenate(sentence_audio)
        pcm16 = np.clip(full_audio, -1.0, 1.0)
        pcm16 = (pcm16 * 32767).astype(np.int16)
        yield pcm16.tobytes()


def audio_stream_generator(text: str, character: str = "default", nfe_step: int = None, cfg_strength: float = None):
    """流式生成器 - 返回 WAV 格式的音频流"""
    gen_start = time.time()
    req_id = f"{int(gen_start * 1000) % 100000}"

    logger.info(f"[{req_id}] Stream Request: '{text[:80]}{'...' if len(text) > 80 else ''}' (character: {character})")

    try:
        # 首先发送 WAV 文件头
        wav_header = create_wav_header(tts_model.sampling_rate)
        yield wav_header
        
        # 获取角色配置
        char_config = tts_model.get_character_config(character)
        ref_file = char_config["ref_file"]
        ref_text = char_config["ref_text"]

        # 加载参考音频
        ref_audio, processed_ref_text = preprocess_ref_audio_text(ref_file, ref_text)
        audio, sr = torchaudio.load(ref_audio)

        nfe = nfe_step or INFERENCE_CONFIG["nfe_step"]
        cfg = cfg_strength or INFERENCE_CONFIG["cfg_strength"]
        
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

            # 生成完整句子的音频
            for audio_bytes in generate_audio_for_sentence(sentence, (audio, sr), processed_ref_text, nfe, cfg):
                chunk_count += 1

                if not first_logged:
                    ttfa = (time.time() - gen_start) * 1000
                    logger.info(f"[{req_id}] ⚡ First audio at {ttfa:.0f}ms")
                    first_logged = True

                # 直接发送 PCM 数据（已经有 WAV 头了）
                yield audio_bytes

            logger.info(f"[{req_id}] Chunk[{i+1}/{len(sentences)}] done in {(time.time()-sent_start)*1000:.0f}ms")

        total_time = (time.time() - gen_start) * 1000
        logger.info(f"[{req_id}] ✅ Complete: {chunk_count} audio chunks in {total_time:.0f}ms")

    except Exception as e:
        logger.error(f"[{req_id}] ❌ Error: {e}", exc_info=True)
        raise


@app.post("/tts/stream")
def tts_stream(req: TTSRequest):
    """流式TTS API - 返回 WAV 格式音频流（Postman 可直接播放）"""
    if not req.text.strip():
        raise HTTPException(400, "Empty text")

    return StreamingResponse(
        audio_stream_generator(req.text, req.character, req.nfe_step, req.cfg_strength),
        media_type="audio/wav",  # 改为 audio/wav
        headers={
            "Content-Disposition": "inline; filename=stream.wav",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        },
    )


@app.post("/tts")
def tts_file(req: TTSRequest):
    """完整文件TTS API - 返回WAV文件"""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # 获取角色配置
    char_config = tts_model.get_character_config(req.character)
    ref_file = char_config["ref_file"]
    ref_text = char_config["ref_text"]
    
    # 检查参考音频文件是否存在
    if not os.path.exists(ref_file):
        raise HTTPException(status_code=400, detail=f"Reference audio file does not exist: {ref_file}")
    
    start_time = time.time()
    
    # 使用参数
    nfe = req.nfe_step or INFERENCE_CONFIG["nfe_step"]
    cfg = req.cfg_strength or INFERENCE_CONFIG["cfg_strength"]
    speed = req.speed or 1.0
    
    logger.info(f"File generation: text='{req.text[:80]}...', character={req.character}, nfe={nfe}, cfg={cfg}, speed={speed}")
    
    # 调用 F5-TTS API
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

    # 返回生成的音频文件
    return FileResponse(OUTPUT_WAV, media_type="audio/wav", filename="output.wav")


@app.get("/characters")
def list_characters():
    """列出所有可用角色"""
    tts_model.load_characters()
    return {
        "characters": list(tts_model.characters.keys()),
        "default": "default"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "config": INFERENCE_CONFIG,
        "characters_loaded": len(tts_model.characters)
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