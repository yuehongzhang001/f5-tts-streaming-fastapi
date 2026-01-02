"""
F5-TTS 音频生成引擎
负责模型加载、音频生成和文本处理
"""
import os
import logging
import re
import time
import json
import struct
from importlib.resources import files
from typing import List, Dict, Tuple, Generator, Any

import numpy as np
import torch
import torchaudio
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

logger = logging.getLogger(__name__)


class TTSConfig:
    """TTS配置类"""
    MODEL_NAME = "F5TTS_v1_Base"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
    
    # 推理配置
    INFERENCE_CONFIG = {
        "nfe_step": 32,
        "cfg_strength": 2.5,
        "sway_sampling_coef": -1.0,
        "chunk_size": 512,
    }
    
    # 文件路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHARACTERS_FILE = os.path.join(BASE_DIR, "json/characters.json")
    OUTPUT_WAV = os.path.join(BASE_DIR, "output.wav")


def create_wav_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
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


def smart_split(text: str, first_chunk_max: int = 135, regular_chunk_max: int = 135) -> List[str]:
    """
    智能分句 - 基于F5-TTS官方实现，使用UTF-8字节计数
    
    Args:
        text: 输入文本
        first_chunk_max: 首句最大字节数
        regular_chunk_max: 后续句子最大字节数
    
    Returns:
        分割后的句子列表
    """
    text = re.sub(r'\s+', ' ', text.strip())
    
    chunks = []
    current_chunk = ""
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # 关键：使用UTF-8字节数而非字符数
        current_bytes = len(current_chunk.encode("utf-8"))
        sentence_bytes = len(sentence.encode("utf-8"))
        max_bytes = first_chunk_max if not chunks else regular_chunk_max
        
        if current_bytes + sentence_bytes <= max_bytes:
            if current_chunk and len(sentence[0].encode("utf-8")) == 1:
                current_chunk += " " + sentence
            else:
                current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return [c for c in chunks if len(c) >= 2]


class CharacterManager:
    """角色配置管理器"""
    
    def __init__(self, characters_file: str):
        self.characters_file = characters_file
        self.characters: Dict[str, Dict[str, str]] = {}
        self.load_characters()
    
    def load_characters(self) -> None:
        """加载角色配置"""
        if not os.path.exists(self.characters_file):
            logger.error(f"❌ Characters file not found: {self.characters_file}")
            logger.error("Please create a characters.json file with at least one character configuration")
            raise FileNotFoundError(f"Required file not found: {self.characters_file}")
        
        try:
            with open(self.characters_file, 'r', encoding='utf-8') as f:
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
    
    def get_character_config(self, character: str = None) -> Dict[str, str]:
        """
        获取角色配置
        
        Args:
            character: 角色名称，None或"default"使用第一个角色
        
        Returns:
            角色配置字典
        
        Raises:
            ValueError: 角色不存在
        """
        # 如果没有指定角色或指定为 default，使用第一个角色
        if not character or character == "default":
            if not self.characters:
                raise ValueError("No characters configured")
            first_character = list(self.characters.keys())[0]
            logger.info(f"Using default character: {first_character}")
            return self.characters[first_character]
        
        # 重新加载配置（支持热更新）
        if os.path.exists(self.characters_file):
            try:
                with open(self.characters_file, 'r', encoding='utf-8') as f:
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
            raise ValueError(f"Character '{character}' not found. Available characters: {available}")
    
    def list_characters(self) -> List[str]:
        """返回所有可用角色名称"""
        return list(self.characters.keys())
    
    def get_default_character(self) -> str:
        """返回默认角色名称"""
        return list(self.characters.keys())[0] if self.characters else None


class TTSEngine:
    """TTS音频生成引擎"""
    
    def __init__(self, config: TTSConfig = None):
        self.config = config or TTSConfig()
        self.model = None
        self.vocoder = None
        self.sampling_rate = None
        self.f5tts_api = None
        self.character_manager = CharacterManager(self.config.CHARACTERS_FILE)
        self.warmup_audio = None
        self.warmup_sr = None
    
    def load(self) -> None:
        """加载模型和相关资源"""
        logger.info(f"Loading on {self.config.DEVICE} ({self.config.DTYPE})...")
        start = time.time()

        model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{self.config.MODEL_NAME}.yaml")))
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        model_arc = model_cfg.model.arch
        mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
        self.sampling_rate = model_cfg.model.mel_spec.target_sample_rate

        ckpt_file = str(hf_hub_download(repo_id="SWivid/F5-TTS", filename="F5TTS_v1_Base/model_1250000.safetensors"))

        self.model = load_model(
            model_cls, model_arc, ckpt_path=ckpt_file, mel_spec_type=mel_spec_type,
            vocab_file="", ode_method="euler", use_ema=True, device=self.config.DEVICE,
        ).to(self.config.DEVICE, dtype=self.config.DTYPE)

        self.model.eval()
        torch.set_grad_enabled(False)

        if self.config.DEVICE == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("✅ Model compiled")
            except:
                pass

        self.vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=False, local_path=None, device=self.config.DEVICE)
        self.vocoder.eval()

        logger.info("Loading F5TTS API for file generation...")
        self.f5tts_api = F5TTS()

        # 预热模型
        self._warmup_model()

        logger.info(f"✅ Ready in {time.time()-start:.1f}s!")
    
    def _warmup_model(self) -> None:
        """使用第一个可用角色进行模型预热"""
        characters = self.character_manager.list_characters()
        if not characters:
            logger.warning("⚠️  No characters configured, skipping warmup")
            return
        
        first_character = characters[0]
        char_config = self.character_manager.get_character_config(first_character)
        
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
                    device=self.config.DEVICE, 
                    streaming=True,
                    chunk_size=self.config.INFERENCE_CONFIG["chunk_size"], 
                    nfe_step=self.config.INFERENCE_CONFIG["nfe_step"],
                ):
                    pass
            
            logger.info("✅ Warmup completed")
        except Exception as e:
            logger.warning(f"⚠️  Warmup failed: {e}")
    
    def generate_audio_for_sentence(
        self, 
        sentence: str, 
        ref_audio_tuple: Tuple, 
        ref_text: str, 
        nfe: int, 
        cfg_strength: float, 
        speed: float = 1.0
    ) -> Generator[bytes, None, None]:
        """
        为单个句子生成音频
        
        Args:
            sentence: 输入句子
            ref_audio_tuple: 参考音频元组 (audio, sample_rate)
            ref_text: 参考文本
            nfe: NFE步数
            cfg_strength: CFG强度
            speed: 语速
        
        Yields:
            音频字节流
        """
        audio_stream = infer_batch_process(
            ref_audio_tuple, ref_text, [sentence],
            self.model, self.vocoder, progress=None, device=self.config.DEVICE,
            streaming=True, chunk_size=self.config.INFERENCE_CONFIG["chunk_size"], nfe_step=nfe,
            cfg_strength=cfg_strength,
            sway_sampling_coef=self.config.INFERENCE_CONFIG["sway_sampling_coef"],
            speed=speed,
        )

        sentence_audio = []
        for audio_chunk, _ in audio_stream:
            if audio_chunk is not None and len(audio_chunk) > 0:
                sentence_audio.append(audio_chunk)
        
        if sentence_audio:
            full_audio = np.concatenate(sentence_audio)
            
            # 归一化，避免削波失真
            peak = np.abs(full_audio).max()
            if peak > 0:
                full_audio = full_audio * (0.95 / peak)
            
            # 添加淡入淡出，减少爆音
            fade_samples = min(100, len(full_audio) // 10)
            if fade_samples > 0:
                fade_in = np.linspace(0, 1, fade_samples)
                full_audio[:fade_samples] *= fade_in
                
                fade_out = np.linspace(1, 0, fade_samples)
                full_audio[-fade_samples:] *= fade_out
            
            # 转换为PCM16
            pcm16 = np.clip(full_audio, -1.0, 1.0)
            pcm16 = np.round(pcm16 * 32767).astype(np.int16)
            
            yield pcm16.tobytes()
    
    def stream_generate(
        self, 
        text: str, 
        character: str = "default", 
        nfe_step: int = None, 
        cfg_strength: float = None, 
        speed: float = 1.0
    ) -> Generator[bytes, None, None]:
        """
        流式生成音频
        
        Args:
            text: 输入文本
            character: 角色名称
            nfe_step: NFE步数
            cfg_strength: CFG强度
            speed: 语速
        
        Yields:
            音频字节流（包含WAV头）
        """
        gen_start = time.time()
        req_id = f"{int(gen_start * 1000) % 100000}"

        logger.info(f"[{req_id}] Stream Request: '{text[:80]}{'...' if len(text) > 80 else ''}' (character: {character})")

        try:
            # 生成WAV头
            wav_header = create_wav_header(self.sampling_rate)
            yield wav_header
            
            # 获取角色配置
            char_config = self.character_manager.get_character_config(character)
            ref_file = char_config["ref_file"]
            ref_text = char_config["ref_text"]

            # 预处理参考音频
            ref_audio, processed_ref_text = preprocess_ref_audio_text(ref_file, ref_text)
            audio, sr = torchaudio.load(ref_audio)

            nfe = nfe_step if nfe_step is not None else self.config.INFERENCE_CONFIG["nfe_step"]
            cfg = cfg_strength if cfg_strength is not None else self.config.INFERENCE_CONFIG["cfg_strength"]
            
            logger.info(f"[{req_id}] Using character='{character}', nfe_step={nfe}, cfg_strength={cfg}, speed={speed}")
            
            # 分割文本
            sentences = smart_split(text, first_chunk_max=135, regular_chunk_max=135)

            logger.info(f"[{req_id}] Split into {len(sentences)} chunks")
            for i, s in enumerate(sentences):
                logger.info(f"[{req_id}]   Chunk[{i+1}]: '{s[:60]}{'...' if len(s) > 60 else ''}' ({len(s)} chars)")

            chunk_count = 0
            first_logged = False

            # 生成音频
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue

                sent_start = time.time()

                for audio_bytes in self.generate_audio_for_sentence(sentence, (audio, sr), processed_ref_text, nfe, cfg, speed=speed):
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
    
    def stream_generate_multi_character(
        self, 
        text_list: List[Dict[str, Any]], 
        nfe_step: int = None, 
        cfg_strength: float = None, 
        speed: float = 1.0
    ) -> Generator[bytes, None, None]:
        """
        多角色流式生成音频
        
        Args:
            text_list: 文本列表，每项包含 {"text": "...", "character": "..."}
            nfe_step: NFE步数
            cfg_strength: CFG强度
            speed: 语速
        
        Yields:
            音频字节流（包含WAV头）
        """
        gen_start = time.time()
        req_id = f"{int(gen_start * 1000) % 100000}"

        logger.info(f"[{req_id}] Multi-character Stream Request: {len(text_list)} segments")

        try:
            # 生成WAV头
            wav_header = create_wav_header(self.sampling_rate)
            yield wav_header

            nfe = nfe_step if nfe_step is not None else self.config.INFERENCE_CONFIG["nfe_step"]
            cfg = cfg_strength if cfg_strength is not None else self.config.INFERENCE_CONFIG["cfg_strength"]

            for idx, item in enumerate(text_list):
                text = item.get("text", "")
                character = item.get("character", "default")
                
                if not text.strip():
                    logger.warning(f"[{req_id}] Skipping empty text at index {idx}")
                    continue

                logger.info(f"[{req_id}] Processing segment {idx+1}/{len(text_list)}: character='{character}', text='{text[:50]}{'...' if len(text) > 50 else ''}'")

                # 获取角色配置
                char_config = self.character_manager.get_character_config(character)
                ref_file = char_config["ref_file"]
                ref_text = char_config["ref_text"]

                # 预处理参考音频
                ref_audio, processed_ref_text = preprocess_ref_audio_text(ref_file, ref_text)
                audio, sr = torchaudio.load(ref_audio)

                logger.info(f"[{req_id}] Using character='{character}', nfe_step={nfe}, cfg_strength={cfg}, speed={speed}")
                
                # 分割文本
                sentences = smart_split(text, first_chunk_max=135, regular_chunk_max=135)

                logger.info(f"[{req_id}] Split into {len(sentences)} chunks")
                for i, s in enumerate(sentences):
                    logger.info(f"[{req_id}]   Chunk[{i+1}]: '{s[:60]}{'...' if len(s) > 60 else ''}' ({len(s)} chars)")

                chunk_count = 0
                first_logged = False

                # 生成音频
                for i, sentence in enumerate(sentences):
                    if not sentence.strip():
                        continue

                    sent_start = time.time()

                    for audio_bytes in self.generate_audio_for_sentence(sentence, (audio, sr), processed_ref_text, nfe, cfg, speed=speed):
                        chunk_count += 1

                        if not first_logged:
                            ttfa = (time.time() - gen_start) * 1000
                            logger.info(f"[{req_id}] ⚡ First audio at {ttfa:.0f}ms")
                            first_logged = True

                        yield audio_bytes

                    logger.info(f"[{req_id}] Segment[{idx+1}] Chunk[{i+1}/{len(sentences)}] done in {(time.time()-sent_start)*1000:.0f}ms")

            total_time = (time.time() - gen_start) * 1000
            logger.info(f"[{req_id}] ✅ Complete: Multi-character stream with total {total_time:.0f}ms")

        except Exception as e:
            logger.error(f"[{req_id}] ❌ Error: {e}", exc_info=True)
            raise
    
    def generate_file(
        self, 
        text: str, 
        character: str = "default", 
        nfe_step: int = None, 
        cfg_strength: float = None, 
        speed: float = 1.0,
        output_path: str = None
    ) -> str:
        """
        生成完整音频文件
        
        Args:
            text: 输入文本
            character: 角色名称
            nfe_step: NFE步数
            cfg_strength: CFG强度
            speed: 语速
            output_path: 输出文件路径
        
        Returns:
            生成的音频文件路径
        """
        char_config = self.character_manager.get_character_config(character)
        ref_file = char_config["ref_file"]
        ref_text = char_config["ref_text"]
        
        if not os.path.exists(ref_file):
            raise FileNotFoundError(f"Reference audio file does not exist: {ref_file}")
        
        start_time = time.time()
        
        nfe = nfe_step or self.config.INFERENCE_CONFIG["nfe_step"]
        cfg = cfg_strength or self.config.INFERENCE_CONFIG["cfg_strength"]
        output_path = output_path or self.config.OUTPUT_WAV
        
        logger.info(f"File generation: text='{text[:80]}...', character={character}, nfe={nfe}, cfg={cfg}, speed={speed}")
        
        wav, sr, spec = self.f5tts_api.infer(
            ref_file=ref_file,
            ref_text=ref_text,
            gen_text=text,
            cfg_strength=cfg,
            nfe_step=nfe,
            speed=speed,
            file_wave=output_path,
            file_spec=None
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"Generated WAV file in {processing_time:.2f}s using character '{character}'")

        return output_path