"""
F5-TTS FastAPI 服务
处理HTTP请求并调用TTS引擎
"""
import logging
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tts_engine import TTSEngine, TTSConfig

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s', 
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 初始化应用
app = FastAPI(title="F5-TTS Unified API")
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# 初始化TTS引擎
config = TTSConfig()
tts_engine = TTSEngine(config)


# ==================== 请求模型 ====================

class TTSRequest(BaseModel):
    """单角色TTS请求"""
    text: str
    character: str = "default"
    nfe_step: int = None
    cfg_strength: float = None
    speed: float = 1.0


class MultiCharacterTTSRequest(BaseModel):
    """多角色TTS请求"""
    text_list: List[Dict[str, Any]]  # [{"text": "...", "character": "..."}]
    nfe_step: int = None
    cfg_strength: float = None
    speed: float = 1.0


# ==================== 事件处理 ====================

@app.on_event("startup")
async def startup():
    """应用启动时加载模型"""
    logger.info("Starting F5-TTS API Server...")
    tts_engine.load()
    logger.info("F5-TTS API Server is ready!")


# ==================== API端点 ====================

@app.post("/tts/stream")
def tts_stream(req: TTSRequest):
    """
    流式TTS API
    
    Args:
        req: TTS请求参数
    
    Returns:
        音频流响应
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        return StreamingResponse(
            tts_engine.stream_generate(
                text=req.text,
                character=req.character,
                nfe_step=req.nfe_step,
                cfg_strength=req.cfg_strength,
                speed=req.speed
            ),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=stream.wav",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Stream generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/tts/stream-multi")
def tts_stream_multi(req: MultiCharacterTTSRequest):
    """
    多角色流式TTS API
    
    Args:
        req: 多角色TTS请求参数
    
    Returns:
        音频流响应
    """
    if not req.text_list:
        raise HTTPException(status_code=400, detail="text_list cannot be empty")
    
    # 验证请求格式
    for idx, item in enumerate(req.text_list):
        if "text" not in item or not item["text"].strip():
            raise HTTPException(
                status_code=400, 
                detail=f"Item {idx} in text_list must have a non-empty text field"
            )
        if "character" not in item:
            raise HTTPException(
                status_code=400, 
                detail=f"Item {idx} in text_list must have a character field"
            )

    try:
        return StreamingResponse(
            tts_engine.stream_generate_multi_character(
                text_list=req.text_list,
                nfe_step=req.nfe_step,
                cfg_strength=req.cfg_strength,
                speed=req.speed
            ),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=multi_stream.wav",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Multi-character stream generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/tts")
def tts_file(req: TTSRequest):
    """
    完整文件TTS API
    
    Args:
        req: TTS请求参数
    
    Returns:
        音频文件响应
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        output_path = tts_engine.generate_file(
            text=req.text,
            character=req.character,
            nfe_step=req.nfe_step,
            cfg_strength=req.cfg_strength,
            speed=req.speed
        )
        
        return FileResponse(
            output_path, 
            media_type="audio/wav", 
            filename="output.wav"
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"File generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/characters")
def list_characters():
    """
    列出所有可用角色
    
    Returns:
        角色列表和默认角色
    """
    try:
        # 重新加载角色配置（支持热更新）
        tts_engine.character_manager.load_characters()
        
        characters = tts_engine.character_manager.list_characters()
        default = tts_engine.character_manager.get_default_character()
        
        return {
            "characters": characters,
            "default": default
        }
    except Exception as e:
        logger.error(f"Error listing characters: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list characters: {str(e)}")


@app.get("/health")
def health():
    """
    健康检查端点
    
    Returns:
        服务状态信息
    """
    try:
        characters = tts_engine.character_manager.list_characters()
        
        return {
            "status": "ok",
            "device": config.DEVICE,
            "config": config.INFERENCE_CONFIG,
            "characters_loaded": len(characters),
            "available_characters": characters
        }
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app, 
        host="0.0.0.0",
        port=8000, 
        log_level="info",
        access_log=False,
    )