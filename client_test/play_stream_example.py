import json
import os
from streaming_tts_player import play_tts


def play_from_json(json_file: str):
    """
    从 JSON 文件读取文本并播放
    
    JSON 格式:
    {
        "text": "要播放的文本",
        "character": "default"  (可选)
    }
    
    Args:
        json_file: JSON 文件路径
    """
    # 读取 JSON 文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    text = data.get("text", "")
    character = data.get("character", "default")
    
    if not text:
        print("❌ 错误: JSON 文件中没有 text 字段或为空")
        return
    
    # 播放
    print(f"📝 文本: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"👤 角色: {character}")
    print()
    
    play_tts(text, character)


if __name__ == "__main__":
    # 使用示例
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    play_from_json(os.path.join(repo_root, "json", "stream_input.json"))
