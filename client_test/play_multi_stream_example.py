import json
import os
from streaming_tts_player import play_tts_multi


def play_multi_from_json(json_file: str, base_url: str = "http://127.0.0.1:8000"):
    """
    从 JSON 文件读取多角色对话并播放
    
    JSON 格式:
    [
        {
            "text": "第一个角色的文本",
            "character": "角色名称"
        },
        {
            "text": "第二个角色的文本",
            "character": "另一个角色名称"
        }
    ]
    
    Args:
        json_file: JSON 文件路径
        base_url: TTS 服务器地址
    """
    # 读取 JSON 文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 验证数据格式
    if not isinstance(data, list):
        print("❌ 错误: JSON 文件应包含一个对话列表")
        return
    
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "text" not in item or "character" not in item:
            print(f"❌ 错误: 第 {i+1} 项格式不正确，应包含 'text' 和 'character' 字段")
            return
        
        if not item["text"].strip():
            print(f"❌ 错误: 第 {i+1} 项的文本字段不能为空")
            return
    
    if not data:
        print("❌ 错误: JSON 文件中没有对话内容")
        return
    
    # 显示对话内容
    print("📝 对话内容:")
    for i, item in enumerate(data):
        text_preview = item["text"][:100] + ("..." if len(item["text"]) > 100 else "")
        print(f"   [{i+1}] 角色: {item['character']}, 文本: {text_preview}")
    print()
    
    # 播放多角色对话
    try:
        stats = play_tts_multi(data, base_url=base_url)
        print(f"✅ 播放完成!")
        print(f"📊 统计信息: RTF = {stats.rtf:.2f}x")
    except Exception as e:
        print(f"❌ 播放失败: {e}")


if __name__ == "__main__":
    # 使用示例
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    play_multi_from_json(os.path.join(repo_root, "json", "multi_stream_input.json"))
