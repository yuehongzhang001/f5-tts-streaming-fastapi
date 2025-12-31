import requests
import pyaudio
import time
from datetime import datetime
import socket

def play_streaming_audio_optimized(text: str, character: str = "default", url: str = "http://127.0.0.1:8000/tts/stream"):
    """优化的实时播放 - 使用更底层的连接控制"""
    
    def log(message):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] {message}")
    
    # 初始化 PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=24000,
        output=True,
        frames_per_buffer=1024  # ⚡ 减小缓冲区
    )
    
    start_time = time.time()
    request_sent_time = None
    first_chunk_received_time = None
    first_audio_played_time = None
    
    try:
        log(f"📝 Text: {text[:80]}{'...' if len(text) > 80 else ''}")
        log(f"👤 Character: {character}")
        log(f"🚀 Sending request to {url}")
        
        # ⚡ 使用 Session 以复用连接
        session = requests.Session()
        
        # ⚡ 禁用连接池延迟
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1,
            pool_maxsize=1,
            max_retries=0
        )
        session.mount('http://', adapter)
        
        # 发送请求
        request_start = time.time()
        response = session.post(
            url,
            json={"text": text, "character": character},
            stream=True,
            timeout=(1, 30),  # ⚡ (连接超时, 读取超时)
            headers={
                'Connection': 'keep-alive',
                'Accept-Encoding': 'identity',  # ⚡ 禁用压缩
            }
        )
        request_sent_time = time.time()
        
        connection_time = (request_sent_time - request_start) * 1000
        log(f"✅ Connection established (took {connection_time:.1f}ms)")
        
        if response.status_code != 200:
            log(f"❌ Error: HTTP {response.status_code}")
            return
        
        log(f"📡 Streaming started, waiting for audio data...")
        
        chunk_count = 0
        total_bytes = 0
        empty_reads = 0
        
        # ⚡ 使用更小的读取块
        for chunk in response.iter_content(chunk_size=2048):  # 从 4096 降到 2048
            if not chunk:
                empty_reads += 1
                if empty_reads > 10:
                    break
                continue
            
            empty_reads = 0
            
            # 记录第一个音频块
            if first_chunk_received_time is None:
                first_chunk_received_time = time.time()
                ttfc = (first_chunk_received_time - start_time) * 1000
                log(f"🎵 First chunk received! (TTFC: {ttfc:.1f}ms)")
            
            # 播放音频
            stream.write(chunk)
            
            # 记录第一次播放
            if first_audio_played_time is None:
                first_audio_played_time = time.time()
                ttfp = (first_audio_played_time - start_time) * 1000
                log(f"🔊 First audio played! (TTFP: {ttfp:.1f}ms)")
                log(f"   ⏱️  Request → First Play: {ttfp:.1f}ms")
            
            chunk_count += 1
            total_bytes += len(chunk)
            
            # 每 10 个 chunk 输出进度
            if chunk_count % 10 == 0:
                elapsed = (time.time() - start_time) * 1000
                log(f"📊 Progress: {chunk_count} chunks, {total_bytes/1024:.1f}KB, {elapsed:.0f}ms elapsed")
        
        # 统计
        end_time = time.time()
        total_duration = (end_time - start_time) * 1000
        
        log(f"✨ Playback finished!")
        log(f"📈 Statistics:")
        log(f"   - Total chunks: {chunk_count}")
        log(f"   - Total data: {total_bytes/1024:.2f} KB")
        log(f"   - Total time: {total_duration:.1f}ms ({total_duration/1000:.2f}s)")
        
        if first_chunk_received_time and first_audio_played_time:
            log(f"   - Connection time: {connection_time:.1f}ms")
            log(f"   - Time to first chunk (TTFC): {(first_chunk_received_time - start_time)*1000:.1f}ms")
            log(f"   - Time to first play (TTFP): {(first_audio_played_time - start_time)*1000:.1f}ms")
            
            audio_duration = total_bytes / (24000 * 2)
            log(f"   - Audio duration: ~{audio_duration:.2f}s (estimated)")
            
            rtf = total_duration / 1000 / audio_duration if audio_duration > 0 else 0
            log(f"   - Real-time Factor (RTF): {rtf:.2f}x {'⚡' if rtf < 1 else '🐌'}")
        
    except KeyboardInterrupt:
        log(f"⏸️  Interrupted")
    except Exception as e:
        log(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        log(f"🧹 Resources cleaned up")


if __name__ == "__main__":
    test_cases = [
        ("“你今天到底怎么回事啊？电话不接消息不回，急死我了！”“我开会呢！不是说了今天项目汇报吗？你明明知道的呀。”“那也总该抽空看一眼手机吧？”“呸呸呸，能不能念我点好？你最近怎么这么黏人啊？”“我黏人？上周你说忙我三天都没打扰你！你一点都不想我是不是？”“想想想！但我也要搬砖啊宝贝，你当我是超人啊？”“你凶什么凶！我就是担心你嘛…”“哎…我错了。就是今天压力太大了，不该冲你发火的。”", "female_1")
        ]
    
    for i, (text, character) in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test Case {i}/{len(test_cases)}")
        print(f"{'='*80}\n")
        play_streaming_audio_optimized(text, character)
        
        if i < len(test_cases):
            print(f"\n⏳ Waiting 2 seconds...\n")
            time.sleep(2)