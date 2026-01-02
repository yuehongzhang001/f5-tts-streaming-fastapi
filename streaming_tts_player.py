import requests
import pyaudio
import time
from datetime import datetime
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class AudioConfig:
    """Audio configuration"""
    sample_rate: int = 24000
    channels: int = 1
    sample_width: int = 2  # 16-bit = 2 bytes
    format: int = pyaudio.paInt16
    frames_per_buffer: int = 2048


@dataclass
class StreamStats:
    """Stream playback statistics"""
    connection_time_ms: float = 0
    first_chunk_time_ms: float = 0
    first_play_time_ms: float = 0
    total_time_ms: float = 0
    total_chunks: int = 0
    total_bytes: int = 0
    audio_duration_s: float = 0
    rtf: float = 0  # Real-time Factor
    
    def __str__(self):
        return (
            f"Statistics:\n"
            f"  - Connection time: {self.connection_time_ms:.1f}ms\n"
            f"  - Time to first chunk: {self.first_chunk_time_ms:.1f}ms\n"
            f"  - Time to first play: {self.first_play_time_ms:.1f}ms\n"
            f"  - Total time: {self.total_time_ms:.1f}ms ({self.total_time_ms/1000:.2f}s)\n"
            f"  - Total chunks: {self.total_chunks}\n"
            f"  - Total data: {self.total_bytes/1024:.2f} KB\n"
            f"  - Audio duration: ~{self.audio_duration_s:.2f}s\n"
            f"  - Real-time Factor: {self.rtf:.2f}x {'⚡' if self.rtf < 1 else '🐌'}"
        )


class StreamingTTSPlayer:
    """Streaming TTS Audio Player"""
    
    def __init__(
        self, 
        base_url: str = "http://127.0.0.1:8000",
        audio_config: Optional[AudioConfig] = None,
        verbose: bool = True
    ):
        """
        Initialize player
        
        Args:
            base_url: TTS server address
            audio_config: Audio configuration, default configuration is used if None
            verbose: Whether to print detailed logs
        """
        self.base_url = base_url.rstrip('/')
        self.audio_config = audio_config or AudioConfig()
        self.verbose = verbose
        self.session = None
        
    def _log(self, message: str):
        """Print log"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] {message}")
    
    def _init_session(self):
        """Initialize HTTP session"""
        if self.session is None:
            self.session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=1,
                pool_maxsize=1,
                max_retries=0
            )
            self.session.mount('http://', adapter)
            self.session.mount('https://', adapter)
    
    def _skip_wav_header(self, response_iter, header_size: int = 44):
        """
        Skip WAV file header
        
        Args:
            response_iter: Response iterator
            header_size: WAV file header size (default 44 bytes)
            
        Yields:
            Audio data chunks
        """
        header_buffer = b''
        header_skipped = False
        
        for chunk in response_iter:
            if not chunk:
                continue
                
            if not header_skipped:
                header_buffer += chunk
                if len(header_buffer) >= header_size:
                    # Skip header, return remaining data
                    audio_data = header_buffer[header_size:]
                    header_skipped = True
                    if audio_data:
                        yield audio_data
                continue
            
            yield chunk

    def _play_audio_stream(self, response, stats: StreamStats, start_time: float, 
                          progress_callback: Optional[Callable[[int, int], None]] = None):
        """
        Internal method to handle audio stream playback
        
        Args:
            response: HTTP response object
            stats: StreamStats object to update
            start_time: Time when request started
            progress_callback: Progress callback function
        """
        p = None
        stream = None
        
        try:
            # Initialize audio stream
            p = pyaudio.PyAudio()
            stream = p.open(
                format=self.audio_config.format,
                channels=self.audio_config.channels,
                rate=self.audio_config.sample_rate,
                output=True,
                frames_per_buffer=self.audio_config.frames_per_buffer
            )
            
            self._log(f"🔊 Audio stream ready ({self.audio_config.sample_rate}Hz, {self.audio_config.channels}ch)")
            
            # Play audio
            last_log_time = time.time()
            empty_reads = 0
            max_empty_reads = 20
            
            for audio_chunk in self._skip_wav_header(
                response.iter_content(chunk_size=4096)
            ):
                if not audio_chunk:
                    empty_reads += 1
                    if empty_reads > max_empty_reads:
                        break
                    continue
                
                empty_reads = 0
                
                # Record first audio chunk
                if stats.total_chunks == 0:
                    stats.first_chunk_time_ms = (time.time() - start_time) * 1000
                    self._log(f"🎵 First chunk received (TTFC: {stats.first_chunk_time_ms:.1f}ms)")
                
                # Play audio
                stream.write(audio_chunk)
                
                # Record first playback
                if stats.total_chunks == 0:
                    stats.first_play_time_ms = (time.time() - start_time) * 1000
                    self._log(f"🔊 First audio played (TTFP: {stats.first_play_time_ms:.1f}ms)")
                
                stats.total_chunks += 1
                stats.total_bytes += len(audio_chunk)
                
                # Progress callback
                if progress_callback:
                    elapsed_ms = (time.time() - start_time) * 1000
                    progress_callback(stats.total_bytes, int(elapsed_ms))
                
                # Periodic progress output
                current_time = time.time()
                if current_time - last_log_time >= 1.0:
                    elapsed = (current_time - start_time) * 1000
                    self._log(
                        f"📊 Progress: {stats.total_chunks} chunks, "
                        f"{stats.total_bytes/1024:.1f}KB, {elapsed:.0f}ms"
                    )
                    last_log_time = current_time
            
            # Wait for playback to complete
            time.sleep(0.1)
            
            # Calculate statistics
            stats.total_time_ms = (time.time() - start_time) * 1000
            stats.audio_duration_s = stats.total_bytes / (
                self.audio_config.sample_rate * self.audio_config.sample_width
            )
            stats.rtf = (stats.total_time_ms / 1000) / stats.audio_duration_s if stats.audio_duration_s > 0 else 0
            
            self._log(f"✨ Playback finished!")
            self._log(str(stats))
            
            return stats
            
        except KeyboardInterrupt:
            self._log(f"⏸️  Interrupted by user")
            raise
        except requests.exceptions.Timeout:
            self._log(f"❌ Request timeout")
            raise
        except requests.exceptions.ConnectionError as e:
            self._log(f"❌ Connection error: {e}")
            raise
        except Exception as e:
            self._log(f"❌ Error: {e}")
            raise
        finally:
            # Clean up resources
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass
            if p:
                try:
                    p.terminate()
                except:
                    pass
    
    def play(
        self, 
        text: str, 
        character: str = "default",
        nfe_step: Optional[int] = None,
        cfg_strength: Optional[float] = None,
        speed: float = 1.0,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> StreamStats:
        """
        Play text-to-speech
        
        Args:
            text: Text to convert
            character: Character name
            nfe_step: Inference steps (optional)
            cfg_strength: CFG strength (optional)
            speed: Speed factor (optional)
            progress_callback: Progress callback function callback(current_bytes, elapsed_ms)
            
        Returns:
            StreamStats: Playback statistics
        """
        stats = StreamStats()
        start_time = time.time()
        
        try:
            self._log(f"📝 Text: {text[:80]}{'...' if len(text) > 80 else ''}")
            self._log(f"👤 Character: {character}")
            self._log(f"🚀 Connecting to {self.base_url}")
            
            # Initialize session
            self._init_session()
            
            # Prepare request data
            request_data = {
                "text": text,
                "character": character,
                "speed": speed
            }
            if nfe_step is not None:
                request_data["nfe_step"] = nfe_step
            if cfg_strength is not None:
                request_data["cfg_strength"] = cfg_strength
            
            # Send request
            request_start = time.time()
            response = self.session.post(
                f"{self.base_url}/tts/stream",
                json=request_data,
                stream=True,
                timeout=(3, 60),
                headers={
                    'Connection': 'keep-alive',
                    'Accept-Encoding': 'identity',
                }
            )
            stats.connection_time_ms = (time.time() - request_start) * 1000
            
            self._log(f"✅ Connected (took {stats.connection_time_ms:.1f}ms)")
            
            if response.status_code != 200:
                self._log(f"❌ Error: HTTP {response.status_code}")
                try:
                    error_detail = response.json()
                    self._log(f"   Error detail: {error_detail}")
                except:
                    error_detail = response.text[:500]
                    self._log(f"   Response: {error_detail}")
                raise Exception(f"Server returned status {response.status_code}: {error_detail}")
            
            # Play audio
            self._play_audio_stream(response, stats, start_time, progress_callback)
            
            return stats
            
        except KeyboardInterrupt:
            self._log(f"⏸️  Interrupted by user")
            raise
        except requests.exceptions.Timeout:
            self._log(f"❌ Request timeout")
            raise
        except requests.exceptions.ConnectionError as e:
            self._log(f"❌ Connection error: {e}")
            raise
        except Exception as e:
            self._log(f"❌ Error: {e}")
            raise

    def play_multi(
        self, 
        text_list: list, 
        nfe_step: Optional[int] = None,
        cfg_strength: Optional[float] = None,
        speed: float = 1.0,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> StreamStats:
        """
        Play multi-character text-to-speech
        
        Args:
            text_list: List of dicts with "text" and "character" keys
            nfe_step: Inference steps (optional)
            cfg_strength: CFG strength (optional)
            speed: Speed factor (optional)
            progress_callback: Progress callback function callback(current_bytes, elapsed_ms)
            
        Returns:
            StreamStats: Playback statistics
        """
        stats = StreamStats()
        start_time = time.time()
        
        try:
            self._log(f"📝 Multi-character text: {len(text_list)} segments")
            for i, item in enumerate(text_list):
                self._log(f"   [{i+1}] Character: {item.get('character', 'default')}, Text: {item.get('text', '')[:60]}{'...' if len(item.get('text', '')) > 60 else ''}")
            
            self._log(f"🚀 Connecting to {self.base_url}")
            
            # Initialize session
            self._init_session()
            
            # Prepare request data
            request_data = {
                "text_list": text_list,
                "speed": speed
            }
            if nfe_step is not None:
                request_data["nfe_step"] = nfe_step
            if cfg_strength is not None:
                request_data["cfg_strength"] = cfg_strength
            
            # Send request
            request_start = time.time()
            response = self.session.post(
                f"{self.base_url}/tts/stream-multi",
                json=request_data,
                stream=True,
                timeout=(3, 60),
                headers={
                    'Connection': 'keep-alive',
                    'Accept-Encoding': 'identity',
                }
            )
            stats.connection_time_ms = (time.time() - request_start) * 1000
            
            self._log(f"✅ Connected (took {stats.connection_time_ms:.1f}ms)")
            
            if response.status_code != 200:
                self._log(f"❌ Error: HTTP {response.status_code}")
                try:
                    error_detail = response.json()
                    self._log(f"   Error detail: {error_detail}")
                except:
                    error_detail = response.text[:500]
                    self._log(f"   Response: {error_detail}")
                raise Exception(f"Server returned status {response.status_code}: {error_detail}")
            
            # Play audio stream
            self._play_audio_stream(response, stats, start_time, progress_callback)
            
            return stats
            
        except KeyboardInterrupt:
            self._log(f"⏸️  Interrupted by user")
            raise
        except requests.exceptions.Timeout:
            self._log(f"❌ Request timeout")
            raise
        except requests.exceptions.ConnectionError as e:
            self._log(f"❌ Connection error: {e}")
            raise
        except Exception as e:
            self._log(f"❌ Error: {e}")
            raise

    def close(self):
        """Close session and clean up resources"""
        if self.session:
            self.session.close()
            self.session = None
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False


def play_tts(
    text: str,
    character: str = "default",
    base_url: str = "http://127.0.0.1:8000",
    verbose: bool = True,
    speed: float = 1.0
) -> StreamStats:
    """
    Convenience function: Play single text
    
    Args:
        text: Text to convert
        character: Character name
        base_url: TTS server address
        verbose: Whether to print detailed logs
        speed: Speed factor
        
    Returns:
        StreamStats: Playback statistics
    """
    player = StreamingTTSPlayer(base_url=base_url, verbose=verbose)
    try:
        return player.play(text, character, speed=speed)
    finally:
        player.close()


def play_tts_multi(
    text_list: list,
    base_url: str = "http://127.0.0.1:8000",
    verbose: bool = True,
    speed: float = 1.0
) -> StreamStats:
    """
    Convenience function: Play multi-character text
    
    Args:
        text_list: List of dicts with "text" and "character" keys
        base_url: TTS server address
        verbose: Whether to print detailed logs
        speed: Speed factor
        
    Returns:
        StreamStats: Playback statistics
    """
    player = StreamingTTSPlayer(base_url=base_url, verbose=verbose)
    try:
        return player.play_multi(text_list, speed=speed)
    finally:
        player.close()


# ============================================================================
# Usage examples
# ============================================================================

if __name__ == "__main__":
    
    # Method 1: Using convenience function (recommended for single playback)
    print("="*80)
    print("Example 1: Using convenience function")
    print("="*80 + "\n")
    
    stats = play_tts(
        text="Hello, this is a simple test.",
        character="default"
    )
    print(f"\nReturned statistics: RTF = {stats.rtf:.2f}x\n")
    
    
    # Method 2: Using player instance (recommended for multiple playbacks)
    print("\n" + "="*80)
    print("Example 2: Using player instance to play multiple texts")
    print("="*80 + "\n")
    
    with StreamingTTSPlayer(base_url="http://127.0.0.1:8000", verbose=True) as player:
        
        texts = [
            "First test text.",
            "Second test text, slightly longer to test the effect of streaming playback.",
            "Third test text to demonstrate multiple sequential playbacks."
        ]
        
        for i, text in enumerate(texts, 1):
            print(f"\n{'─'*80}")
            print(f"Playing {i}/{len(texts)}")
            print(f"{'─'*80}\n")
            
            stats = player.play(text, character="default")
            
            if i < len(texts):
                print(f"\n⏳ Waiting 1 second...\n")
                time.sleep(1)
    
    
    # Method 3: With progress callback
    print("\n" + "="*80)
    print("Example 3: Using progress callback")
    print("="*80 + "\n")
    
    def my_progress_callback(bytes_received, elapsed_ms):
        """Custom progress callback"""
        # Here you can update UI progress bar, etc.
        pass
    
    with StreamingTTSPlayer(verbose=True) as player:
        player.play(
            text="This is an example with progress callback.",
            character="default",
            progress_callback=my_progress_callback
        )
    
    
    # Method 4: Custom audio configuration
    print("\n" + "="*80)
    print("Example 4: Custom audio configuration")
    print("="*80 + "\n")
    
    custom_config = AudioConfig(
        sample_rate=24000,
        channels=1,
        sample_width=2,
        format=pyaudio.paInt16,
        frames_per_buffer=4096  # Larger buffer
    )
    
    with StreamingTTSPlayer(audio_config=custom_config, verbose=True) as player:
        player.play("Playback using custom audio configuration.")
    
    
    # Method 5: Multi-character dialogue
    print("\n" + "="*80)
    print("Example 5: Multi-character dialogue")
    print("="*80 + "\n")
    
    dialogue = [
        {
            "text": "你今天到底怎么回事啊？电话不接消息不回，急死我了！",
            "character": "female_2"
        },
        {
            "text": "我开会呢！不是说了今天项目汇报吗？你明明知道的呀。",
            "character": "male_1"
        },
        {
            "text": "那也总该抽空看一眼手机吧？",
            "character": "female_2"
        },
        {
            "text": "你最近怎么这么黏人啊？",
            "character": "male_1"
        }
    ]
    
    stats = play_tts_multi(dialogue)
    print(f"\nReturned statistics: RTF = {stats.rtf:.2f}x\n")
    
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)