import pyaudio
import numpy as np
import scipy.signal
from typing import Optional, Callable, Generator
import threading
import queue
import time

DESIRED_RATE = 16000
CHUNK_SIZE = 1024
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1

class AudioInput:
    def __init__(
            self,
            input_device_index: int = None,
            debug_mode: bool = False,
            target_samplerate: int = DESIRED_RATE,
            chunk_size: int = CHUNK_SIZE,
            audio_format: int = AUDIO_FORMAT,
            channels: int = CHANNELS,
            resample_to_target: bool = True,
        ):

        self.input_device_index = input_device_index
        self.debug_mode = debug_mode
        self.audio_interface = None
        self.stream = None
        self.device_sample_rate = None
        self.target_samplerate = target_samplerate
        self.chunk_size = chunk_size
        self.audio_format = audio_format
        self.channels = channels
        self.resample_to_target = resample_to_target
        
        # Additional attributes for audio processing
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        self.resampler = None
        
        # Initialize PyAudio
        self._initialize_audio()

    def _initialize_audio(self):
        """Initialize PyAudio interface and set up device."""
        try:
            self.audio_interface = pyaudio.PyAudio()
            
            # Auto-detect best input device if none specified
            if self.input_device_index is None:
                self.input_device_index = self._find_best_input_device()
            
            # Get device sample rate
            device_info = self.audio_interface.get_device_info_by_index(self.input_device_index)
            self.device_sample_rate = int(device_info.get('defaultSampleRate', self.target_samplerate))
            
            if self.debug_mode:
                print(f"Selected input device: {device_info['name']}")
                print(f"Device sample rate: {self.device_sample_rate} Hz")
                print(f"Target sample rate: {self.target_samplerate} Hz")
                
        except Exception as e:
            print(f"Error initializing audio: {e}")
            raise

    def _find_best_input_device(self) -> int:
        """Find the best available input device."""
        default_device = self.audio_interface.get_default_input_device_info()
        device_index = default_device['index']
        
        if self.debug_mode:
            print(f"Using default input device: {default_device['name']}")
            
        return device_index

    def get_supported_sample_rates(self, device_index):
        """Test which standard sample rates are supported by the specified device."""
        standard_rates = [8000, 9600, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000]
        supported_rates = []

        device_info = self.audio_interface.get_device_info_by_index(device_index)
        max_channels = device_info.get('maxInputChannels')

        for rate in standard_rates:
            try:
                if self.audio_interface.is_format_supported(
                    rate,
                    input_device=device_index,
                    input_channels=max_channels,
                    input_format=self.audio_format,
                ):
                    supported_rates.append(rate)
            except:
                continue
        return supported_rates
    
    def list_audio_devices(self):
        """List all available audio input devices."""
        print("Available audio input devices:")
        print("-" * 50)
        
        device_count = self.audio_interface.get_device_count()
        for i in range(device_count):
            device_info = self.audio_interface.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:  # Only input devices
                print(f"Device {i}: {device_info['name']}")
                print(f"  Max Input Channels: {device_info['maxInputChannels']}")
                print(f"  Default Sample Rate: {device_info['defaultSampleRate']} Hz")
                supported_rates = self.get_supported_sample_rates(i)
                print(f"  Supported Sample Rates: {supported_rates}")
                print()

    def start_recording(self, callback: Optional[Callable] = None):
        """Start recording audio from the input device."""
        if self.is_recording:
            print("Already recording!")
            return
            
        try:
            # Setup resampler if needed
            if self.resample_to_target and self.device_sample_rate != self.target_samplerate:
                self._setup_resampler()
            
            # Open audio stream
            self.stream = self.audio_interface.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.device_sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback if callback is None else callback
            )
            
            self.is_recording = True
            self.stream.start_stream()
            
            if self.debug_mode:
                print(f"Started recording at {self.device_sample_rate} Hz")
                
        except Exception as e:
            print(f"Error starting recording: {e}")
            raise

    def stop_recording(self):
        """Stop recording audio."""
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
        if self.debug_mode:
            print("Stopped recording")

    def _setup_resampler(self):
        """Setup audio resampler for sample rate conversion."""
        self.resampler = {
            'ratio': self.target_samplerate / self.device_sample_rate,
            'state': None
        }
        
        if self.debug_mode:
            print(f"Setup resampler: {self.device_sample_rate} -> {self.target_samplerate} Hz")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Default audio callback function."""
        if status:
            if self.debug_mode:
                print(f"Audio callback status: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Resample if needed
        if self.resample_to_target and self.device_sample_rate != self.target_samplerate:
            audio_data = self._resample_audio(audio_data)
        
        # Put audio data in queue for processing
        try:
            self.audio_queue.put_nowait(audio_data)
        except queue.Full:
            if self.debug_mode:
                print("Audio queue full, dropping frame")
        
        return (None, pyaudio.paContinue)

    def _resample_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Resample audio data to target sample rate."""
        try:
            # Use scipy.signal.resample for resampling
            target_length = int(len(audio_data) * self.resampler['ratio'])
            resampled = scipy.signal.resample(audio_data, target_length)
            return resampled.astype(np.int16)
        except Exception as e:
            if self.debug_mode:
                print(f"Resampling error: {e}")
            return audio_data

    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get the next audio chunk from the queue."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def record_for_duration(self, duration: float) -> np.ndarray:
        """Record audio for a specified duration and return as numpy array."""
        chunks = []
        self.start_recording()
        
        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                chunk = self.get_audio_chunk(timeout=0.1)
                if chunk is not None:
                    chunks.append(chunk)
        finally:
            self.stop_recording()
        
        if chunks:
            return np.concatenate(chunks)
        else:
            return np.array([], dtype=np.int16)

    def audio_stream_generator(self) -> Generator[np.ndarray, None, None]:
        """Generator that yields audio chunks as they become available."""
        while self.is_recording:
            chunk = self.get_audio_chunk(timeout=0.1)
            if chunk is not None:
                yield chunk

    def get_device_info(self) -> dict:
        """Get information about the current input device."""
        if self.input_device_index is not None:
            return self.audio_interface.get_device_info_by_index(self.input_device_index)
        return {}

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.stop_recording()
        if self.audio_interface:
            self.audio_interface.terminate()

    def __del__(self):
        """Destructor - cleanup resources."""
        self.stop_recording()
        if hasattr(self, 'audio_interface') and self.audio_interface:
            self.audio_interface.terminate()


# Example usage and testing
if __name__ == "__main__":
    # Create audio input instance
    audio_input = AudioInput(debug_mode=True)
    
    try:
        # List available devices
        audio_input.list_audio_devices()
        
        # Record for 5 seconds
        print("\nRecording for 5 seconds...")
        audio_data = audio_input.record_for_duration(5.0)
        print(f"Recorded {len(audio_data)} samples")
        print(f"Recording duration: {len(audio_data) / audio_input.target_samplerate:.2f} seconds")
        
        # Example of streaming recording
        print("\nStarting streaming recording for 3 seconds...")
        audio_input.start_recording()
        
        start_time = time.time()
        chunk_count = 0
        while time.time() - start_time < 3.0:
            chunk = audio_input.get_audio_chunk(timeout=0.1)
            if chunk is not None:
                chunk_count += 1
                print(f"Received chunk {chunk_count}: {len(chunk)} samples")
        
        audio_input.stop_recording()
        print(f"Received {chunk_count} chunks")
        
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        del audio_input