#!/usr/bin/env python3
"""
Audio Input Demo Script

This script demonstrates various usage patterns of the AudioInput class.
"""

from audio_input import AudioInput
import numpy as np
import time

def demo_list_devices():
    """Demonstrate listing available audio devices."""
    print("=== Audio Device Listing Demo ===")
    with AudioInput(debug_mode=True) as audio_input:
        audio_input.list_audio_devices()
    print()

def demo_simple_recording():
    """Demonstrate simple duration-based recording."""
    print("=== Simple Recording Demo ===")
    
    duration = 3.0
    print(f"Recording for {duration} seconds...")
    
    with AudioInput(debug_mode=True) as audio_input:
        audio_data = audio_input.record_for_duration(duration)
        
        print(f"Recorded {len(audio_data)} samples")
        print(f"Recording duration: {len(audio_data) / audio_input.target_samplerate:.2f} seconds")
        print(f"Audio level (RMS): {np.sqrt(np.mean(audio_data.astype(np.float32)**2)):.2f}")
    print()

def demo_streaming_recording():
    """Demonstrate streaming audio capture."""
    print("=== Streaming Recording Demo ===")
    
    duration = 5.0
    print(f"Streaming audio for {duration} seconds...")
    
    with AudioInput(debug_mode=True) as audio_input:
        audio_input.start_recording()
        
        start_time = time.time()
        chunk_count = 0
        total_samples = 0
        
        try:
            while time.time() - start_time < duration:
                chunk = audio_input.get_audio_chunk(timeout=0.1)
                if chunk is not None:
                    chunk_count += 1
                    total_samples += len(chunk)
                    
                    # Calculate audio level for this chunk
                    rms = np.sqrt(np.mean(chunk.astype(np.float32)**2))
                    print(f"Chunk {chunk_count}: {len(chunk)} samples, RMS: {rms:.2f}")
                    
        except KeyboardInterrupt:
            print("\nStopped by user")
        
        audio_input.stop_recording()
        
        print(f"\nTotal chunks received: {chunk_count}")
        print(f"Total samples: {total_samples}")
        print(f"Average chunk size: {total_samples/chunk_count if chunk_count > 0 else 0:.1f}")
    print()

def demo_generator_usage():
    """Demonstrate using the audio stream generator."""
    print("=== Generator Usage Demo ===")
    
    duration = 3.0
    print(f"Using generator to process audio for {duration} seconds...")
    
    with AudioInput(debug_mode=True) as audio_input:
        audio_input.start_recording()
        
        start_time = time.time()
        chunk_count = 0
        
        try:
            for audio_chunk in audio_input.audio_stream_generator():
                if time.time() - start_time >= duration:
                    break
                    
                chunk_count += 1
                rms = np.sqrt(np.mean(audio_chunk.astype(np.float32)**2))
                print(f"Generator chunk {chunk_count}: {len(audio_chunk)} samples, RMS: {rms:.2f}")
                
        except KeyboardInterrupt:
            print("\nStopped by user")
        
        audio_input.stop_recording()
        print(f"Processed {chunk_count} chunks via generator")
    print()

def demo_custom_callback():
    """Demonstrate using a custom audio callback."""
    print("=== Custom Callback Demo ===")
    
    chunk_count = 0
    max_chunks = 50
    
    def custom_callback(in_data, frame_count, time_info, status):
        nonlocal chunk_count
        
        if status:
            print(f"Callback status: {status}")
        
        # Convert to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
        
        chunk_count += 1
        print(f"Custom callback chunk {chunk_count}: {len(audio_data)} samples, RMS: {rms:.2f}")
        
        # Stop after max_chunks
        if chunk_count >= max_chunks:
            return (None, 1)  # paComplete
        
        return (None, 0)  # paContinue
    
    print(f"Recording with custom callback for {max_chunks} chunks...")
    
    with AudioInput(debug_mode=True) as audio_input:
        audio_input.start_recording(callback=custom_callback)
        
        # Wait for recording to complete
        while audio_input.is_recording and chunk_count < max_chunks:
            time.sleep(0.1)
        
        audio_input.stop_recording()
        print(f"Custom callback processed {chunk_count} chunks")
    print()

def demo_device_info():
    """Demonstrate getting device information."""
    print("=== Device Information Demo ===")
    
    with AudioInput(debug_mode=True) as audio_input:
        device_info = audio_input.get_device_info()
        print("Current device information:")
        for key, value in device_info.items():
            print(f"  {key}: {value}")
        
        print(f"\nSupported sample rates for current device:")
        supported_rates = audio_input.get_supported_sample_rates(audio_input.input_device_index)
        print(f"  {supported_rates}")
    print()

def main():
    """Run all demo functions."""
    print("AudioInput Class Demo")
    print("=" * 50)
    print()
    
    try:
        demo_device_info()
        demo_list_devices()
        
        print("Starting audio recording demos...")
        print("Make sure your microphone is working and not muted!")
        print()
        
        demo_simple_recording()
        demo_streaming_recording()
        demo_generator_usage()
        demo_custom_callback()
        
        print("All demos completed successfully!")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("Make sure PyAudio is properly installed and you have a working microphone.")

if __name__ == "__main__":
    main()