import numpy as np
import librosa
import os

def convert_wav_to_npy(wav_path, sr=16000, save_dir='example_data'):
    """
    Convert a WAV file to a normalized NPY array and save it.
    
    Args:
        wav_path (str): Path to the WAV file
        sr (int): Sampling rate (default: 16000)
        save_dir (str): Directory to save the NPY file
    
    Returns:
        str: Path to the saved NPY file
    """
    print(f"Converting {wav_path} to NPY format...")
    
    # Load and normalize audio
    audio, _ = librosa.load(wav_path, sr=sr)
    audio = audio / np.max(np.abs(audio))
    
    # Create save path
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    npy_path = os.path.join(save_dir, f"{base_name}.npy")
    
    # Save as NPY file
    np.save(npy_path, audio)
    print(f"Saved NPY file to: {npy_path}")
    print(f"Audio length: {len(audio)} samples")
    return npy_path

if __name__ == "__main__":
    # Convert the Bach WAV file
    wav_file = "./example_data/data_gt_bach.wav"
    npy_file = convert_wav_to_npy(wav_file) 