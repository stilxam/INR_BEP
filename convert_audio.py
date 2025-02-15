import os
import numpy as np
import soundfile as sf

def convert_wav_to_npy(input_dir="./example_data/Audio"):
    """
    Convert all .wav files in the given directory structure to .npy files.
    Maintains the same folder structure.
    """
    # Iterate through all category folders
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        
        # Skip if not a directory
        if not os.path.isdir(category_path):
            continue
            
        # Process all wav files in the category folder
        for filename in os.listdir(category_path):
            if filename.endswith('.wav'):
                wav_path = os.path.join(category_path, filename)
                
                # Read the wav file
                audio_data, sample_rate = sf.read(wav_path)
                
                # Convert stereo to mono if necessary
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                # Create output filename (replace .wav with .npy)
                npy_filename = os.path.splitext(filename)[0] + '.npy'
                npy_path = os.path.join(category_path, npy_filename)
                
                # Save as numpy array
                np.save(npy_path, audio_data)
                print(f"Converted {wav_path} to {npy_path}")

if __name__ == "__main__":
    convert_wav_to_npy()

# To run this script:
# 1. Make sure you have the required packages installed:
#    pip install numpy soundfile  
#
# 2. Save this file as convert_audio.py
#
# 3. Open a terminal/command prompt and navigate to the directory containing this script
#
# 4. Run the script using:
#    python convert_audio.py
#
# The script will process all .wav files in ./example_data/Audio and its subfolders
# creating .npy files in the same locations

