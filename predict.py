import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import argparse

# --- Configuration (Must match training script) ---
IMG_HEIGHT = 19 
IMG_WIDTH = 36 
SEQ_LENGTH = 10 
CHANNELS = 3
BATCH_SIZE = 8 # Batch size for prediction (can be different from training)

def load_and_preprocess_sequence(input_dir, seq_length):
    """
    Loads all images from a single directory and creates the
    sliding window sequences needed for the model.
    """
    # Find and sort all input PNGs
    image_paths = sorted(Path(input_dir).glob("*.png"))
    
    if len(image_paths) < seq_length:
        print(f"Error: Not enough frames in {input_dir}. Found {len(image_paths)}, but need at least {seq_length}.")
        return None, None

    frames = []
    filenames = []
    
    # Load, resize, and normalize all frames
    for img_path in image_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype('float32') / 255.0  # Normalize
        frames.append(img)
        filenames.append(img_path.name) # Save the original filename

    # Create sliding windows
    sequences = []
    for i in range(len(frames) - seq_length + 1):
        sequences.append(frames[i : i + seq_length])
        
    return np.array(sequences), filenames

def save_reconstructed_predictions(predictions, all_filenames, output_dir, seq_length):
    """
    Averages the overlapping predictions from the sliding window
    to reconstruct the full sequence of images.
    """
    num_windows, seq_len, h, w, c = predictions.shape
    num_frames = len(all_filenames)
    
    # Create empty arrays to store the summed predictions and counts
    reconstructed_frames = np.zeros((num_frames, h, w, c), dtype=np.float32)
    counts = np.zeros((num_frames,), dtype=np.float32)

    # Loop over each predicted window
    for i in range(num_windows):
        pred_sequence = predictions[i]
        # Add the predictions to the corresponding frame "slots"
        for j in range(seq_len):
            frame_index = i + j
            reconstructed_frames[frame_index] += pred_sequence[j]
            counts[frame_index] += 1
            
    # Divide the summed predictions by the count to get the average
    # We add [:, None, None, None] to broadcast the 1D counts array
    # across the 4D frames array for division.
    average_frames = reconstructed_frames / counts[:, None, None, None]
    
    print(f"Reconstructing and saving {num_frames} frames to {output_dir}...")
    
    # Save each averaged frame
    for i in range(num_frames):
        # De-normalize from [0, 1] back to [0, 255]
        img_data = (average_frames[i] * 255).astype(np.uint8)
        
        # Get the original filename and change the extension to .jpg
        base_name = Path(all_filenames[i]).stem
        output_filename = f"{base_name}.jpg"
        output_path = Path(output_dir) / output_filename
        
        cv2.imwrite(str(output_path), img_data)

    print("Saving complete.")

def main(args):
    # 1. Load the trained model
    print(f"Loading model from {args.model}...")
    try:
        model = tf.keras.models.load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model file exists and was trained with the same TensorFlow version.")
        return

    # 2. Create the output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # 3. Load and preprocess the input sequence
    print(f"Loading and preprocessing data from {args.input}...")
    sequences, filenames = load_and_preprocess_sequence(args.input, SEQ_LENGTH)
    
    if sequences is None:
        return
        
    print(f"Input data shape (X): {sequences.shape}") # (num_windows, seq_len, 36, 19, 3)

    # 4. Run prediction
    print("Running model.predict()...")
    predictions = model.predict(sequences, batch_size=BATCH_SIZE, verbose=1)
    print(f"Prediction output shape: {predictions.shape}")

    # 5. Reconstruct and save the outputs
    save_reconstructed_predictions(predictions, filenames, args.output, SEQ_LENGTH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ConvLSTM inference on a sequence of images.")
    
    parser.add_argument(
        "-m", "--model", 
        type=str, 
        required=True, 
        help="Path to the saved .keras model file (e.g., best_model.keras)."
    )
    parser.add_argument(
        "-i", "--input", 
        type=str, 
        required=True, 
        help="Path to the *single* input directory containing PNGs (e.g., input/test_scene_1)."
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        required=True, 
        help="Path to the output directory to save predicted JPGs (e.g., predictions/test_scene_1)."
    )
    
    cli_args = parser.parse_args()
    main(cli_args)
