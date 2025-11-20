import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import mixed_precision
from pathlib import Path
from collections import deque
from tqdm import tqdm
import argparse

# --- Configuration ---
# Must match your training configuration
IMG_HEIGHT = 360
IMG_WIDTH = 720
SEQ_LENGTH = 10
MODEL_PATH = "best_model.keras"  # Or "final_model.keras"

def setup_environment():
    # Ensure Mixed Precision is set if it was used during training
    # This prevents dtype conflicts when loading the weights
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    
    # Prevent TF from eating all GPU memory during inference
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def preprocess_image(img_path):
    """Loads, resizes, and normalizes a single image."""
    # Read image
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    
    # Resize to model expected input
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
    # Normalize to [0, 1]
    img = img.astype('float32') / 255.0
    return img

def predict_directory(input_dir, output_dir, model_path):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded successfully.")

    # Get sorted list of PNGs
    image_files = sorted(input_path.glob("*.png"))
    if not image_files:
        print("No .png files found in input directory.")
        return

    print(f"Found {len(image_files)} images. Starting inference...")

    # Initialize the sliding window buffer
    # We use a deque with a max length to automatically pop old frames
    frame_buffer = deque(maxlen=SEQ_LENGTH)

    # --- PRE-FILLING BUFFER (PADDING) ---
    # To generate predictions for the first SEQ_LENGTH-1 frames,
    # we pad the start of the buffer with copies of the first frame.
    first_img = preprocess_image(image_files[0])
    for _ in range(SEQ_LENGTH - 1):
        frame_buffer.append(first_img)

    # Loop through every image in the directory
    for img_file in tqdm(image_files, desc="Processing Frames"):
        
        # 1. Load and Preprocess
        current_img = preprocess_image(img_file)
        if current_img is None:
            print(f"Warning: Could not read {img_file}")
            continue
            
        # 2. Update Sliding Window
        frame_buffer.append(current_img)
        
        # Ensure we have enough frames (should always be true due to padding)
        if len(frame_buffer) == SEQ_LENGTH:
            
            # 3. Prepare Batch
            # Shape: (1, SEQ_LENGTH, H, W, 3)
            sequence = np.array(frame_buffer)
            batch = np.expand_dims(sequence, axis=0)
            
            # 4. Predict
            # Returns shape: (1, SEQ_LENGTH, H, W, 3)
            # verbose=0 prevents progress bar spam for every single frame
            prediction_seq = model.predict(batch, verbose=0)
            
            # 5. Extract the result for the CURRENT frame
            # In a sequence-to-sequence model, the last timestep corresponds 
            # to the prediction for the last image added to the buffer.
            last_frame_pred = prediction_seq[0, -1, :, :, :]
            
            # 6. Post-process (Scale back to 0-255)
            output_img = (last_frame_pred * 255.0).astype('uint8')
            
            # 7. Save
            # Use the same filename as input, but in the output directory
            save_path = output_path / img_file.name
            cv2.imwrite(str(save_path), output_img)

    print(f"\nDone! Predictions saved to: {output_path.resolve()}")

if __name__ == "__main__":
    # Argument parsing for flexibility
    parser = argparse.ArgumentParser(description="Run ConvLSTM Inference on a directory of images.")
    parser.add_argument("--input", type=str, required=True, help="Path to input directory containing .png images")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory to save predictions")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to the .keras model file")
    
    args = parser.parse_args()
    
    setup_environment()
    predict_directory(args.input, args.output, args.model)
