import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv3D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
INPUT_DIR = Path("output")
OUTPUT_DIR = Path("HM")
IMG_HEIGHT = 19 
IMG_WIDTH = 36 
# ConvLSTM expects sequences. We'll use a sliding window to create them.
# This means the model will look at `SEQ_LENGTH` volume maps to predict
# `SEQ_LENGTH` saliency maps.
SEQ_LENGTH = 10 
CHANNELS = 3  # We'll load images as color (3 channels)
BATCH_SIZE = 8
EPOCHS = 50

def load_data(input_path, output_path, seq_length):
    """
    Loads and preprocesses data into sequences for the ConvLSTM.
    """
    X_data, Y_data = [], []
    
    # Find all subdirectories in the input path
    input_subdirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(input_subdirs)} subdirectories (sequences)...")
    
    for subdir in tqdm(input_subdirs, desc="Loading sequences"):
        output_subdir = output_path / subdir.name
        
        # Check if the corresponding output directory exists
        if not output_subdir.exists():
            print(f"Warning: Skipping {subdir.name}, no matching output dir found.")
            continue
            
        # Get sorted lists of input (PNG) and output (JPG) images
        input_files = sorted(subdir.glob("*.png"))
        output_files = sorted(output_subdir.glob("*.jpg"))
        
        # Verify that the files match up
        # We check by comparing filenames, replacing .png with .jpg
        expected_outputs = {p.with_suffix('.jpg').name for p in input_files}
        found_outputs = {p.name for p in output_files}
        
        if expected_outputs != found_outputs:
            print(f"Warning: Skipping {subdir.name}, file mismatch between input and output.")
            continue
        
        if len(input_files) < seq_length:
            print(f"Warning: Skipping {subdir.name}, not enough frames ({len(input_files)}) for seq_length ({seq_length}).")
            continue

        # Load images for this sequence
        sequence_vols = []
        sequence_sals = []
        
        for img_path in input_files:
            # Construct the corresponding output path
            sal_path = output_subdir / img_path.with_suffix('.jpg').name
            
            # Load volume image (input)
            # We use IMREAD_COLOR to ensure 3 channels, even if PNG is grayscale
            vol_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            vol_img = cv2.resize(vol_img, (IMG_WIDTH, IMG_HEIGHT))
            vol_img = vol_img.astype('float32') / 255.0  # Normalize
            sequence_vols.append(vol_img)
            
            # Load saliency image (output)
            sal_img = cv2.imread(str(sal_path), cv2.IMREAD_COLOR)
            sal_img = cv2.resize(sal_img, (IMG_WIDTH, IMG_HEIGHT))
            sal_img = sal_img.astype('float32') / 255.0  # Normalize
            sequence_sals.append(sal_img)

        # Create sliding windows of SEQ_LENGTH
        for i in range(len(sequence_vols) - seq_length + 1):
            X_data.append(sequence_vols[i : i + seq_length])
            Y_data.append(sequence_sals[i : i + seq_length])

    print(f"Total sequences created: {len(X_data)}")
    
    # Convert lists to NumPy arrays
    return np.array(X_data), np.array(Y_data)

def build_convlstm_model(input_shape):
    """
    Builds the ConvLSTM sequence-to-sequence model.
    """
    inputs = Input(shape=input_shape)
    
    # Stack of ConvLSTM layers
    # `return_sequences=True` is crucial. It makes each ConvLSTM layer
    # output the full sequence, not just the last time step.
    x = ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
    )(inputs)
    x = BatchNormalization()(x)
    
    x = ConvLSTM2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
    )(x)
    x = BatchNormalization()(x)
    
    x = ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
    )(x)
    x = BatchNormalization()(x)
    
    # The output of the last ConvLSTM layer has shape (None, seq_length, 36, 19, 64)
    # We need to map the 64 filters back down to our 3 output channels (RGB).
    # A 3D Convolution with a 1x1x1 kernel is perfect for this.
    # It acts like a "TimeDistributed" Dense layer applied to the channel dimension.
    outputs = Conv3D(
        filters=CHANNELS,
        kernel_size=(1, 1, 1),
        activation="sigmoid",  # Sigmoid for [0, 1] normalized pixel values
        padding="same",
    )(x)
    
    model = Model(inputs, outputs)
    return model

def main():
    # 1. Load Data
    print("Loading data...")
    X, Y = load_data(INPUT_DIR, OUTPUT_DIR, SEQ_LENGTH)
    
    if X.shape[0] == 0:
        print("Error: No data was loaded. Check directory paths and file structure.")
        return

    print(f"Input data shape (X): {X.shape}")  # (num_samples, seq_len, 36, 19, 3)
    print(f"Output data shape (Y): {Y.shape}") # (num_samples, seq_len, 36, 19, 3)

    # 2. Split Data
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")

    # 3. Build Model
    # The input shape for the model is one sample: (seq_length, height, width, channels)
    input_shape = (SEQ_LENGTH, IMG_HEIGHT, IMG_WIDTH, CHANNELS)
    model = build_convlstm_model(input_shape)
    
    # We use Mean Squared Error for image-to-image regression
    model.compile(
        optimizer="adam", 
        loss="mean_squared_error", 
        metrics=["mean_absolute_error"]
    )
    
    model.summary()

    # 4. Define Callbacks
    callbacks = [
        # Save the best model based on validation loss
        ModelCheckpoint(
            "best_model.keras", 
            monitor="val_loss", 
            save_best_only=True
        ),
        # Stop training if validation loss doesn't improve for 5 epochs
        EarlyStopping(
            monitor="val_loss", 
            patience=5, 
            restore_best_weights=True
        )
    ]

    # 5. Train Model
    print("\nStarting training...")
    history = model.fit(
        X_train,
        Y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, Y_val),
        callbacks=callbacks,
    )
    
    print("Training complete.")
    
    # 6. Save the final model
    model.save("final_model.keras")
    print("Final model saved as 'final_model.keras'")
    print("Best model (based on val_loss) saved as 'best_model.keras'")

if __name__ == "__main__":
    main()
