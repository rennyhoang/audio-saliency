import os
import numpy as np
from PIL import Image
from pathlib import Path

def images_to_csv(input_dir, output_dir):
    """
    Converts 720x360 PNGs to CSVs with 720 rows and 360 columns.
    Values are normalized brightness (0-1).
    """
    # Convert string paths to Path objects
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    # Create output directory if it doesn't exist
    out_path.mkdir(parents=True, exist_ok=True)

    # Verify input directory exists
    if not in_path.exists():
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    files = list(in_path.glob("*.png"))
    print(f"Found {len(files)} PNG files to process...")

    for file_path in files:
        try:
            # 1. Open image and convert to Grayscale (Luminance)
            # This calculates brightness using standard weighting: 
            # L = R * 0.299 + G * 0.587 + B * 0.114
            with Image.open(file_path) as img:
                
                # Ensure it is the correct resolution (Optional safeguard)
                if img.size != (720, 360):
                    print(f"Skipping {file_path.name}: Size is {img.size}, expected (720, 360)")
                    continue

                # Convert to grayscale
                grayscale_img = img.convert('L')

                # 2. Convert to NumPy array and Normalize (0-1)
                # Standard image array shape is (Height, Width) -> (360, 720)
                img_array = np.array(grayscale_img) / 255.0

                # 3. Transpose to match specific requirement: 720 Rows, 360 Columns
                # We swap the axes so Width becomes Rows and Height becomes Columns
                transposed_array = img_array.T 

                # 4. Generate Output Filename
                output_filename = file_path.stem + ".csv"
                output_file_path = out_path / output_filename

                # 5. Save to CSV
                # fmt='%.6f' keeps 6 decimal places for precision
                np.savetxt(output_file_path, transposed_array, delimiter=",", fmt='%.6f')
                
                print(f"Saved: {output_filename}")

        except Exception as e:
            print(f"Failed to process {file_path.name}: {e}")

    print("Processing complete.")

# --- Configuration ---
if __name__ == "__main__":
    # CHANGE THESE PATHS TO YOUR ACTUAL DIRECTORIES
    INPUT_DIRECTORY = "surf-out"
    OUTPUT_DIRECTORY = "surf-csvs"

    images_to_csv(INPUT_DIRECTORY, OUTPUT_DIRECTORY)
