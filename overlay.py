import cv2
import glob
import os
import re

# --- User Configuration ---

# Path to your base video
VIDEO_PATH = "video.mp4"

# Directory containing your 'output[num].jpg' images
IMAGE_DIR = "predictions"

# Path for the final output video
OUTPUT_PATH = "final_video.mp4"

# Set the transparency of the overlay (0.0 = fully transparent, 1.0 = fully opaque)
OVERLAY_ALPHA = 0.5

# --------------------------

def get_image_num(filename):
    """Extracts the number from filenames like 'output123.jpg' for correct sorting."""
    match = re.search(r'output(\d+)\.jpg', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return -1

def main():
    # 1. Load and numerically sort all images
    image_pattern = os.path.join(IMAGE_DIR, "output*.jpg")
    image_paths = sorted(glob.glob(image_pattern), key=get_image_num)
    
    num_images = len(image_paths)
    if num_images == 0:
        print(f"Error: No images found at '{image_pattern}'")
        return

    print(f"Found {num_images} images to overlay.")

    # 2. Open the base video and get its properties
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{VIDEO_PATH}'")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_size = (video_width, video_height)

    print(f"Video properties: {video_width}x{video_height} @ {fps:.2f} FPS, {total_frames} total frames.")

    # 3. Calculate timing
    # This determines how many video frames each single image should span
    if total_frames == 0:
        print("Error: Video has 0 frames or metadata is corrupt.")
        cap.release()
        return
        
    frames_per_image = total_frames / num_images
    print(f"Each image will be displayed for approximately {frames_per_image:.2f} frames.")

    # 4. Set up the video writer
    # Use 'mp4v' for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, video_size)

    # 5. Load images into memory
    # For performance, we load them once instead of reading from disk in the loop
    # If you have too many images to fit in RAM, this needs to be modified.
    images = [cv2.imread(p) for p in image_paths]
    
    current_frame_num = 0
    print("Processing video...")

    # 6. Loop through video frames, resize images, and composite
    while True:
        ret, video_frame = cap.read()
        if not ret:
            # End of video
            break

        # Determine which image to use for this frame
        image_index = int(current_frame_num / frames_per_image)
        # Clamp index to the last image in case of calculation rounding
        image_index = min(image_index, num_images - 1) 

        # Get the correct overlay image
        overlay_image = images[image_index]

        # Resize the overlay image to match the video's resolution
        resized_overlay = cv2.resize(overlay_image, video_size, interpolation=cv2.INTER_AREA)

        # Blend the video frame and the resized overlay
        # new_frame = (video_frame * (1 - alpha)) + (resized_overlay * alpha)
        composited_frame = cv2.addWeighted(
            video_frame,        # Source 1 (base)
            1.0 - OVERLAY_ALPHA,  # Weight for source 1
            resized_overlay,    # Source 2 (overlay)
            OVERLAY_ALPHA,      # Weight for source 2
            0                   # Gamma correction (0 = no change)
        )

        # Write the new composited frame to the output file
        out.write(composited_frame)
        
        current_frame_num += 1
        if current_frame_num % int(fps) == 0: # Print update every second
             print(f"  Processed frame {current_frame_num}/{total_frames}", end='\r')


    # 7. Release all resources
    print(f"\nProcessing complete. Video saved to '{OUTPUT_PATH}'")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
