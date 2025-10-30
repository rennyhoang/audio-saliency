import os
import numpy as np
from PIL import Image
import glob
from pathlib import Path
import shutil

def count_files_in_subfolders(base_directory="output"):
    """
    Counts the number of files directly within each subfolder of a given base directory.

    This function iterates through each item in the base_directory. If an item
    is a directory (a "subfolder"), it counts how many files are *directly*
    inside that subfolder (it is not recursive).

    Args:
        base_directory (str): The path to the main directory (e.g., "output").

    Returns:
        dict: A dictionary where keys are subfolder names (str) and
              values are the file counts (int).
    """
    files_per_folder = {}
    base_path = Path(base_directory)

    # Check if the base directory exists and is a directory
    if not base_path.is_dir():
        print(f"Error: Directory '{base_directory}' not found or is not a directory.")
        return files_per_folder

    # Iterate over all items (files, dirs, etc.) in the base directory
    for entry in base_path.iterdir():
        
        # We only care about entries that are directories (subfolders)
        if entry.is_dir():
            subfolder_name = entry.name
            
            # Count items inside this subfolder *if* they are files
            # This uses a generator expression for efficiency.
            try:
                file_count = sum(1 for item in entry.iterdir() if item.is_file())
                files_per_folder[subfolder_name] = file_count
            except PermissionError:
                print(f"Warning: Could not access contents of '{entry}'. Skipping.")
    
    return files_per_folder

def _process_single_directory(directory_path: str, num_frames_to_keep: int, resolution: tuple[int, int]):
    """
    Internal helper function.
    Processes a single directory of video frames to keep a specified number
    of evenly spaced frames, resizes them, and deletes the rest.
    
    (This was the original 'process_video_frames' function)
    """
    print(f"Starting processing for single directory: {directory_path}")

    # --- 1. Find and sort all relevant image files ---
    search_pattern = os.path.join(directory_path, "*_h.jpg")
    all_frame_files = sorted(glob.glob(search_pattern))

    total_frames_found = len(all_frame_files)
    
    if total_frames_found == 0:
        print("Info: No matching frame files found (e.g., '0001_h.jpg'). Skipping.")
        return

    print(f"Found {total_frames_found} total frames.")

    if num_frames_to_keep <= 0:
        print("Error: 'num_frames_to_keep' must be greater than 0.")
        return
        
    if num_frames_to_keep > total_frames_found:
        print(f"Warning: Requested {num_frames_to_keep} frames, but only "
              f"{total_frames_found} were found. Will process all found frames.")
        num_frames_to_keep = total_frames_found

    # --- 2. Calculate which frames to keep ---
    indices_to_keep = np.linspace(
        0, total_frames_found - 1, 
        num=num_frames_to_keep, 
        dtype=int
    )
    
    files_to_keep = set()
    for idx in indices_to_keep:
        files_to_keep.add(all_frame_files[idx])

    # --- 3. Process frames (Resize or Delete) ---
    frames_resized = 0
    frames_deleted = 0

    for file_path in all_frame_files:
        if file_path in files_to_keep:
            try:
                with Image.open(file_path) as img:
                    img_resized = img.resize(resolution, Image.Resampling.LANCZOS)
                    img_resized.save(file_path)
                frames_resized += 1
            except Exception as e:
                print(f"Error resizing/saving {file_path}: {e}")
        else:
            try:
                os.remove(file_path)
                frames_deleted += 1
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

    # --- 4. Final Report ---
    print(f"--- Directory Complete: {directory_path} ---")
    print(f"Frames resized and kept: {frames_resized}")
    print(f"Frames deleted: {frames_deleted}")

def process_video_frames(directory_path: str, num_frames_to_keep: int, resolution: tuple[int, int], recursive: bool = False, frame_map: dict[str, int] | None = None):
    """
    Processes video frames in a directory.

    If 'recursive' is True, it finds all immediate subdirectories within
    'directory_path' and runs the processing on each one independently.

    If 'recursive' is False (default), it runs the processing on
    'directory_path' itself.

    Args:
        directory_path (str): The path to the directory.
        num_frames_to_keep (int): The *default* number of frames to keep.
        resolution (tuple[int, int]): The new resolution (width, height).
        recursive (bool): Whether to process subdirectories.
        frame_map (dict[str, int] | None): Optional. A dictionary mapping 
            subdirectory basenames (e.g., 'subdir1') to a specific number 
            of frames. If a subdirectory is in the map, that number is 
            used; otherwise, 'num_frames_to_keep' is used. 
            Only active when 'recursive=True'.
    """
    if not recursive:
        # Original behavior: process the specified directory
        _process_single_directory(directory_path, num_frames_to_keep, resolution)
    else:
        # New behavior: find and process all subdirectories
        print(f"--- Recursive mode enabled. Processing subdirectories in: {directory_path} ---")
        try:
            # Find all entries in the directory
            all_entries = os.listdir(directory_path)
            subdirectories = []
            
            # Filter for directories
            for entry in all_entries:
                full_path = os.path.join(directory_path, entry)
                if os.path.isdir(full_path):
                    subdirectories.append(full_path)
            
            if not subdirectories:
                print(f"No subdirectories found in {directory_path}.")
            else:
                print(f"Found subdirectories: {[os.path.basename(s) for s in subdirectories]}")
                
                # Loop and process each subdirectory
                for subdir in subdirectories:
                    subdir_name = os.path.basename(subdir)
                    
                    # Determine the number of frames to keep for this specific subdirectory
                    # Start with the default
                    frames_for_subdir = num_frames_to_keep
                    
                    if frame_map and subdir_name in frame_map:
                        # If a specific count is provided in the map, use it
                        frames_for_subdir = frame_map[subdir_name]
                        print(f"\nProcessing '{subdir_name}': Found in frame_map, target frames = {frames_for_subdir}")
                    else:
                        # Otherwise, use the default
                        print(f"\nProcessing '{subdir_name}': Using default target frames = {frames_for_subdir}")

                    _process_single_directory(subdir, frames_for_subdir, resolution)

        except FileNotFoundError:
            print(f"Error: The parent directory '{directory_path}' was not found.")
        except Exception as e:
            print(f"An error occurred while finding subdirectories: {e}")


if __name__ == "__main__":
    target_frames_default = 10
    target_resolution = (36, 19)  # (width, height)
    parent_frame_dir = "HM" 
    custom_frame_counts = count_files_in_subfolders(base_directory="output")

    process_video_frames(
        parent_frame_dir, 
        target_frames_default, 
        target_resolution, 
        recursive=True, 
        frame_map=custom_frame_counts
    )
    
    print("\nScript is ready. Please update 'parent_frame_dir', 'custom_frame_counts',")
    print("and uncomment the 'process_video_frames' call to run it in recursive mode.")


