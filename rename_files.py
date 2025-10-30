import os
import sys

def rename_files_in_subdirs(main_dir="HM"):
    """
    Renames files in subdirectories of main_dir.

    Looks for files like '0001_h.jpg', '0003_h.jpg' and renames them
    to a consecutive sequence '0001.jpg', '0002.jpg', etc.
    """
    print(f"Starting scan in directory: {main_dir}")

    # Check if the main directory exists
    if not os.path.isdir(main_dir):
        print(f"Error: Directory '{main_dir}' not found.", file=sys.stderr)
        return

    # os.walk traverses directories top-down
    # (dirpath, dirnames, filenames)
    for dirpath, dirnames, filenames in os.walk(main_dir):
        
        # Skip the main directory itself, we only want to process subdirectories
        if dirpath == main_dir:
            print(f"Found subdirectories: {dirnames}")
            continue

        print(f"\nProcessing subdirectory: {dirpath}")

        # 1. Filter files that match the pattern (ending in _h.jpg)
        files_to_rename = [f for f in filenames if f.lower().endswith(".jpg")]
        
        # 2. Sort the files alphabetically/numerically
        files_to_rename.sort()

        if not files_to_rename:
            print("  No '_h.jpg' files found to rename.")
            continue

        # 3. Initialize counter for new filenames
        counter = 0

        # 4. Iterate and rename
        for old_name in files_to_rename:
            # Get the file extension (e.g., '.jpg')
            file_ext = os.path.splitext(old_name)[1]
            
            # Create the new filename, padding with leading zeros
            # e.g., 1 -> "0001.jpg", 12 -> "0012.jpg"
            new_name = f"{counter:04d}_h{file_ext}"

            # Get full paths for old and new files
            old_path = os.path.join(dirpath, old_name)
            new_path = os.path.join(dirpath, new_name)

            # 5. Rename the file
            try:
                os.rename(old_path, new_path)
                print(f"  Renamed: {old_name} -> {new_name}")
                counter += 1
            except OSError as e:
                print(f"  Error renaming {old_name}: {e}", file=sys.stderr)
            except FileExistsError:
                print(f"  Error: {new_name} already exists. Skipping {old_name}.", file=sys.stderr)


if __name__ == "__main__":
    # Set the root directory here
    # You can change "HM" to the full path if needed
    root_directory = "HM"
    rename_files_in_subdirs(root_directory)
    print("\nFile renaming process complete.")

