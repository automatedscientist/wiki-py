"""Filter converted files to only include successfully executed ones."""
import shutil
from pathlib import Path

# Load successful indices
with open("data/successful_indices.txt") as f:
    valid_indices = set(int(line.strip()) for line in f if line.strip())

print(f"Loaded {len(valid_indices)} valid indices")

# Find and copy matching files
src_dir = Path("data/converted")
dst_dir = Path("data/converted_valid")
dst_dir.mkdir(exist_ok=True)

copied = 0
for py_file in src_dir.glob("*.py"):
    # Extract index from filename like "00123_Article_Name.py"
    idx = int(py_file.name.split("_")[0])
    if idx in valid_indices:
        shutil.copy(py_file, dst_dir / py_file.name)
        copied += 1
        if copied % 5000 == 0:
            print(f"Copied {copied} files...")

print(f"Done! Copied {copied} valid files to {dst_dir}")
