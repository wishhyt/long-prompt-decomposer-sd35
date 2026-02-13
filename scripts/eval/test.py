import os
import sys

# BOOTSTRAP_PATHS: allow running scripts directly without installing the package.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import os
from pathlib import Path

# Debug: Check file existence
base_path = "//data/sa-1b"
missing_files = []
corrupted_files = []

for i in range(1, 201):
    filepath = f"{base_path}/{i:03d}.tar.gz"
    if not os.path.exists(filepath):
        print(filepath)
        missing_files.append(filepath)
    else:
        # Check if file is readable and not corrupted
        try:
            import tarfile
            with tarfile.open(filepath, 'r') as tar:
                # Try to read the first entry
                names = tar.getnames()
                if not names:
                    corrupted_files.append(filepath)
        except Exception as e:
            print((filepath, str(e)))
            corrupted_files.append((filepath, str(e)))

print(f"Missing files: {len(missing_files)}")
print(f"Corrupted files: {len(corrupted_files)}")
if missing_files[:5]:  # Show first 5
    print("First missing files:", missing_files[:5])
if corrupted_files[:5]:
    print("First corrupted files:", corrupted_files[:5])