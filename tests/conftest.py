import sys
from pathlib import Path

# Ensure metal-kans directory is in sys.path
repo_dir = Path(__file__).resolve().parent.parent
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))
