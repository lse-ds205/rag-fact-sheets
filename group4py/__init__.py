import sys
from pathlib import Path

package_dir = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(package_dir))