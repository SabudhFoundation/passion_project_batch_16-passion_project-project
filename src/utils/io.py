import os
from pathlib import Path
import shutil

def copy_to_downloads(ttf_path):
    win_user = os.getenv("USER")  # often same as Windows username in WSL
    win_downloads = Path(f"/mnt/c/Users/{win_user}/Downloads")

    if not win_downloads.exists():
        print(f"Windows Downloads not found: {win_downloads}")
        return

    target = win_downloads / ttf_path.name

    try:
        shutil.copy(ttf_path, target)
        print(f"Copied to Windows Downloads: {target}")
        print(f"Open here: file:///{target}")
    except Exception as e:
        print(f"Copy failed: {e}")