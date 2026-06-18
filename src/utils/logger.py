import sys

def log_step(msg):
    print(f"\n▶ {msg}")
    sys.stdout.flush()

def log_info(msg):
    print(f"   → {msg}")
    sys.stdout.flush()

def log_done(msg="Done"):
    print(f"✔ {msg}")
    sys.stdout.flush()