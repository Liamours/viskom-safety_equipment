import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = [
    Path(__file__).parent / "train_chv.py",
    Path(__file__).parent / "train_cppe5.py",
    Path(__file__).parent / "train_sh17.py",
]


def run_script(script: Path, split: str) -> int:
    print("\n" + "=" * 60)
    print(f"  RUNNING: {script.name}  [split={split}]")
    print("=" * 60 + "\n")
    t0     = time.time()
    result = subprocess.run([sys.executable, str(script), "--split", split])
    elapsed = time.time() - t0
    status  = "DONE" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"\n  [{script.name}] {status} — {elapsed / 60:.1f} min")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="original", choices=["original", "8020"])
    args = parser.parse_args()

    print("=" * 60)
    print(f"  PHASE 1 — SEQUENTIAL TRAINING  [split={args.split}]")
    print("=" * 60)

    total_start = time.time()
    failed      = []

    for script in SCRIPTS:
        code = run_script(script, args.split)
        if code != 0:
            failed.append(script.name)

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print(f"  ALL DONE — total time: {total_elapsed / 3600:.2f} hours")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    else:
        print("  All runs completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
