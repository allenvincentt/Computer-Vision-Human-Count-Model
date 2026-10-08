import argparse
import urllib.request
from pathlib import Path

BASE = "https://zenodo.org/records/15731095/files"
FILES = {
    "n": "Yolo_v11_n.pt",
    "x": "Yolo_v11_x.pt",
}


def main():
    p = argparse.ArgumentParser(description="Download the top-view person YOLO11 weights")
    p.add_argument("--size", choices=["n", "x", "all"], default="all")
    args = p.parse_args()

    dest = Path(__file__).resolve().parent / "models"
    dest.mkdir(exist_ok=True)
    for key, name in FILES.items():
        if args.size not in (key, "all"):
            continue
        target = dest / name
        if target.exists():
            print(f"Exists: {target}")
            continue
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(f"{BASE}/{name}?download=1", target)
        print(f"Saved: {target}")


if __name__ == "__main__":
    main()
