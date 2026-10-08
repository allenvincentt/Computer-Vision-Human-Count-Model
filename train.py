import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

from people_counter.detector import MODELS, resolve_model


def main():
    p = argparse.ArgumentParser(description="Fine-tune the top-view model on your own overhead images")
    p.add_argument("--data", required=True, help="Path to a YOLO data.yaml with one class: person")
    p.add_argument("--base", default="top-n", help="top-n | top-x | path to .pt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--patience", type=int, default=15)
    args = p.parse_args()

    model = YOLO(str(resolve_model(args.base)))
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=0 if torch.cuda.is_available() else "cpu",
        project="runs",
        name="overhead",
        exist_ok=True,
        single_cls=True,
        degrees=180.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
        workers=2,
    )

    best = Path("runs/overhead/weights/best.pt")
    if best.exists():
        shutil.copy(best, MODELS["custom"])
        print(f"Saved fine-tuned model to {MODELS['custom']}. Run: python app.py --model custom")


if __name__ == "__main__":
    main()
