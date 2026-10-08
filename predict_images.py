import argparse
from pathlib import Path

import cv2

from people_counter import PersonDetector
from people_counter.draw import draw_detections, draw_panel


def main():
    p = argparse.ArgumentParser(description="Detect and count people in a folder of images")
    p.add_argument("--images", default="samples/images")
    p.add_argument("--model", default="top-n")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--out", default="outputs")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    detector = PersonDetector(args.model, conf=args.conf, imgsz=args.imgsz, tta=args.tta)
    out = Path(args.out) / args.model.replace("/", "_").replace("\\", "_")
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in Path(args.images).iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    total = 0
    for f in files:
        image = cv2.imread(str(f))
        if image is None:
            continue
        detections = detector.detect(image)
        total += len(detections)
        draw_detections(image, detections)
        draw_panel(image, [f"People: {len(detections)}"])
        cv2.imwrite(str(out / f.name), image)
        print(f"{f.name}: {len(detections)}")
        if args.show:
            cv2.imshow("Detections", image)
            if cv2.waitKey(0) & 0xFF in (ord("q"), 27):
                break
    print(f"Images: {len(files)}  Total people: {total}  Saved to: {out}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
