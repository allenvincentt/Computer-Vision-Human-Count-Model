import argparse
import time

import cv2

from people_counter import CountSmoother, LineCounter, PersonDetector, UniqueCounter
from people_counter.draw import draw_detections, draw_line, draw_panel


def parse_args():
    p = argparse.ArgumentParser(description="Overhead people detection and counting")
    p.add_argument("--source", default="0", help="Camera index, video file, or RTSP/HTTP stream URL")
    p.add_argument("--model", default="top-n", help="top-n | top-x | custom | front | path to .pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.65)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None, help="0 for GPU, cpu for CPU")
    p.add_argument("--tta", action="store_true", help="Test-time augmentation for higher accuracy, slower")
    p.add_argument("--smooth", type=int, default=9, help="Frames used to smooth the live count")
    p.add_argument("--line", choices=["none", "h", "v"], default="none", help="Enable a counting line")
    p.add_argument("--line-pos", type=float, default=0.5, help="Line position from 0 to 1")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--flip", action="store_true", help="Mirror the camera image")
    p.add_argument("--save", default=None, help="Save the annotated video to this .mp4 path")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def open_source(source, width, height):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    return cap


def main():
    args = parse_args()
    device = int(args.device) if args.device and args.device.isdigit() else args.device
    detector = PersonDetector(args.model, args.conf, args.iou, args.imgsz, device, args.tta)
    smoother = CountSmoother(args.smooth)
    unique = UniqueCounter()
    line = LineCounter(args.line, args.line_pos) if args.line != "none" else None

    cap = open_source(args.source, args.width, args.height)
    writer = None
    fps, last = 0.0, time.perf_counter()
    window = "HumanDetectionModel"
    print(f"Model: {detector.path.name} on {detector.device} | keys: q/ESC quit, r reset")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.flip:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        detections = detector.track(frame)
        count = smoother.update(len(detections))
        unique.update(detections)

        draw_detections(frame, detections)
        lines = [f"People: {count}", f"Raw: {len(detections)}  Unique: {unique.total}"]
        if line:
            line.update(detections, w, h)
            draw_line(frame, *line.line(w, h))
            lines.append(f"In: {line.entered}  Out: {line.exited}  Inside: {line.inside}")

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
        last = now
        lines.append(f"FPS: {fps:.1f}  Model: {detector.name}")
        draw_panel(frame, lines)

        if args.save:
            if writer is None:
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            writer.write(frame)

        if not args.no_show:
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                smoother.reset()
                unique.reset()
                detector.reset_tracker()
                if line:
                    line.reset()
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
