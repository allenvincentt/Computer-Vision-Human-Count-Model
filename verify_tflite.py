import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter


def iou(a, b):
    l, t = max(a[0], b[0]), max(a[1], b[1])
    r, btm = min(a[2], b[2]), min(a[3], b[3])
    o = max(0, r - l) * max(0, btm - t)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - o
    return o / u if u > 0 else 0


def letterbox(path, size, pad=0.0):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    px = np.asarray(img.resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255
    canvas = np.full((size, size, 3), pad, np.float32)
    padx, pady = (size - nw) // 2, (size - nh) // 2
    canvas[pady:pady + nh, padx:padx + nw] = px
    return canvas


def count_people(interpreter, path, conf, nms, pad=0.0):
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    nchw = inp["shape"][1] == 3
    size = int(inp["shape"][2] if nchw else inp["shape"][1])
    canvas = letterbox(path, size, pad)
    interpreter.set_tensor(inp["index"], canvas.transpose(2, 0, 1)[None] if nchw else canvas[None])
    interpreter.invoke()
    y = interpreter.get_tensor(out["index"])[0]
    if y.shape[0] > y.shape[1]:
        y = y.T
    unit = size if y[:4].max() > 1.5 else 1
    boxes = []
    for a in np.where(y[4] >= conf)[0]:
        cx, cy, bw, bh = y[:4, a] / unit
        boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, float(y[4, a])))
    boxes.sort(key=lambda b: -b[4])
    kept = []
    for b in boxes:
        if all(iou(b, k) < nms for k in kept):
            kept.append(b)
    return len(kept)


def main():
    p = argparse.ArgumentParser(description="Count people with a .tflite export using the mobile app's preprocessing")
    p.add_argument("model")
    p.add_argument("--images", default="samples/images")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--nms", type=float, default=0.65)
    p.add_argument("--pad", type=int, default=114, help="Letterbox fill value, 0 to 255")
    args = p.parse_args()

    interpreter = Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    print(f"input {inp['shape'].tolist()} {inp['dtype'].__name__}  output {out['shape'].tolist()} {out['dtype'].__name__}")

    total = 0
    for f in sorted(Path(args.images).glob("*.jpg")):
        n = count_people(interpreter, f, args.conf, args.nms, args.pad / 255)
        total += n
        print(f"{f.name}: {n}")
    print(f"Total people: {total}")


if __name__ == "__main__":
    main()

