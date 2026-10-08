import cv2

GREEN = (80, 220, 80)
YELLOW = (0, 220, 255)
WHITE = (255, 255, 255)
DARK = (20, 20, 20)


def _color(track_id):
    if track_id is None:
        return GREEN
    return tuple(int(c) for c in ((track_id * 47) % 255, (track_id * 91) % 255, (track_id * 151) % 255))


def draw_detections(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d.box
        color = _color(d.track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, d.center, 4, color, -1)
        label = f"#{d.track_id} {d.conf:.2f}" if d.track_id is not None else f"{d.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DARK, 1, cv2.LINE_AA)
    return frame


def draw_line(frame, p1, p2):
    cv2.line(frame, p1, p2, YELLOW, 2)
    return frame


def draw_panel(frame, lines):
    pad, lh = 10, 26
    width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0] for t in lines) + pad * 2
    height = lh * len(lines) + pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), DARK, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (pad, pad + lh * i + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)
    return frame
