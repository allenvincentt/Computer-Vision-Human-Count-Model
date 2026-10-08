from collections import deque

import numpy as np


class CountSmoother:
    def __init__(self, window=9):
        self.history = deque(maxlen=max(1, window))

    def update(self, count):
        self.history.append(count)
        return int(round(float(np.median(self.history))))

    def reset(self):
        self.history.clear()


class LineCounter:
    def __init__(self, orientation="h", position=0.5, min_frames=2):
        self.orientation = orientation
        self.position = position
        self.min_frames = min_frames
        self.sides = {}
        self.entered = 0
        self.exited = 0

    def line(self, width, height):
        if self.orientation == "h":
            y = int(height * self.position)
            return (0, y), (width, y)
        x = int(width * self.position)
        return (x, 0), (x, height)

    def _side(self, point, width, height):
        if self.orientation == "h":
            return 1 if point[1] > height * self.position else -1
        return 1 if point[0] > width * self.position else -1

    def update(self, detections, width, height):
        active = set()
        for d in detections:
            if d.track_id is None:
                continue
            active.add(d.track_id)
            side = self._side(d.center, width, height)
            prev = self.sides.get(d.track_id)
            if prev is None:
                self.sides[d.track_id] = [side, 1]
                continue
            if side == prev[0]:
                prev[1] += 1
                continue
            if prev[1] >= self.min_frames:
                if side == 1:
                    self.entered += 1
                else:
                    self.exited += 1
            self.sides[d.track_id] = [side, 1]
        if len(self.sides) > 500:
            for tid in list(self.sides)[:-200]:
                if tid not in active:
                    del self.sides[tid]

    @property
    def inside(self):
        return max(0, self.entered - self.exited)

    def reset(self):
        self.sides.clear()
        self.entered = 0
        self.exited = 0


class UniqueCounter:
    def __init__(self, min_frames=5):
        self.min_frames = min_frames
        self.seen = {}
        self.total = 0

    def update(self, detections):
        for d in detections:
            if d.track_id is None:
                continue
            n = self.seen.get(d.track_id, 0) + 1
            self.seen[d.track_id] = n
            if n == self.min_frames:
                self.total += 1

    def reset(self):
        self.seen.clear()
        self.total = 0
