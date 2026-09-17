#!/usr/bin/env python3
"""
its_giving_v2.py — meme reactions on top of your face, live in Zoom / Meet.

Same as its_giving.py, but expression thresholds are measured in standard
deviations above your own resting face rather than against fixed constants.
Seven seconds of calibration; see README.md.

  python its_giving_v2.py --calibrate
  python its_giving_v2.py [--camera 1] [--no-vcam] [--size 640x480] [--no-flip] [--diag]

Keys:  q quit   d toggle HUD   c recalibrate   1-9 0 - = [ ] force-show a pose
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

POSES = ["time_out", "heart", "cover_nose", "crashing_out", "dance", "nose_closed", "flirty", "hand_up",
         "hands_on_hips", "tongue_out", "open_mouth", "nihuya", "disgusted", "talking_to_wall", "suspicious",
         "shifty_eyes", "spin", "laugh", "hello"]
TEST_KEYS = "1234567890-=[];,'./"
# A pose with no file (or folder) in assets/ is switched off: it never fires and never shows a placeholder.
# A folder assets/<pose>/ with numbered images is a sequence: 1.png, 2.png, 3.png shown in turn, SEQ_MS each;
# name a frame 2_900.png to hold it for 900 ms instead.
SEQ_MS = 600

FACE_SCALE = 2.0
HOLD_SECONDS = 0.4
DETECT_WIDTH = 480       # detection runs on a copy this wide; capture/output stay at the camera's full size
PREVIEW_WIDTH = 640      # the preview window is drawn this wide; the call still gets the full frame
POSE_EVERY = 1           # the body detector runs every Nth frame (1 = every frame; 2 buys ~15% detection speed)
# How long a pose must be held before its meme appears, in seconds (independent of camera FPS).
# Raise a pose's number if it fires when you didn't mean it; anything not listed uses DELAY_DEFAULT.
DELAY_DEFAULT = 0.15
DELAY = {
    "spin": 1.0, "suspicious": 0.3, "talking_to_wall": 0.4, "dance": 0.4, "crashing_out": 0.25,
    "open_mouth": 0.25, "tongue_out": 0.3, "disgusted": 0.4, "nihuya": 0.3, "hands_on_hips": 0.2,
    "flirty": 0.4, "nose_closed": 0.4,   # a hand resting on the face while thinking looks like both
    "hand_up": 1.0,                      # a palm beside the head must be HELD still; a waving one is 'hello'
    "shifty_eyes": 0.5, "laugh": 0.5, "hello": 0.0,
}

Z = dict(
    jaw_open=6.0,
    scream_jaw=3.5,
    tongue_jaw=3.5,
    sneer=4.5,
    disgust=10.0,        # this face never registers noseSneer (camera above), so brows+frown+lip must carry it;
                         # 95 min of calls peaked at 9.6, a deliberate grimace reads 11-12
    squint=4.0,
    stretch=5.0,
    brow_up=4.0,
    laugh=4.0,
)
Z_CAP = 8.0
FLOOR = dict(
    jaw_open=0.32,       # 95 min of real calls: talking sits under 0.12 (p99), touched 0.40 once; the 0.25 s hold covers that
    scream_jaw=0.18,
    tongue_jaw=0.18,
    sneer=0.06,
    squint=0.18,
    stretch=0.40,
    brow_up=0.10,
    laugh=0.75,          # a laugh: wide smile (raw) plus EITHER crinkled eyes OR a slightly open jaw — this face
                         # (0.60 fired 37x per 95 min of real calls on plain smiles; 0.75 -> 11x, genuine laugh is 0.83+)
    laugh_squint=0.05,   # laughs with the mouth (stretch 0.8-0.9) while the eyes and jaw barely register
    laugh_jaw=0.06,
)
T = dict(
    tongue=0.5,
    head_turn=0.31,      # atanh units (see Face.turn_signed); 0.31 ~ the old 0.15 face-fraction for a frontal camera
    gesture=0.035,
    hands_fast=0.10,     # above this the hands are travelling, not posing (jitter of a held pair sits at 0.02-0.05)
    look_down=0.25,      # squint only counts as 'suspicious' if the eyes are NOT dropped this far below neutral
    gaze_swing=0.50,     # shifty eyes: LONG sweeps only — gaze must travel this far between reversals (scale -1..1)
    gaze_window=5.0,     # ...and reverse direction >= GAZE_REVERSALS times within this many seconds
)
GAZE_REVERSALS = 5       # ~5 s of deliberate corner-to-corner; reading gives short hops that never count
WAVE_SWING = 0.25        # a wave: open palm beside the head moving sideways >= this many face widths...
WAVE_WINDOW = 1.5        # ...reversing direction WAVE_REVERSALS times within this many seconds
WAVE_REVERSALS = 2
TURN_CLAMP = 0.45
TURN_SIGMA_K = 1.5       # head turn must also exceed this many sigma of your normal head movement
GEO_SIGMA_FLOOR = 0.006  # geometric channels (landmark ratios) are far less noisy than blendshapes
ONLINE_MIN_FRAMES = 200  # a channel missing from calibration.json learns its neutral from the first N frames

CALIB_FILE = "calibration.json"
CALIB_SECONDS = 7.0
CALIB_RANGE_SECONDS = 8.0   # phase 2: look around the screen as you normally do, so sigma covers head movement
CALIB_WARMUP = 1.5
CALIB_MIN_SAMPLES = 30
SIGMA_FLOOR = 0.015
SIGMA_CEIL = 0.080
CALIB_VERSION = 2

GENERIC_SIGMA = 0.035
GENERIC_MEAN = {
    "jawOpen": 0.08, "eyeSquintLeft": 0.10, "eyeSquintRight": 0.10,
    "eyeBlinkLeft": 0.10, "eyeBlinkRight": 0.10, "noseSneerLeft": 0.03, "noseSneerRight": 0.03,
    "browDownLeft": 0.06, "browDownRight": 0.06, "mouthFrownLeft": 0.05, "mouthFrownRight": 0.05,
    "mouthUpperUpLeft": 0.05, "mouthUpperUpRight": 0.05,
}

INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

MODELS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}
HERE = os.path.dirname(os.path.abspath(__file__))


def ensure_models():
    mdir = os.path.join(HERE, "models")
    os.makedirs(mdir, exist_ok=True)
    paths = {}
    for name, url in MODELS.items():
        path = os.path.join(mdir, name)
        if not os.path.exists(path):
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, path)
        paths[name] = path
    return paths


def preflight(model_path):
    """Open a detector in a throwaway subprocess: bad macOS builds abort() uncatchably."""
    code = (
        "import sys\n"
        "from mediapipe.tasks import python as t\n"
        "from mediapipe.tasks.python import vision\n"
        "vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(\n"
        "    base_options=t.BaseOptions(model_asset_path=sys.argv[1]),\n"
        "    running_mode=vision.RunningMode.VIDEO, num_faces=1,\n"
        "    output_face_blendshapes=True)).close()\n"
    )
    proc = subprocess.run([sys.executable, "-c", code, model_path], capture_output=True, text=True)
    if proc.returncode == 0:
        return
    err = (proc.stderr or "") + (proc.stdout or "")
    print(f"\nMediaPipe cannot start a detector here (python {platform.python_version()}, "
          f"mediapipe {getattr(mp, '__version__', '?')}, exit {proc.returncode}).\n")
    if "Service is unavailable" in err or "MetalHelper" in err or proc.returncode == -6:
        print("Cause: mediapipe 0.10.30+ ships macOS wheels that abort on startup.\n"
              "Fix (Python 3.11 or 3.12) — install the pinned set:\n"
              "  pip install -r requirements.txt\n"
              "If you already installed something newer by hand, force it back:\n"
              '  pip install "mediapipe==0.10.21" "numpy<2" "opencv-python<5" "opencv-contrib-python<5"\n')
    else:
        print(err[-1500:])
    sys.exit(1)


def boost_process():
    """Windows slows a process down once its window leaves the foreground: 'efficiency mode' (EcoQoS) caps the
    CPU clock and the sleep timer coarsens to ~16 ms. Both halve the frame rate when the preview is minimised
    behind Zoom — which is exactly when it matters. Opt out of both."""
    if platform.system() != "Windows":
        return
    import ctypes
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x8000)          # ABOVE_NORMAL_PRIORITY_CLASS

        class PowerThrottling(ctypes.Structure):
            _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]
        state = PowerThrottling(1, 0x1, 0)                         # EXECUTION_SPEED controlled, not throttled
        k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state))
    except Exception as e:
        print(f"(could not opt out of background throttling: {e})")


def build_detectors(model_paths):
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_paths["face_landmarker.task"]),
        running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True))
    # hands in front of the face are the ones we care about and the ones the detector is least sure of
    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_paths["hand_landmarker.task"]),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4, min_tracking_confidence=0.4))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_paths["pose_landmarker_lite.task"]),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))
    return face, hand, pose


class Camera(threading.Thread):
    """Reads the webcam continuously and keeps only the newest frame. Without this the main loop blocks
    in cap.read() until the driver has a frame, and any frames it couldn't keep up with queue in the
    driver's buffer — that queue is the lag you see between moving and the meme moving."""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap, self.frame, self.seq, self.ok = cap, None, 0, True
        self.lock = threading.Lock()
        self.start()

    def run(self):
        while self.ok:
            ok, frame = self.cap.read()
            if not ok:
                self.ok = False
                break
            with self.lock:
                self.frame, self.seq = frame, self.seq + 1

    def read(self, last_seq=None, timeout=0.5):
        """Newest frame; waits (briefly) for a frame newer than last_seq. Returns (ok, frame, seq)."""
        t0 = time.monotonic()
        while self.ok and (self.seq == last_seq or self.frame is None) and time.monotonic() - t0 < timeout:
            time.sleep(0.001)
        with self.lock:
            return self.ok and self.frame is not None, self.frame, self.seq


def open_camera(index, size):
    """DirectShow first: on Windows it hands frames over with less latency than the default Media Foundation."""
    for backend in ((cv2.CAP_DSHOW, "dshow"), (cv2.CAP_ANY, "default")):
        cap = cv2.VideoCapture(index, backend[0])
        if cap.isOpened() and "x" in size:
            w, h = size.lower().split("x")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = False, None
        if cap.isOpened():
            for _ in range(5):
                ok, frame = cap.read()
                if not ok:
                    break
        if ok:
            return cap, frame, backend[1]
        cap.release()
    return None, None, None


class Clock:
    """Strictly increasing timestamps for the life of a detector, recalibrations included."""

    def __init__(self):
        self.t0, self.last = time.monotonic(), -1

    def next(self):
        self.last = max(int((time.monotonic() - self.t0) * 1000), self.last + 1)
        return self.last


class Baseline:
    """Your resting face: a mean and a wobble for every channel."""

    def __init__(self, mean=None, sigma=None, samples=0, made=None):
        self.mean = mean or {}
        self.sigma = sigma or {}
        self.samples = samples
        self.made = made
        self.generic = not self.mean
        self.online = {}      # name -> [n, sum, sumsq] for channels the saved calibration doesn't know

    def _learn(self, name, value):
        """Geometric channels added after a calibration was saved: learn the resting value on the fly.
        Returns (mean, sigma) once enough frames are in, else None (the channel stays inert)."""
        st = self.online.setdefault(name, [0, 0.0, 0.0])
        if st[0] < ONLINE_MIN_FRAMES:      # learn from the first N frames only, then freeze: otherwise the
            st[0] += 1                      # expressions we are trying to detect widen their own sigma
            st[1] += value
            st[2] += value * value
            return None
        m = st[1] / st[0]
        return m, max(max(st[2] / st[0] - m * m, 0.0) ** 0.5, GEO_SIGMA_FLOOR)

    def z(self, name, value):
        """How far above your neutral this channel is, in standard deviations."""
        if name.startswith("geo") and (self.generic or name not in self.mean):
            learned = self._learn(name, value)
            return 0.0 if learned is None else (value - learned[0]) / learned[1]
        if self.generic:
            return (value - GENERIC_MEAN.get(name, 0.02)) / GENERIC_SIGMA
        m = self.mean.get(name)
        if m is None:
            return (value - GENERIC_MEAN.get(name, 0.02)) / GENERIC_SIGMA
        return (value - m) / self.sigma.get(name, SIGMA_CEIL)

    def raw(self, name):
        """Where this channel rests for you, raw (for floors that should mean 'this much above neutral')."""
        m = None if self.generic else self.mean.get(name)
        if m is None and name.startswith("geo"):
            st = self.online.get(name)
            return None if not st or st[0] < ONLINE_MIN_FRAMES else st[1] / st[0]
        return GENERIC_MEAN.get(name, 0.02) if m is None else m

    def learning(self, name):
        """Frames still needed before an online channel is usable (0 = ready)."""
        if not self.generic and name in self.mean:
            return 0
        return max(ONLINE_MIN_FRAMES - self.online.get(name, [0])[0], 0)

    @property
    def neutral_turn(self):
        return self.mean.get("turn_signed", 0.0) if not self.generic else 0.0

    @property
    def turn_thresh(self):
        """A turn counts only if it is bigger than T AND bigger than your normal head wandering."""
        if self.generic:
            return T["head_turn"]
        return max(T["head_turn"], TURN_SIGMA_K * self.sigma.get("turn_signed", 0.0))

    def save(self, path):
        with open(path, "w") as fh:
            json.dump({"version": CALIB_VERSION, "made": self.made, "samples": self.samples,
                       "mean": self.mean, "sigma": self.sigma}, fh, indent=1, sort_keys=True)

    @staticmethod
    def load(path):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return Baseline()
        if data.get("version") != CALIB_VERSION or not data.get("mean"):
            return Baseline()
        return Baseline(data["mean"], data.get("sigma", {}), data.get("samples", 0), data.get("made"))


class Collector:
    """Mean from the bored-face phase; sigma from bored face + normal-behaviour phase, around that mean.

    Phase 2 is what makes an off-axis camera survivable: looking down at your screen swings browDown
    and eyeSquint by a lot when the camera sits above you, and that swing must land inside sigma,
    not be read as an expression."""

    def __init__(self):
        self.n1, self.s1 = 0, {}
        self.n, self.s, self.ss = 0, {}, {}

    def add(self, face, phase=1):
        vals = list(face.bs.items()) + [("turn_signed", face.turn_signed)]
        self.n += 1
        if phase == 1:
            self.n1 += 1
        for name, v in vals:
            self.s[name] = self.s.get(name, 0.0) + v
            self.ss[name] = self.ss.get(name, 0.0) + v * v
            if phase == 1:
                self.s1[name] = self.s1.get(name, 0.0) + v

    def finish(self):
        mean, sigma = {}, {}
        for name in self.s:
            m = self.s1.get(name, 0.0) / max(self.n1, 1)
            var = max((self.ss[name] - 2 * m * self.s[name]) / self.n + m * m, 0.0)
            mean[name] = round(m, 5)
            floor = GEO_SIGMA_FLOOR if name.startswith("geo") else SIGMA_FLOOR
            sigma[name] = round(min(max(var ** 0.5, floor), SIGMA_CEIL), 5)
        var = max((self.ss.get("turn_signed", 0.0) - 2 * mean.get("turn_signed", 0.0) * self.s.get("turn_signed", 0.0))
                  / self.n + mean.get("turn_signed", 0.0) ** 2, 0.0)
        sigma["turn_signed"] = round(min(max(var ** 0.5, 0.03), 0.50), 5)
        return Baseline(mean, sigma, self.n, time.strftime("%Y-%m-%d %H:%M"))


def calibration_warnings(base):
    """Catch the two ways a calibration goes wrong: mid-expression, or fidgeting."""
    out = []
    if base.mean.get("jawOpen", 0) > 0.30:
        out.append("your mouth looks like it was open — don't talk during calibration")
    if max(base.mean.get("noseSneerLeft", 0), base.mean.get("noseSneerRight", 0)) > 0.15:
        out.append("your nose was scrunched — hold a bored face, not a reaction")
    if max(base.mean.get("browInnerUp", 0), base.mean.get("browOuterUpLeft", 0)) > 0.35:
        out.append("your eyebrows were up — relax them")
    pinned = sum(1 for k, v in base.sigma.items() if v >= SIGMA_CEIL)
    if pinned > 12:
        out.append("you moved a lot — sit still and try again for a tighter baseline")
    # Camera geometry. None of these break calibration, but they explain odd readings.
    if min(base.mean.get("browDownLeft", 0), base.mean.get("browDownRight", 0)) > 0.45:
        out.append("browDown rests very high: the camera looks down at you (or you look up under your brows) — "
                   "raise the camera to eye level if you can")
    if max(base.mean.get("eyeLookUpLeft", 0), base.mean.get("eyeLookUpRight", 0)) > 0.15:
        out.append("your eyes rest looking up — the camera is above your screen")
    if abs(base.mean.get("turn_signed", 0.0)) > 0.4:
        out.append(f"your head rests turned {abs(base.mean['turn_signed']):.2f} from the camera axis — "
                   "the camera is off to one side, or you calibrated looking at your screen")
    return out


def run_calibration(read, face_det, clock, args, W, H, window):
    """Phase 1: a bored face for CALIB_SECONDS (your neutral). Phase 2: behave normally for
    CALIB_RANGE_SECONDS — look at your screen, read, shift in the chair — so 'normal' movement widens
    sigma instead of firing memes later."""
    print(f"\nCalibrating. Phase 1 ({CALIB_SECONDS:.0f}s): sit how you normally sit, look at the "
          "camera, hold a bored face.\nBlinking is fine. Don't talk, smile or raise your eyebrows.\n"
          f"Phase 2 ({CALIB_RANGE_SECONDS:.0f}s): look around your screen, read something, move as you "
          "normally do — still no expressions, no talking.")
    col, start, seen_face = Collector(), time.monotonic(), 0
    total = CALIB_SECONDS + CALIB_RANGE_SECONDS
    while True:
        elapsed = time.monotonic() - start
        if elapsed > total:
            break
        phase = 1 if elapsed <= CALIB_SECONDS else 2
        ok, frame = read()
        if not ok:
            break
        if frame.shape[0] != H or frame.shape[1] != W:
            frame = cv2.resize(frame, (W, H))
        if not args.no_flip:
            frame = cv2.flip(frame, 1)
        small = cv2.resize(frame, (DETECT_WIDTH, DETECT_WIDTH * H // W), interpolation=cv2.INTER_AREA) \
            if W > DETECT_WIDTH else frame
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        fr = face_det.detect_for_video(mp_img, clock.next())
        face = Face(fr.face_landmarks[0], fr.face_blendshapes[0] if fr.face_blendshapes else None, W, H) \
            if fr.face_landmarks else None
        if face is not None:
            seen_face += 1
            # skip the first moments of each phase: people settle in / start moving
            settled = elapsed > CALIB_WARMUP if phase == 1 else elapsed > CALIB_SECONDS + 1.0
            if settled and face.bs:
                col.add(face, phase)
        draw_calibration(frame, elapsed, phase, col.n, face)
        cv2.imshow(window, frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            print("Calibration cancelled.")
            return None

    if col.n1 < CALIB_MIN_SAMPLES:
        print(f"Calibration failed: only {col.n1} usable frames"
              f"{' — your face was never detected' if not seen_face else ''}.\n"
              "  - light your face from the front, sit head-and-shoulders in frame, and try again")
        return None
    base = col.finish()
    print(f"Calibrated on {col.n1} neutral + {col.n - col.n1} normal-behaviour frames. Your neutral face:")
    for name in ("jawOpen", "noseSneerLeft", "browDownLeft", "mouthFrownLeft", "eyeSquintLeft", "turn_signed"):
        if name in base.mean:
            print(f"  {name:16s} {base.mean[name]:.3f} ± {base.sigma[name]:.3f}")
    print(f"  head turn fires above {base.turn_thresh:.2f}")
    for w in calibration_warnings(base):
        print(f"  ! {w}")
    return base


def draw_calibration(img, elapsed, phase, samples, face):
    H, W = img.shape[:2]
    total = CALIB_SECONDS + CALIB_RANGE_SECONDS
    left = max(0.0, (CALIB_SECONDS if phase == 1 else total) - elapsed)
    cv2.rectangle(img, (0, 0), (W, 96), (0, 0, 0), -1)
    cv2.putText(img, "CALIBRATING 1/2 - look at the camera, hold a bored face" if phase == 1 else
                "CALIBRATING 2/2 - look around your screen, move normally, no expressions",
                (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"{left:0.1f}s   {samples} frames" + ("" if face is not None else "   NO FACE"),
                (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255) if face is not None else (0, 140, 255), 2)
    done = int(W * min(elapsed / total, 1.0))
    cv2.rectangle(img, (0, 84), (done, 96), (0, 220, 0), -1)
    cv2.line(img, (int(W * CALIB_SECONDS / total), 84), (int(W * CALIB_SECONDS / total), 96), (255, 255, 255), 2)
    if face is not None:
        x0, y0, x1, y1 = face.box
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 220, 0), 1)


class Asset:
    """One reaction: a list of BGRA frames plus per-frame durations (ms) for GIFs."""

    def __init__(self, frames, durations):
        self.frames = frames
        self.durations = durations
        self.cum = np.cumsum(durations)
        self.total = int(self.cum[-1])
        h, w = frames[0].shape[:2]
        self.aspect = w / h
        self._cache = {}

    def frame_at(self, ms):
        if len(self.frames) == 1:
            return 0
        return int(np.searchsorted(self.cum, ms % self.total, side="right"))

    def scaled(self, idx, height):
        key = (idx, height)
        if key not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            w = max(1, int(round(height * self.aspect)))
            self._cache[key] = cv2.resize(self.frames[idx], (w, height), interpolation=cv2.INTER_AREA)
        return self._cache[key]


def to_bgra(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def placeholder(label):
    img = np.zeros((300, 300, 4), np.uint8)
    cv2.circle(img, (150, 150), 140, (0, 0, 255, 220), -1)
    cv2.putText(img, label, (12, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255, 255), 2)
    return Asset([img], [100])


IMAGE_EXTS = (".gif", ".png", ".jpg", ".jpeg")


def find_asset_files(pose):
    """Every file or folder that belongs to the pose: 'time_out.gif', 'x_time_out.png', a 'time_out/' sequence
    folder, and pool members 'time_out~1.gif', 'time_out~yanukovych.gif', 'time_out~2/' — one of the pool is
    picked at random each time the pose fires."""
    adir = os.path.join(HERE, "assets")
    if not os.path.isdir(adir):
        return []
    out = []
    for fn in sorted(os.listdir(adir)):
        path = os.path.join(adir, fn)
        stem, ext = os.path.splitext(fn) if not os.path.isdir(path) else (fn, "")
        named = stem == pose or stem.endswith("_" + pose) or stem.startswith(pose + "~")
        if named and (os.path.isdir(path) or ext.lower() in IMAGE_EXTS):
            out.append(path)
    return out


def find_asset_file(pose):
    files = find_asset_files(pose)
    return files[0] if files else None


def load_sequence(folder):
    """Numbered images shown one after another; '2_900.png' holds frame 2 for 900 ms."""
    frames, durations = [], []
    for fn in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in IMAGE_EXTS or ext.lower() == ".gif":
            continue
        img = cv2.imread(os.path.join(folder, fn), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        ms = stem.rsplit("_", 1)[-1]
        frames.append(to_bgra(img))
        durations.append(int(ms) if ms.isdigit() and "_" in stem else SEQ_MS)
    return frames, durations


def load_assets(pose):
    """All assets of a pose as a list (the random pool); [] switches the pose off."""
    paths = find_asset_files(pose)
    if not paths:
        print(f"  {pose:16s} no asset -> pose switched off")
        return []
    pool = [a for a in (load_asset(pose, p) for p in paths) if a is not None]
    if len(pool) > 1:
        print(f"  {pose:16s} = pool of {len(pool)}, picked at random")
    return pool


def load_asset(pose, path):
    frames, durations = [], []
    if os.path.isdir(path):
        frames, durations = load_sequence(path)
    elif path.lower().endswith(".gif"):
        from PIL import Image, ImageSequence
        with Image.open(path) as im:
            for f in ImageSequence.Iterator(im):
                frames.append(cv2.cvtColor(np.array(f.convert("RGBA")), cv2.COLOR_RGBA2BGRA))
                durations.append(max(20, int(f.info.get("duration", 100))))
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            frames, durations = [to_bgra(img)], [100]
    if not frames:
        print(f"  {pose:16s} could not read {os.path.basename(path)} -> placeholder")
        return placeholder(pose)
    kind = "sequence, " if os.path.isdir(path) else ""
    print(f"  {pose:16s} {os.path.basename(path)}  ({kind}{len(frames)} frame{'s' if len(frames) > 1 else ''})")
    return Asset(frames, durations)


def overlay(frame, sprite, x, y):
    """Alpha-composite BGRA sprite onto BGR frame at top-left (x, y), clipped to the frame."""
    H, W = frame.shape[:2]
    h, w = sprite.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return frame
    s = sprite[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (a * s[:, :, :3] + (1 - a) * roi).astype(np.uint8)
    return frame


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class Face:
    def __init__(self, lms, blendshapes, W, H):
        p = np.array([[l.x * W, l.y * H] for l in lms], np.float32)
        self.pts = p
        x0, y0 = p.min(0)
        x1, y1 = p.max(0)
        self.box = (int(x0), int(y0), int(x1), int(y1))
        self.w, self.h = float(x1 - x0), float(y1 - y0)
        self.center = ((x0 + x1) / 2, (y0 + y1) / 2)
        self.nose, self.chin, self.top = p[1], p[152], p[10]
        self.mouth = (p[13] + p[14]) / 2
        self.eye_y = float((p[33][1] + p[263][1]) / 2)
        cl, cr = p[234], p[454]
        # nose position between the cheek edges: 0 frontal, +-0.5 full profile. The raw ratio flattens
        # out near profile, so a camera off to one side gets a squashed, one-sided scale; atanh
        # straightens it back to roughly linear in head angle, so 'turned a bit more' means the same
        # whether you rest facing the lens or 30 degrees off it.
        t = (self.nose[0] - cl[0]) / max(cr[0] - cl[0], 1e-3) - 0.5
        self.turn_signed = float(np.arctanh(2 * max(-TURN_CLAMP, min(TURN_CLAMP, t))))
        self.bs = {c.category_name: c.score for c in (blendshapes or [])}
        # nose scrunch, geometrically: the upper lip rides up towards the nose tip. The noseSneer
        # blendshape reads a flat 0 on some faces/camera angles, this ratio doesn't. Bridge-to-chin
        # normalises away distance and (mostly) head pitch. Stored under a 'geo' key so calibration
        # records it like any other channel.
        if self.bs:
            self.bs["geoNoseLip"] = dist(p[1], p[0]) / max(dist(p[168], p[152]), 1e-3)

    def b(self, name):
        return self.bs.get(name, 0.0)


class Hand:
    def __init__(self, lms, W, H):
        p = np.array([[l.x * W, l.y * H] for l in lms], np.float32)
        self.palm = p[[0, 5, 9, 13, 17]].mean(0)
        self.thumb, self.index, self.middle = p[4], p[8], p[12]
        d = p[9] - p[0]
        self.horizontal = abs(d[0]) > 1.5 * abs(d[1])
        self.vertical = abs(d[1]) > 1.5 * abs(d[0])
        ext = [dist(p[0], p[t]) > 1.2 * dist(p[0], p[t - 2]) for t in (8, 12, 16, 20)]
        self.open = sum(ext) >= 3
        self.pointing = ext[0] and sum(ext[1:]) <= 1   # index out, the rest curled


class Body:
    """Upper-body pose: shoulders 11/12, elbows 13/14, wrists 15/16."""

    def __init__(self, lms, W, H):
        p = np.array([[l.x * W, l.y * H] for l in lms], np.float32)
        self.shoulders, self.elbows, self.wrists = p[[11, 12]], p[[13, 14]], p[[15, 16]]
        vis = [getattr(lms[i], "visibility", 1.0) for i in (11, 12, 13, 14)]
        self.seen = min(vis) > 0.5
        shoulder_y = float(self.shoulders[:, 1].mean())
        self.elbows_up = self.seen and bool((self.elbows[:, 1] < shoulder_y).all())
        # arms akimbo: both elbows pushed out past the shoulders and hanging below them. Hands
        # themselves are usually below the frame edge, so this is the only signal available.
        sw = max(abs(float(self.shoulders[0][0] - self.shoulders[1][0])), 1.0)
        mid_x = float(self.shoulders[:, 0].mean())
        out = [abs(float(e[0]) - mid_x) > 0.9 * sw for e in self.elbows]
        down = [float(e[1]) > shoulder_y + 0.25 * sw for e in self.elbows]
        self.hands_on_hips = self.seen and all(out) and all(down)


def tongue_score(frame, face, hands, jaw_ready):
    """Fraction of the mouth opening that reads pink: saturated and lit, unlike teeth or throat."""
    if not jaw_ready:
        return 0.0
    if any(dist(h.palm, face.mouth) < 0.7 * face.w for h in hands):
        return 0.0
    poly = face.pts[INNER_LIPS].astype(np.int32)
    x0, y0 = poly.min(0)
    x1, y1 = poly.max(0)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0
    x0, y0 = max(x0, 0), max(y0, 0)
    roi = frame[y0:y1 + 1, x0:x1 + 1]
    if roi.size == 0:
        return 0.0
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly - [x0, y0]], 255)
    k = max(3, int(0.15 * (y1 - y0)))
    mask = cv2.erode(mask, np.ones((k, k), np.uint8))
    n = int(np.count_nonzero(mask))
    if n < 40:
        return 0.0
    h, s, v = cv2.split(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
    pink = ((h < 12) | (h > 160)) & (s > 70) & (v > 110)
    return float(np.count_nonzero(pink & (mask > 0)) / n)


class Motion:
    """Smoothed hand speed across frames, in face-widths per frame."""

    def __init__(self):
        self.prev, self.energy, self.fw = [], 0.0, 200.0

    def update(self, hands, face):
        if face is not None:
            self.fw = max(face.w, 1.0)
        cur = [h.palm for h in hands]
        speed = 0.0
        if cur and self.prev:
            moved = [min(dist(c, p) for p in self.prev) for c in cur]
            moved = [m for m in moved if m < self.fw]
            if moved:
                speed = max(moved) / self.fw
        self.energy = 0.8 * self.energy + 0.2 * speed
        self.prev = cur
        return self.energy


class Gaze:
    """Counts left-right reversals of horizontal gaze: eyes darting side to side, head still or not."""

    def __init__(self):
        self.gx, self.extreme, self.direction, self.reversals = 0.0, 0.0, 0, []

    def update(self, face, now):
        raw = ((face.b("eyeLookOutRight") + face.b("eyeLookInLeft"))
               - (face.b("eyeLookOutLeft") + face.b("eyeLookInRight"))) / 2
        self.gx = 0.5 * self.gx + 0.5 * raw
        delta = self.gx - self.extreme
        if self.direction == 0:
            if abs(delta) >= T["gaze_swing"]:
                self.direction, self.extreme = (1 if delta > 0 else -1), self.gx
        elif delta * self.direction > 0:
            self.extreme = self.gx                       # still travelling the same way
        elif abs(delta) >= T["gaze_swing"]:
            self.direction, self.extreme = -self.direction, self.gx
            self.reversals.append(now)
        self.reversals = [t for t in self.reversals if now - t <= T["gaze_window"]]
        return len(self.reversals)


class Wave:
    """Counts side-to-side reversals of an open palm raised beside the head: 'hello there'."""

    def __init__(self):
        self.extreme, self.direction, self.reversals, self.last_seen = None, 0, [], 0.0
        self.prev_x, self.speed = None, 0.0      # speed: face-widths per frame, so hand_up can insist on 'held still'

    def update(self, hands, face, now):
        fw = max(face.w, 1.0)
        raised = [h for h in hands if h.open and h.palm[1] < face.nose[1] and abs(h.palm[0] - face.nose[0]) > 0.6 * fw]
        if not raised or now - self.last_seen > 0.5:          # hand dropped: start over
            self.extreme, self.direction, self.reversals, self.prev_x = None, 0, [], None
        if not raised:
            self.speed = 0.0
            return 0
        self.last_seen = now
        x = float(raised[0].palm[0]) / fw
        self.speed = 0.0 if self.prev_x is None else 0.5 * self.speed + 0.5 * abs(x - self.prev_x)
        self.prev_x = x
        if self.extreme is None:
            self.extreme = x
        delta = x - self.extreme
        if self.direction == 0:
            if abs(delta) >= WAVE_SWING:
                self.direction, self.extreme = (1 if delta > 0 else -1), x
        elif delta * self.direction > 0:
            self.extreme = x
        elif abs(delta) >= WAVE_SWING:
            self.direction, self.extreme = -self.direction, x
            self.reversals.append(now)
        self.reversals = [t for t in self.reversals if now - t <= WAVE_WINDOW]
        return len(self.reversals)


def measure(face, base):
    """Every expression channel, raw and in sigma above your own neutral."""
    zpair = lambda n: (base.z(n + "Left", face.b(n + "Left")) + base.z(n + "Right", face.b(n + "Right"))) / 2
    pair = lambda n: (face.b(n + "Left") + face.b(n + "Right")) / 2
    rest = lambda n: (base.raw(n + "Left") + base.raw(n + "Right")) / 2
    geo = face.b("geoNoseLip")
    geo_rest = base.raw("geoNoseLip")
    shrink = (geo_rest - geo) / geo_rest if geo_rest else 0.0     # fraction the nose-lip gap closed
    z_geo = -base.z("geoNoseLip", geo)     # always called: this is also what teaches the channel its neutral
    if pair("mouthSmile") >= 0.3:
        shrink = 0.0      # a smile lifts the upper lip too; only count the gap closing on a straight mouth
    m = {
        "jaw": face.b("jawOpen"), "z_jaw": base.z("jawOpen", face.b("jawOpen")),
        "sneer": max(pair("noseSneer"), shrink),
        "z_sneer": max(zpair("noseSneer"), z_geo if shrink else 0.0),
        "nose_shrink": shrink, "nose_learning": base.learning("geoNoseLip"),
        "z_brow": zpair("browDown"), "z_frown": zpair("mouthFrown"), "z_lip": zpair("mouthUpperUp"),
        # squint floor is 'this much above YOUR resting squint': some faces (and cameras looking down
        # at you) rest at 0.5 and would pass a fixed floor without doing anything
        "squint": max(pair("eyeSquint") - rest("eyeSquint"), pair("eyeBlink") - rest("eyeBlink")),
        "z_squint": max(zpair("eyeSquint"), zpair("eyeBlink")),
        "turn": abs(face.turn_signed - base.neutral_turn),
        "turn_thresh": base.turn_thresh,
        # eyes dropped to read the screen register as a half-blink/squint; that's reading, not suspicion
        "look_down": pair("eyeLookDown") - rest("eyeLookDown"),
        # the 'NI-' of ni-hu-ya: lips stretched wide sideways (an 'ee' grimace, teeth showing) with the
        # brows up. A real smile stretches the lips too but pulls the eyes into a squint, not the brows up.
        "stretch": max(pair("mouthSmile"), pair("mouthStretch")),
        "z_stretch": max(zpair("mouthSmile"), zpair("mouthStretch")),
        "brow_up": face.b("browInnerUp"), "z_brow_up": base.z("browInnerUp", face.b("browInnerUp")),
    }
    cap = lambda v: min(v, Z_CAP)
    m["z_disgust"] = 2 * cap(m["z_sneer"]) + cap(m["z_brow"]) + cap(m["z_frown"]) + cap(m["z_lip"])
    return m


def over(key, m, zkey, rawkey):
    """Sigma above your neutral AND a raw floor, so a tiny sigma can't become a hair trigger."""
    return m[zkey] >= Z[key] and m[rawkey] >= FLOOR[key]


def decide(face, hands, body, tongue, gesture, m, last_face=None):
    """Return (pose or None, debug dict). last_face: where the face was a moment ago, for when a hand hides it."""
    d = {"hands": len(hands)}
    if face is None:
        if hands and last_face is not None:
            spot = (last_face.nose + last_face.mouth) / 2
            if any(dist(h.palm, spot) < 0.6 * last_face.w for h in hands):
                return "cover_nose", d          # the hand over the nose is exactly what made the face vanish
        gone = not hands and (body is None or not body.seen)
        return ("spin" if gone else None), d

    fw = face.w
    near = lambda a, b, k: dist(a, b) < k * fw
    elbows_up = bool(body and body.elbows_up)
    d.update(m, gesture=gesture, tongue=tongue, elbows_up=elbows_up)
    screaming = over("scream_jaw", m, "z_jaw", "jaw")

    if len(hands) >= 2:
        a, b = hands[0], hands[1]
        for top, under in ((a, b), (b, a)):
            if top.horizontal and under.vertical and top.palm[1] < under.palm[1] \
                    and near(under.middle, top.palm, 0.6):
                return "time_out", d
        if near(a.index, b.index, 0.3) and near(a.thumb, b.thumb, 0.3) \
                and (a.index[1] + b.index[1]) < (a.thumb[1] + b.thumb[1]):
            return "heart", d
        if near(a.palm, face.mouth, 0.6) and near(b.palm, face.mouth, 0.6):
            return "cover_nose", d
        # 'NI-HU-YA': two palms held up facing each other, side by side, above the chin
        gap, dy = abs(a.palm[0] - b.palm[0]), abs(a.palm[1] - b.palm[1])
        if a.vertical and b.vertical and 0.3 * fw < gap < 2.5 * fw and dy < 0.6 * fw \
                and max(a.palm[1], b.palm[1]) < face.chin[1] + 0.5 * face.h \
                and not near(a.palm, face.mouth, 0.7) and not near(b.palm, face.mouth, 0.7) \
                and gesture < T["hands_fast"]:       # not sweeping through: hands leaving the nose look the same
            return "nihuya", d
        on_head = lambda h: (h.palm[1] < face.eye_y and abs(h.palm[0] - face.nose[0]) < 1.1 * fw
                             and h.palm[1] > face.top[1] - 0.8 * face.h)
        if on_head(a) and on_head(b) and screaming:
            return "crashing_out", d

    # one palm flat over nose+mouth: the detector often sees only one of two hands in front of a face.
    # Checked before the elbows-up poses, which a hand raised to the face also satisfies.
    under_nose = (face.nose + face.mouth) / 2
    for h in hands:
        if near(h.palm, under_nose, 0.45):
            return "cover_nose", d

    near_head = lambda h: abs(h.palm[0] - face.nose[0]) < 1.3 * fw and h.palm[1] < face.eye_y + 0.3 * face.h
    if elbows_up and hands and all(near_head(h) for h in hands):
        return ("crashing_out" if screaming else "dance"), d
    if body and body.hands_on_hips and not hands:
        return "hands_on_hips", d

    if m.get("wave_reversals", 0) >= WAVE_REVERSALS:
        return "hello", d
    for h in hands:
        if near(h.thumb, face.nose, 0.35) and near(h.index, face.nose, 0.35) and near(h.thumb, h.index, 0.3):
            return "nose_closed", d
        if near(h.index, face.mouth, 0.22) and not near(h.palm, face.mouth, 0.3):
            return "flirty", d
        if h.open and h.palm[1] < face.nose[1] and abs(h.palm[0] - face.nose[0]) > 0.8 * fw \
                and m.get("wave_speed", 0.0) < 0.04:          # held still — a moving palm is a wave in progress
            return "hand_up", d
        # Pepe Silvia: index finger jabbing at something off to the side, at head height
        if h.pointing and abs(h.index[0] - face.nose[0]) > 1.2 * fw and h.index[1] < face.chin[1]:
            return "talking_to_wall", d

    if tongue > T["tongue"]:
        return "tongue_out", d
    if over("jaw_open", m, "z_jaw", "jaw"):
        return "open_mouth", d
    # a real laugh crinkles the eyes and drops the jaw a little; checked before 'NI', which is the same
    # wide mouth with the brows up and the eyes open
    if m["z_stretch"] >= Z["laugh"] and m["stretch"] >= FLOOR["laugh"] \
            and (m["squint"] >= FLOOR["laugh_squint"] or m["jaw"] >= FLOOR["laugh_jaw"]):
        return "laugh", d
    if over("stretch", m, "z_stretch", "stretch") and over("brow_up", m, "z_brow_up", "brow_up"):
        return "nihuya", d
    if over("sneer", m, "z_sneer", "sneer") or m["z_disgust"] >= Z["disgust"]:
        return "disgusted", d
    if m["turn"] > m["turn_thresh"] and over("squint", m, "z_squint", "squint") and m["look_down"] < T["look_down"]:
        return "suspicious", d
    if m.get("gaze_reversals", 0) >= GAZE_REVERSALS:
        return "shifty_eyes", d
    return None, d


class Brain(threading.Thread):
    """Detection and decisions, in their own thread, always on the newest frame. The main loop only draws
    and sends video, so the call sees the camera's own frame rate whatever the detectors manage — the
    detectors' speed now only sets how quickly a meme reacts, not how smooth you look."""

    def __init__(self, cam, detectors, clock, base, assets, args, W, H, diag):
        super().__init__(daemon=True)
        self.cam, (self.face_det, self.hand_det, self.pose_det) = cam, detectors
        self.clock, self.base, self.assets, self.args, self.W, self.H, self.diag = clock, base, assets, args, W, H, diag
        self.lock = threading.Lock()
        self.state = dict(shown=None, shown_since=0.0, asset=None, center=np.array([W / 2, H / 2], np.float32),
                          h=H * 0.45, raw=None, dbg={}, face=None, hands=[], body=None, fps=0.0)
        self.forced, self.forced_until = None, 0.0
        self.running = True
        self.start()

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def force(self, pose, until):
        self.forced, self.forced_until = pose, until

    def stop(self):
        self.running = False
        self.join(timeout=2.0)

    def run(self):
        W, H, args = self.W, self.H, self.args
        pool = ThreadPoolExecutor(max_workers=2)
        motion, gaze, wave = Motion(), Gaze(), Wave()
        seq, frame_no, body = None, 0, None
        fps, last_frame = 0.0, time.monotonic()
        last_face, last_face_at = None, 0.0
        shown, shown_since, hold_until, armed_since, asset = None, 0.0, 0.0, {}, None
        sm_center, sm_h = np.array([W / 2, H / 2], np.float32), H * 0.45
        try:
            while self.running:
                ok, frame, seq = self.cam.read(seq)
                if not ok:
                    break
                frame_no += 1
                if frame.shape[0] != H or frame.shape[1] != W:
                    frame = cv2.resize(frame, (W, H))
                if not args.no_flip:
                    frame = cv2.flip(frame, 1)     # detectors work in the mirrored view, same as the preview

                ts = self.clock.next()
                # detectors see a DETECT_WIDTH-wide copy: they resize to ~256 px internally anyway, and landmarks
                # come back normalised, so the full-resolution frame goes to the call at no detection cost
                small = cv2.resize(frame, (DETECT_WIDTH, DETECT_WIDTH * H // W), interpolation=cv2.INTER_AREA) \
                    if W > DETECT_WIDTH else frame
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                # the three detectors are independent C++ graphs that release the GIL: run them side by side
                hand_job = pool.submit(self.hand_det.detect_for_video, mp_img, ts)
                pose_job = pool.submit(self.pose_det.detect_for_video, mp_img, ts) if frame_no % POSE_EVERY == 0 else None
                fr = self.face_det.detect_for_video(mp_img, ts)
                hr = hand_job.result()
                face = Face(fr.face_landmarks[0], fr.face_blendshapes[0] if fr.face_blendshapes else None, W, H) \
                    if fr.face_landmarks else None
                hands = [Hand(h, W, H) for h in hr.hand_landmarks]
                if pose_job is not None:
                    pr = pose_job.result()
                    body = Body(pr.pose_landmarks[0], W, H) if pr.pose_landmarks else None
                now = time.monotonic()
                fps = 0.9 * fps + 0.1 / max(now - last_frame, 1e-3)
                last_frame = now

                m = measure(face, self.base) if face is not None else {}
                tongue = tongue_score(frame, face, hands,
                                      over("tongue_jaw", m, "z_jaw", "jaw")) if face is not None else 0.0
                gesture = motion.update(hands, face)
                if face is not None:
                    m["gaze_reversals"] = gaze.update(face, now)
                    m["wave_reversals"] = wave.update(hands, face, now)
                    m["wave_speed"] = wave.speed
                    last_face, last_face_at = face, now
                recent = last_face if now - last_face_at < 1.0 else None
                matched, dbg = decide(face, hands, body, tongue, gesture, m, recent)
                raw = matched if matched in self.assets else None    # a pose without an asset is switched off

                fired = None
                if raw:
                    armed_since = {raw: armed_since.get(raw, now)}
                    if now - armed_since[raw] >= DELAY.get(raw, DELAY_DEFAULT):
                        fired = raw
                else:
                    armed_since = {}
                if self.forced in self.assets and now < self.forced_until:
                    fired = self.forced
                if fired:
                    if fired != shown:
                        shown_since, asset = now, random.choice(self.assets[fired])
                    shown, hold_until = fired, now + HOLD_SECONDS
                elif now >= hold_until:
                    shown = None
                if self.diag:
                    self.diag.log(matched, fired, shown, face, hands, body, dbg)   # raw = what matched, asset or not

                if face is not None:
                    sm_center = 0.7 * sm_center + 0.3 * np.array(face.center, np.float32)
                    sm_h = 0.7 * sm_h + 0.3 * face.h * FACE_SCALE
                with self.lock:
                    self.state = dict(shown=shown, shown_since=shown_since, asset=asset, center=sm_center.copy(),
                                      h=sm_h, raw=raw, dbg=dbg, face=face, hands=hands, body=body, fps=fps)
        except Exception:
            import traceback
            print("\nDetection thread crashed — video keeps going, reactions stopped:")
            traceback.print_exc()
            with self.lock:
                self.state = dict(self.state, shown=None, raw=None, fps=0.0)
        finally:
            pool.shutdown(wait=False)


def draw_hud(img, shown, raw, d, face, hands, body, base, fps=0.0, k=1.0, render_fps=0.0):
    """k: preview scale relative to the frame the landmarks were measured on."""
    if face:
        x0, y0, x1, y1 = face.box
        cv2.rectangle(img, (int(x0 * k), int(y0 * k)), (int(x1 * k), int(y1 * k)), (0, 255, 0), 1)
    for h in hands:
        cv2.circle(img, (int(h.palm[0] * k), int(h.palm[1] * k)), 6, (0, 200, 255), -1)
    if body and body.seen:
        for pt in np.vstack([body.shoulders, body.elbows]):
            cv2.circle(img, (int(pt[0] * k), int(pt[1] * k)), 6, (255, 120, 0), -1)
    g = d.get
    lines = [
        (f"showing: {shown or '-'}   raw: {raw or '-'}   hands: {g('hands', 0)}"
         f"   elbows up: {'Y' if g('elbows_up') else 'n'}   video {render_fps:.0f} fps / detect {fps:.0f} fps",
         (0, 255, 0)),
        (f"jaw {g('jaw', 0):.2f} = {g('z_jaw', 0):+.1f}s/{Z['jaw_open']:.0f}   "
         f"squint +{g('squint', 0):.2f} = {g('z_squint', 0):+.1f}s/{Z['squint']:.0f}   "
         f"tongue {g('tongue', 0):.2f}   turn {g('turn', 0):.2f}/{g('turn_thresh', T['head_turn']):.2f}   "
         f"lookdn {g('look_down', 0):+.2f}   gaze rev {g('gaze_reversals', 0)}/{GAZE_REVERSALS}   "
         f"wave {g('wave_reversals', 0)}/{WAVE_REVERSALS}   "
         f"gesture {g('gesture', 0):.3f}", (0, 255, 0)),
        (f"disgust {g('z_disgust', 0):+.1f}s/{Z['disgust']:.0f} = 2x sneer {g('z_sneer', 0):+.1f}"
         f"(nose {g('nose_shrink', 0):+.2f}{' learning ' + str(g('nose_learning')) if g('nose_learning') else ''}) "
         f"+ brow {g('z_brow', 0):+.1f} + frown {g('z_frown', 0):+.1f} + lip {g('z_lip', 0):+.1f}   "
         f"stretch {g('stretch', 0):.2f}={g('z_stretch', 0):+.1f}s/{Z['stretch']:.0f} "
         f"browup {g('brow_up', 0):.2f}={g('z_brow_up', 0):+.1f}s/{Z['brow_up']:.0f}", (0, 255, 0)),
        (("NOT CALIBRATED - generic baseline, everything is harder to trigger. press 'c'"
          if base.generic else
          f"calibrated {base.made} on {base.samples} frames   (s = sigma above your neutral)"),
         (0, 140, 255) if base.generic else (200, 200, 200)),
        ("keys: q quit  d hud  c recalibrate  1-9 0 - = [ ] test poses", (0, 255, 0)),
    ]
    for i, (t, colour) in enumerate(lines):
        y = 24 + 22 * i
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)


DIAG_COLS = ["t", "raw", "fired", "shown", "face", "hands", "body", "elbows_up", "jaw", "z_jaw", "squint", "z_squint",
             "sneer", "z_sneer", "z_brow", "z_frown", "z_lip", "z_disgust", "turn_signed", "turn", "turn_thresh",
             "stretch", "z_stretch", "brow_up", "z_brow_up", "look_down", "gaze_reversals", "nose_shrink",
             "wave_reversals", "tongue", "gesture"]


class Diag:
    """Per-frame CSV of every number decide() saw, plus a per-pose tally at exit — the answer to 'why did that fire?'."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fh = open(path, "w", newline="")
        self.w = csv.writer(self.fh)
        self.w.writerow(DIAG_COLS)
        self.t0, self.raw_frames, self.fires, self.last_fired = time.monotonic(), {}, {}, None

    def log(self, raw, fired, shown, face, hands, body, d):
        row = {"t": round(time.monotonic() - self.t0, 3), "raw": raw or "", "fired": fired or "", "shown": shown or "",
               "face": int(face is not None), "hands": len(hands), "body": int(bool(body and body.seen)),
               "turn_signed": round(face.turn_signed, 3) if face is not None else ""}
        for k in DIAG_COLS:
            if k not in row:
                v = d.get(k, "")
                row[k] = round(v, 3) if isinstance(v, float) else v
        self.w.writerow([row[k] for k in DIAG_COLS])
        if raw:
            self.raw_frames[raw] = self.raw_frames.get(raw, 0) + 1
        if fired and fired != self.last_fired:
            self.fires[fired] = self.fires.get(fired, 0) + 1
        self.last_fired = fired

    def close(self):
        self.fh.close()
        print(f"\nDiag written to {self.path}")
        for p in POSES:
            if self.raw_frames.get(p) or self.fires.get(p):
                print(f"  {p:16s} matched {self.raw_frames.get(p, 0):5d} frames, fired {self.fires.get(p, 0)} time(s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="webcam index (try 1 if 0 is your iPhone)")
    ap.add_argument("--calibrate", action="store_true", help="learn your neutral face, save it, and exit")
    ap.add_argument("--no-calibration", action="store_true", help="ignore calibration.json; use the generic baseline")
    ap.add_argument("--no-vcam", action="store_true", help="preview only; don't start the virtual camera")
    ap.add_argument("--skip-check", action="store_true", help="skip the MediaPipe startup check")
    ap.add_argument("--size", default="1280x720", help="capture size, e.g. 1280x720 (what the call sees; "
                                                       "detection always runs at DETECT_WIDTH so this costs nothing)")
    ap.add_argument("--no-flip", action="store_true", help="don't mirror the image")
    ap.add_argument("--diag", action="store_true", help="write diag.csv (one row per frame) and a per-pose tally on exit")
    args = ap.parse_args()

    calib_path = os.path.join(HERE, CALIB_FILE)
    base = Baseline() if args.no_calibration else Baseline.load(calib_path)
    if base.generic and not args.calibrate and not args.no_calibration:
        print(f"No usable {CALIB_FILE}. Running on the generic baseline — everything is harder to\n"
              f"trigger than it should be. Run:  python {os.path.basename(__file__)} --calibrate")

    model_paths = ensure_models()
    if not args.skip_check:
        preflight(model_paths["face_landmarker.task"])

    cap, frame, backend = open_camera(args.camera, args.size)
    if cap is None:
        sys.exit(f"Could not read from camera {args.camera}.\n"
                 "  - try --camera 1\n"
                 "  - System Settings > Privacy & Security > Camera: allow your terminal app, then re-run")
    H, W = frame.shape[:2]
    print(f"Camera {args.camera}: {W}x{H} ({backend})")

    clock = Clock()
    window = "it's giving v2  (q quit, d HUD, c recalibrate, 1-9 0 - = [ ] test)"

    if args.calibrate:
        face_det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_paths["face_landmarker.task"]),
            running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True))
        try:
            new = run_calibration(cap.read, face_det, clock, args, W, H, window)
        finally:
            face_det.close()
            cap.release()
            cv2.destroyAllWindows()
        if new is None:
            sys.exit(1)
        new.save(calib_path)
        print(f"Saved {CALIB_FILE}. Now run:  python {os.path.basename(__file__)}")
        return

    print("Assets:")
    assets = {pose: pool for pose in POSES if (pool := load_assets(pose))}
    if not assets:
        sys.exit("No assets at all — put at least one image named after a pose into assets/")

    vcam = None
    if not args.no_vcam:
        try:
            import pyvirtualcam
            vcam = pyvirtualcam.Camera(width=W, height=H, fps=30, fmt=pyvirtualcam.PixelFormat.BGR)
            print(f"Virtual camera: '{vcam.device}'  <- pick this camera in Zoom / Meet")
        except Exception as e:
            print(f"Virtual camera unavailable ({e}). Preview-only.")

    face_det, hand_det, pose_det = build_detectors(model_paths)
    boost_process()
    cam, seq = Camera(cap), None

    def read_latest():
        nonlocal seq
        ok, f, seq = cam.read(seq)
        return ok, f
    show_hud, render_fps, last_render = True, 0.0, time.monotonic()
    # one file per session, so sessions accumulate for tuning instead of overwriting each other
    diag = Diag(os.path.join(HERE, "diag", time.strftime("diag_%Y%m%d_%H%M.csv"))) if args.diag else None
    brain = Brain(cam, (face_det, hand_det, pose_det), clock, base, assets, args, W, H, diag)
    print("Running. Focus the preview window: q quit, d HUD, c recalibrate, 1-9 0 - = [ ] test a pose")

    try:
        while True:
            ok, frame = read_latest()
            if not ok:
                print("Camera stopped returning frames.")
                break
            frame = cv2.resize(frame, (W, H)) if frame.shape[0] != H or frame.shape[1] != W else frame.copy()
            raw_frame = frame                      # unmirrored: what the other side of the call should see
            if not args.no_flip:
                frame = cv2.flip(frame, 1)         # mirrored: what you see
            st = brain.snapshot()
            now = time.monotonic()
            render_fps = 0.9 * render_fps + 0.1 / max(now - last_render, 1e-3)
            last_render = now

            if st["shown"] and st["asset"] is not None:
                asset = st["asset"]
                idx = asset.frame_at(int((now - st["shown_since"]) * 1000))
                h = int(min(st["h"], H * 0.98, (W * 0.98) / asset.aspect)) // 8 * 8
                sprite = asset.scaled(idx, max(h, 8))
                sh, sw = sprite.shape[:2]
                x, y = int(st["center"][0] - sw / 2), int(st["center"][1] - sh / 2 - 0.05 * sh)
                overlay(frame, sprite, x, y)
                if vcam and raw_frame is not frame:
                    # the call gets the unmirrored picture with the same (unmirrored) sprite at the mirrored spot,
                    # so text on the meme reads correctly for everyone else
                    overlay(raw_frame, sprite, W - x - sw, y)

            if vcam:
                vcam.send(raw_frame)
                vcam.sleep_until_next_frame()

            preview = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_WIDTH * H // W)) if W > PREVIEW_WIDTH else frame.copy()
            if show_hud:
                draw_hud(preview, st["shown"], st["raw"], st["dbg"], st["face"], st["hands"], st["body"], brain.base,
                         st["fps"], PREVIEW_WIDTH / W if W > PREVIEW_WIDTH else 1.0, render_fps)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_hud = not show_hud
            elif key == ord("c"):
                brain.stop()                       # the calibration needs the face detector to itself
                new = run_calibration(read_latest, face_det, clock, args, W, H, window)
                if new is not None:
                    new.save(calib_path)
                    print(f"Saved {CALIB_FILE}.")
                brain = Brain(cam, (face_det, hand_det, pose_det), clock, new or brain.base, assets, args, W, H, diag)
            elif 0 < key < 256 and chr(key) in TEST_KEYS:
                brain.force(POSES[TEST_KEYS.index(chr(key))], now + 2.0)
    finally:
        brain.stop()
        if diag:
            diag.close()
        cam.ok = False
        cap.release()
        face_det.close()
        hand_det.close()
        pose_det.close()
        if vcam:
            vcam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
