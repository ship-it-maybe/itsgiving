"""Put the calls-based calibration back after a 15-second one overwrote it.

calls/calibration_calls.json was built from ~95 min of real Zoom calls on UNmirrored video, so Left/Right
channels are swapped and turn_signed is negated relative to what the app sees (it mirrors the webcam).
Two things are taken from the current (webcam) calibration.json instead, because they depend on the webcam
framing rather than the face: the head-turn neutral (turn_signed) and the geometric nose channel (geo*).
The overwritten webcam calibration is kept as calibration_webcam.json.

    python restore_calls_calibration.py
"""
import json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import its_giving_v2 as g

HERE = os.path.dirname(os.path.abspath(__file__))
CUR, CALLS, BACKUP = (os.path.join(HERE, p) for p in ("calibration.json", os.path.join("calls", "calibration_calls.json"),
                                                      "calibration_webcam.json"))

def flip(d):
    out = {}
    for k, v in d.items():
        if k.endswith("Left"):
            out[k[:-4] + "Right"] = v
        elif k.endswith("Right"):
            out[k[:-5] + "Left"] = v
        else:
            out[k] = v
    return out

calls = json.load(open(CALLS))
web = json.load(open(CUR)) if os.path.exists(CUR) else {"mean": {}, "sigma": {}}
if "calls" in web.get("made", ""):
    sys.exit("calibration.json is already the calls-based one — nothing to do.")
if web["mean"]:
    shutil.copy(CUR, BACKUP)
mean, sigma = flip(calls["mean"]), flip(calls["sigma"])
mean["turn_signed"] = round(-calls["mean"]["turn_signed"], 5)
for k in list(web["mean"]):
    if k == "turn_signed" or k.startswith("geo"):
        mean[k], sigma[k] = web["mean"][k], web["sigma"][k]
json.dump({"version": g.CALIB_VERSION, "made": calls["made"] + " (from 95 min of real calls)",
           "samples": calls["samples"], "mean": mean, "sigma": sigma}, open(CUR, "w"), indent=1, sort_keys=True)
b = g.Baseline.load(CUR)
print(f"calibration.json <- calls ({b.samples} frames); turn neutral {b.mean['turn_signed']:+.2f}, "
      f"threshold {b.turn_thresh:.2f}; nose channel {'from webcam' if 'geoNoseLip' in b.mean else 'will self-learn'}"
      + (f"; webcam calibration saved as {os.path.basename(BACKUP)}" if web["mean"] else ""))
