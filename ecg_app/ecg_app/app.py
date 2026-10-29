# ECG beat classifier web app: upload MIT-BIH record files, get N/S/V/F predictions
import os, json, uuid, tempfile
import numpy as np
import pandas as pd
import pywt
import wfdb
import joblib
from scipy.signal import butter, sosfiltfilt
from scipy.stats import skew, kurtosis
from flask import Flask, request, jsonify, render_template

# imported so joblib can unpickle the pipelines
import imblearn, xgboost  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "best_models.joblib")

FS = 360
PRE = POST = 90
CLASSES = ["N", "S", "V", "F"]
LABEL_ID = {c: i for i, c in enumerate(CLASSES)}
AAMI = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
        "A": "S", "a": "S", "J": "S", "S": "S",
        "V": "V", "E": "V", "F": "F",
        "/": "Q", "f": "Q", "Q": "Q"}

# same filter as the notebook: 3rd-order band-pass 0.5-40 Hz
sos = butter(3, [0.5, 40], btype="band", fs=FS, output="sos")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
models = joblib.load(MODEL_PATH)      # dict: LogReg, RandomForest, XGBoost, SVM
CACHE = {}                            # upload id -> filtered signal + beats


def morph_features(beat):
    # 17 wavelet approx coeffs + 4 detail energies + 8 simple stats
    coeffs = pywt.wavedec(beat, "db4", level=4)
    energy = [np.sum(c ** 2) for c in coeffs[1:]]
    stats = [beat.max(), beat.min(), beat.max() - beat.min(), beat.std(),
             skew(beat), kurtosis(beat), np.argmax(beat), np.argmin(beat)]
    return np.concatenate([coeffs[0], energy, stats])


def extract_beats(sig, ann):
    # same logic as the notebook's extract_record, but returns sample positions too
    idx = [i for i, s in enumerate(ann.symbol) if s in AAMI]
    peaks = ann.sample[idx]
    syms = [AAMI[ann.symbol[i]] for i in idx]
    rr = np.diff(peaks) / FS
    pre_rr = np.r_[np.median(rr), rr]
    post_rr = np.r_[rr, np.median(rr)]
    local_avg = pd.Series(pre_rr).rolling(10, min_periods=1).mean().values
    rec_avg = rr.mean()

    X, samples, truth = [], [], []
    for k, (p, lab) in enumerate(zip(peaks, syms)):
        if lab not in LABEL_ID or p - PRE < 0 or p + POST > len(sig):
            continue
        beat = sig[p - PRE:p + POST]
        beat = (beat - beat.mean()) / (beat.std() + 1e-8)
        rr_feats = [pre_rr[k], post_rr[k], local_avg[k],
                    pre_rr[k] / rec_avg, post_rr[k] / rec_avg, pre_rr[k] / local_avg[k]]
        X.append(np.concatenate([rr_feats, morph_features(beat)]))
        samples.append(int(p))
        truth.append(lab)
    return np.array(X), samples, truth


@app.route("/")
def home():
    return render_template("index.html", models=list(models.keys()))


@app.route("/predict", methods=["POST"])
def predict():
    files = request.files.getlist("files")
    by_ext = {}
    for f in files:
        name = os.path.basename(f.filename or "")
        stem, ext = os.path.splitext(name)
        by_ext[ext.lower()] = (stem, f)
    missing = [e for e in (".hea", ".dat", ".atr") if e not in by_ext]
    if missing:
        return jsonify(error="Missing file(s): " + ", ".join(missing) + ". Upload .hea, .dat and .atr together."), 400
    stems = {by_ext[e][0] for e in (".hea", ".dat", ".atr")}
    if len(stems) != 1:
        return jsonify(error="The three files must share one record name (e.g. 200.hea, 200.dat, 200.atr)."), 400
    rec = stems.pop()
    if not rec.replace("_", "").isalnum():
        return jsonify(error="Invalid record name."), 400

    model_name = request.form.get("model", "RandomForest")
    if model_name not in models:
        return jsonify(error="Unknown model."), 400

    try:
        with tempfile.TemporaryDirectory() as tmp:
            for e in (".hea", ".dat", ".atr"):
                by_ext[e][1].save(os.path.join(tmp, rec + e))
            path = os.path.join(tmp, rec)
            hdr = wfdb.rdheader(path)
            ch = "MLII" if "MLII" in hdr.sig_name else hdr.sig_name[0]
            r = wfdb.rdrecord(path, channel_names=[ch])
            ann = wfdb.rdann(path, "atr")
            sig = sosfiltfilt(sos, r.p_signal[:, 0])
            X, samples, truth = extract_beats(sig, ann)
    except Exception as ex:
        return jsonify(error="Could not read this record: %s" % ex), 400

    if len(X) == 0:
        return jsonify(error="No usable beats found in this record."), 400

    pred = models[model_name].predict(X)
    pred = [CLASSES[int(p)] for p in pred]
    correct = sum(p == t for p, t in zip(pred, truth))

    uid = uuid.uuid4().hex
    if len(CACHE) > 20:
        CACHE.pop(next(iter(CACHE)))
    CACHE[uid] = sig.astype(np.float32)

    return jsonify(
        id=uid, record=rec, channel=ch, model=model_name, fs=FS,
        duration=round(len(sig) / FS, 1), n_beats=len(pred),
        counts={c: pred.count(c) for c in CLASSES},
        true_counts={c: truth.count(c) for c in CLASSES},
        accuracy=round(correct / len(pred), 4),
        beats=[{"s": s, "p": p, "t": t} for s, p, t in zip(samples, pred, truth)],
    )


@app.route("/signal/<uid>")
def signal(uid):
    # return a window of the filtered signal for the viewer
    sig = CACHE.get(uid)
    if sig is None:
        return jsonify(error="Expired, upload again."), 404
    start = max(0, int(request.args.get("start", 0)))
    n = min(int(request.args.get("n", FS * 10)), FS * 30)
    return jsonify(start=start, y=[round(float(v), 3) for v in sig[start:start + n]])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
