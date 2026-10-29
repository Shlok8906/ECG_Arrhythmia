# ECG Beat Predictor

Flask app that classifies heartbeats (N, S, V, F) from an MIT-BIH record using your trained models.

## Setup
1. Make a folder `models/` next to `app.py` and copy `best_models.joblib` into it
   (the file saved by your notebook).
2. Install packages (use the same versions as Colab, see `config.json` in your work folder):
   `pip install -r requirements.txt`
3. Run: `python app.py`, then open http://127.0.0.1:5000

## Use
- Pick the three files of one record together (e.g. `200.hea`, `200.dat`, `200.atr`).
- Choose a model (Random Forest is default) and press Predict.
- The viewer shows the filtered ECG with a coloured dot per beat. A ring means the
  prediction differs from the cardiologist annotation.

## Notes
- The .atr file gives the R-peak positions and the reference labels used for the agreement score.
- Same preprocessing and 35 features as the notebook (band-pass 0.5-40 Hz, 180-sample beat, RR + wavelet features).
- If loading the model fails, your local scikit-learn / xgboost / imbalanced-learn versions differ from Colab's.
