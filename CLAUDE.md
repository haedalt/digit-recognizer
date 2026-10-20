# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A single-file Tkinter desktop app (`digit_recognizer.py`) where the user draws a digit with the mouse and a scikit-learn `MLPClassifier` trained on MNIST predicts it, showing per-digit probability bars.

## Commands

- Run with console output (training progress, errors visible): `python digit_recognizer.py`
- Run without a console window: `Run Digit Recognizer.bat` (uses `pythonw`, so tracebacks are silently lost; use the command above when debugging)
- Dependencies (no requirements file): `pip install numpy pillow scikit-learn joblib`. Tkinter ships with the standard Python installer.
- Force retraining: delete `mnist_mlp.joblib`. The next run downloads MNIST via `fetch_openml` (network required) and trains (several minutes).

There are no tests, linter config, or build step.

## Architecture

- **Model caching:** `load_or_train_model()` loads `mnist_mlp.joblib` from the script's directory if present; otherwise it downloads `mnist_784`, trains on the standard 60k/10k split, and dumps the model. The cached file is a pickled sklearn object, so it can break when the scikit-learn version changes; retrain if loading fails.
- **Dual drawing surfaces:** every stroke is drawn twice — on the visible `tk.Canvas` and on an offscreen PIL `Image` (`self.image`, 280x280, mode `L`). Prediction reads only the PIL image, so any new drawing or clearing logic must update both.
- **Preprocessing must match MNIST:** `preprocess()` crops to the bounding box, scales the long side to 20px, pads into 28x28, then shifts the center of mass to the image center, and scales pixels to [0, 1]. White-on-black is required (MNIST convention). Changes here directly affect accuracy.
- **Class labels are strings:** `fetch_openml` yields labels `'0'..'9'`, so `predict()` reorders `predict_proba` output via `model.classes_`. Keep this mapping if the training data source changes.
- Prediction runs automatically on mouse release as well as via the Predict button.

## Notes

- This folder has its own git repository (branch `main`), nested inside an unrelated repo rooted at the user's home directory. Run git commands from here.
- The project is being split into `desktop_version/` and `web_version/`, each with its own CLAUDE.md. The desktop app files above still live in this root folder for now.
- `web_version/` is a static site (inference runs in the browser) deployed to GitHub Pages by `.github/workflows/pages.yml` on push to `main`.
- `*.joblib` is gitignored, so a fresh clone of the desktop app retrains on first run.
