"""Export the trained MLP to static files the browser can load.

Writes model.json (layer sizes) and model.bin (float32 weights and biases,
little-endian, in order W1, b1, W2, b2, ...; each W is row-major with shape
(inputs, outputs)). The output layer is reordered so column d is digit d.

Run this after (re)training; the site itself needs no Python.
"""

import json
import os

import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "mnist_mlp.joblib")


def load_or_train_model():
    """Load the cached model, or download MNIST and train a new one."""
    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        return joblib.load(MODEL_PATH)

    print("Downloading MNIST (first run only, this may take a minute)...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False,
                        parser="liac-arff")
    X = X / 255.0  # Scale pixel values to [0, 1]

    # Standard split: first 60,000 for training, last 10,000 for testing
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    print("Training neural network...")
    model = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=20,
                          early_stopping=True, random_state=42, verbose=True)
    model.fit(X_train, y_train)
    print(f"Test accuracy: {model.score(X_test, y_test):.4f}")

    joblib.dump(model, MODEL_PATH)
    return model


def main():
    model = load_or_train_model()
    # The browser code implements exactly this network shape
    assert model.activation == "relu" and model.out_activation_ == "softmax"

    weights = [w.copy() for w in model.coefs_]
    biases = [b.copy() for b in model.intercepts_]
    # Model classes are strings '0'..'9'; reorder output columns by digit
    order = [list(model.classes_).index(str(d)) for d in range(10)]
    weights[-1] = weights[-1][:, order]
    biases[-1] = biases[-1][order]

    with open(os.path.join(HERE, "model.bin"), "wb") as f:
        for w, b in zip(weights, biases):
            f.write(w.astype("<f4").tobytes())
            f.write(b.astype("<f4").tobytes())

    sizes = [weights[0].shape[0]] + [w.shape[1] for w in weights]
    with open(os.path.join(HERE, "model.json"), "w") as f:
        json.dump({"sizes": sizes}, f)
    print(f"Wrote model.bin and model.json (layer sizes {sizes})")


if __name__ == "__main__":
    main()
