"""Handwritten digit recognizer.

Draw a digit (0-9) with the mouse and the app predicts it using a neural
network (MLP) trained on the MNIST dataset. The trained model is cached to
disk so training only happens on the first run.
"""

import os
import tkinter as tk

import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_mlp.joblib")
CANVAS_SIZE = 280  # Drawing area in pixels (10x the MNIST image size)
BRUSH_RADIUS = 10


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


def preprocess(image):
    """Convert a drawn image into a 784-length vector in MNIST format.

    MNIST digits fit in a 20x20 box centered by center of mass inside a
    28x28 image, so the drawing is normalized the same way.
    """
    bbox = image.getbbox()
    if bbox is None:
        return None  # Nothing drawn

    digit = image.crop(bbox)

    # Scale the longer side to 20 pixels, keeping the aspect ratio
    w, h = digit.size
    scale = 20.0 / max(w, h)
    digit = digit.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)

    # Paste into the middle of a 28x28 black image
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(digit, ((28 - digit.size[0]) // 2, (28 - digit.size[1]) // 2))

    # Shift so the center of mass sits at the image center
    arr = np.asarray(canvas, dtype=np.float32)
    ys, xs = np.nonzero(arr)
    weights = arr[ys, xs]
    cy = np.average(ys, weights=weights)
    cx = np.average(xs, weights=weights)
    canvas = canvas.transform((28, 28), Image.AFFINE,
                              (1, 0, cx - 14, 0, 1, cy - 14), fill=0)

    return (np.asarray(canvas, dtype=np.float32) / 255.0).reshape(1, -1)


class DigitApp:
    def __init__(self, root, model):
        self.model = model
        root.title("Handwritten Digit Recognizer")
        root.resizable(False, False)

        # Left side: drawing canvas and buttons
        left = tk.Frame(root, padx=10, pady=10)
        left.pack(side=tk.LEFT)

        tk.Label(left, text="Draw a digit (0-9)", font=("Segoe UI", 12)).pack()
        self.canvas = tk.Canvas(left, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="black", cursor="cross")
        self.canvas.pack(pady=5)

        buttons = tk.Frame(left)
        buttons.pack(fill=tk.X)
        tk.Button(buttons, text="Predict", width=12, command=self.predict).pack(side=tk.LEFT, expand=True)
        tk.Button(buttons, text="Clear", width=12, command=self.clear).pack(side=tk.LEFT, expand=True)

        # Right side: prediction result and probability bars
        right = tk.Frame(root, padx=10, pady=10)
        right.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(right, text="Prediction", font=("Segoe UI", 12)).pack()
        self.result = tk.Label(right, text="?", font=("Segoe UI", 64, "bold"), fg="#2a6fdb")
        self.result.pack()
        self.confidence = tk.Label(right, text="", font=("Segoe UI", 10))
        self.confidence.pack()

        self.bars = tk.Canvas(right, width=200, height=220)
        self.bars.pack(pady=5)

        # Offscreen image mirrors the canvas so the strokes can be processed
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.last = None

        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.predict())
        self.draw_bars(np.zeros(10))

    def on_press(self, event):
        self.last = (event.x, event.y)
        self.paint_dot(event.x, event.y)

    def on_drag(self, event):
        x0, y0 = self.last
        x1, y1 = event.x, event.y
        width = BRUSH_RADIUS * 2
        self.canvas.create_line(x0, y0, x1, y1, fill="white", width=width, capstyle=tk.ROUND)
        self.draw.line((x0, y0, x1, y1), fill=255, width=width)
        self.paint_dot(x1, y1)  # Round joint between line segments
        self.last = (x1, y1)

    def paint_dot(self, x, y):
        r = BRUSH_RADIUS
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse((x - r, y - r, x + r, y + r), fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, CANVAS_SIZE, CANVAS_SIZE), fill=0)
        self.result.config(text="?")
        self.confidence.config(text="")
        self.draw_bars(np.zeros(10))

    def predict(self):
        x = preprocess(self.image)
        if x is None:
            return
        probs = self.model.predict_proba(x)[0]
        # Model classes are strings '0'..'9'; reorder probabilities by digit
        order = [list(self.model.classes_).index(str(d)) for d in range(10)]
        probs = probs[order]

        best = int(np.argmax(probs))
        self.result.config(text=str(best))
        self.confidence.config(text=f"Confidence: {probs[best] * 100:.1f}%")
        self.draw_bars(probs)

    def draw_bars(self, probs):
        """Draw a horizontal bar chart of the probability for each digit."""
        self.bars.delete("all")
        best = int(np.argmax(probs)) if probs.any() else -1
        for d, p in enumerate(probs):
            y = d * 22 + 2
            self.bars.create_text(10, y + 9, text=str(d), font=("Segoe UI", 10))
            self.bars.create_rectangle(25, y + 2, 175, y + 16, outline="#ccc")
            color = "#2a6fdb" if d == best else "#9bb7e8"
            if p > 0:
                self.bars.create_rectangle(25, y + 2, 25 + 150 * p, y + 16, fill=color, outline="")
            self.bars.create_text(180, y + 9, text=f"{p * 100:.0f}%", anchor=tk.W, font=("Segoe UI", 8))


def main():
    model = load_or_train_model()
    root = tk.Tk()
    DigitApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
