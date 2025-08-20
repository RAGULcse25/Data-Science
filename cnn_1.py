"""
Build a CNN from scratch using Keras/TensorFlow on CIFAR-10 — with robust
TensorFlow import handling, optional auto-install, and lightweight tests.

Why this rewrite?
- Your error was: ModuleNotFoundError: No module named 'tensorflow'.
- This file now **detects** missing TensorFlow and can **auto-install** it (CPU build)
  when you pass `--auto_install_tf 1` (or leave default true). If install fails
  or your environment blocks pip, it exits gracefully with clear instructions.
- Added **self-tests** (`--run_tests 1`) that avoid internet downloads by using
  synthetic data, so you can validate the pipeline even offline.

Quickstart (CPU-only):
    python keras_cnn_from_scratch_cifar10.py --epochs 10 --batch_size 128

If TensorFlow is missing, try auto-install (default true):
    python keras_cnn_from_scratch_cifar10.py --auto_install_tf 1

Manual install options (pick one):
    # CPU only
    pip install -U "tensorflow>=2.14,<3"
    # NVIDIA GPU (TF 2.16+ includes CUDA wheels; ensure compatible drivers)
    pip install -U "tensorflow[and-cuda]>=2.16,<3"

Run quick self-tests (fast, synthetic data):
    python keras_cnn_from_scratch_cifar10.py --run_tests 1

Tip: On very tight CPUs, add: --epochs 1 --batch_size 64 --quick 1
"""

import os
import sys
import subprocess
import random
import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Matplotlib is optional for plots; if unavailable we disable plotting gracefully.
try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    plt = None
    _HAVE_MPL = False

# ---------------------------------------------------------
# TensorFlow lazy import & optional auto-install
# ---------------------------------------------------------
_tf = None
_keras = None
_layers = None


def _try_import_tf() -> Tuple[object, object, object]:
    """Attempt to import TensorFlow/Keras modules if available."""
    global _tf, _keras, _layers
    if _tf is not None:
        return _tf, _keras, _layers
    try:
        import tensorflow as tf  # type: ignore
        from tensorflow import keras  # type: ignore
        from tensorflow.keras import layers  # type: ignore
        _tf, _keras, _layers = tf, keras, layers
    except ModuleNotFoundError:
        _tf = _keras = _layers = None
    return _tf, _keras, _layers


def require_tf(auto_install: bool = True, version_spec: str = "tensorflow>=2.14,<3"):
    """Ensure TensorFlow is importable; optionally attempt a pip install.

    Parameters
    ----------
    auto_install : bool
        If True and TensorFlow is missing, try to install a CPU wheel via pip.
    version_spec : str
        Pip requirement string for installation.
    """
    tf, keras, layers = _try_import_tf()
    if tf is not None:
        return tf, keras, layers

    if auto_install:
        print("TensorFlow not found. Attempting installation:", version_spec)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", version_spec])
        except Exception as e:
            print("\nAuto-install failed:", e)
            print("\nPlease install TensorFlow manually, for example:\n"
                  "  pip install -U \"tensorflow>=2.14,<3\"   # CPU only\n"
                  "  pip install -U \"tensorflow[and-cuda]>=2.16,<3\"  # NVIDIA GPU\n")
            raise SystemExit(1)
        # Retry import after install
        tf, keras, layers = _try_import_tf()
        if tf is None:
            print("TensorFlow still not importable after installation.")
            raise SystemExit(1)
        return tf, keras, layers

    # No auto-install; give a clear message and exit
    print("ERROR: TensorFlow is not installed. Install one of:\n"
          "  pip install -U \"tensorflow>=2.14,<3\"\n"
          "  pip install -U \"tensorflow[and-cuda]>=2.16,<3\"\n")
    raise SystemExit(1)


def TF() -> Tuple[object, object, object]:
    """Get (tf, keras, layers) after ensure/require has been called."""
    if _tf is None:
        print("Internal: TF() called before require_tf().")
        tf, keras, layers = _try_import_tf()
    else:
        tf, keras, layers = _tf, _keras, _layers
    if tf is None:
        raise RuntimeError("TensorFlow not available. Call require_tf() first.")
    return tf, keras, layers


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
@dataclass
class Config:
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_split: float = 0.1
    model_path: str = "cnn_cifar10.keras"
    auto_install_tf: bool = True
    dataset: str = "cifar10"  # choices: cifar10, synthetic
    quick: bool = False        # reduce filters/epochs for very slow CPUs


cfg = None  # will be set in __main__


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def _l2_reg(keras, wd: float):
    return keras.regularizers.l2(wd)


# ---------------------------------------------------------
# Model definition (from scratch, no global TF deps)
# ---------------------------------------------------------

def build_model(input_shape=(32, 32, 3), num_classes=10, weight_decay: float = 1e-4, quick: bool = False):
    tf, keras, layers = TF()

    f1, f2, f3, f4 = (16, 32, 64, 128) if quick else (32, 64, 128, 256)

    def conv_block(x, filters: int):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                          kernel_regularizer=_l2_reg(keras, weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                          kernel_regularizer=_l2_reg(keras, weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
        return x

    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)

    x = conv_block(x, f1)
    x = conv_block(x, f2)
    x = conv_block(x, f3)

    x = layers.Conv2D(f4, 3, padding="same", use_bias=False,
                      kernel_regularizer=_l2_reg(keras, weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128 if not quick else 64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="cnn_cifar10_scratch")


# ---------------------------------------------------------
# Data loading with tf.data (fallback to synthetic if offline)
# ---------------------------------------------------------

def _make_synthetic(num_train=5000, num_val=500, num_test=1000):
    # CIFAR-10 like shapes
    x_train = (np.random.rand(num_train, 32, 32, 3) * 255).astype(np.uint8)
    y_train = np.random.randint(0, 10, size=(num_train,), dtype=np.int64)
    x_val = (np.random.rand(num_val, 32, 32, 3) * 255).astype(np.uint8)
    y_val = np.random.randint(0, 10, size=(num_val,), dtype=np.int64)
    x_test = (np.random.rand(num_test, 32, 32, 3) * 255).astype(np.uint8)
    y_test = np.random.randint(0, 10, size=(num_test,), dtype=np.int64)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def make_datasets(batch_size: int, validation_split: float, dataset: str = "cifar10"):
    tf, keras, _ = TF()

    if dataset == "synthetic":
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = _make_synthetic()
    else:
        try:
            (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
            y_train = y_train.squeeze().astype(np.int64)
            y_test = y_test.squeeze().astype(np.int64)
            # Split train into train/val
            num_train = x_train.shape[0]
            val_size = int(num_train * validation_split)
            idx = np.arange(num_train)
            rng = np.random.default_rng(SEED)
            rng.shuffle(idx)
            x_val = x_train[idx[:val_size]]
            y_val = y_train[idx[:val_size]]
            x_train = x_train[idx[val_size:]]
            y_train = y_train[idx[val_size:]]
        except Exception as e:
            print("Failed to load CIFAR-10 (likely offline). Using synthetic data instead.")
            (x_train, y_train), (x_val, y_val), (x_test, y_test) = _make_synthetic()

    AUTOTUNE = tf.data.AUTOTUNE

    def make_ds(images, labels, training=False):
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        if training:
            ds = ds.shuffle(10_000, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    train_ds = make_ds(x_train, y_train, training=True)
    val_ds = make_ds(x_val, y_val, training=False)
    test_ds = make_ds(x_test, y_test, training=False)
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------
# Training utilities
# ---------------------------------------------------------

def compile_model(model, learning_rate: float):
    _, keras, _ = TF()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def get_callbacks(model_path: str):
    _, keras, _ = TF()
    return [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(filepath=model_path, monitor="val_accuracy", save_best_only=True),
        keras.callbacks.TerminateOnNaN(),
    ]


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------

def plot_history(history, out_path="training_curves.png"):
    if not _HAVE_MPL:
        print("matplotlib not available; skipping plots.")
        return
    hist = history.history
    epochs = range(1, len(hist.get("loss", [])) + 1)

    if len(hist.get("loss", [])):
        plt.figure()
        plt.plot(epochs, hist["loss"], label="Train Loss")
        if "val_loss" in hist:
            plt.plot(epochs, hist["val_loss"], label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss")
        plt.legend(); plt.tight_layout(); plt.savefig(out_path)
        print(f"Saved plot: {out_path}")

    if len(hist.get("accuracy", [])):
        plt.figure()
        plt.plot(epochs, hist["accuracy"], label="Train Acc")
        if "val_accuracy" in hist:
            plt.plot(epochs, hist["val_accuracy"], label="Val Acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training vs Validation Accuracy")
        plt.legend(); plt.tight_layout(); plt.savefig(out_path.replace(".png", "_acc.png"))
        print(f"Saved plot: {out_path.replace('.png', '_acc.png')}")


# ---------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------

def evaluate_and_show_samples(model, test_ds, class_names):
    tf, _, _ = TF()
    print("Evaluating on test set…")
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    if _HAVE_MPL:
        for batch_images, batch_labels in test_ds.take(1):
            preds = model.predict(batch_images, verbose=0)
            pred_classes = tf.argmax(preds, axis=1).numpy()
            import matplotlib.pyplot as plt  # ensure local import after Agg setup if needed
            plt.figure(figsize=(10, 10))
            for i in range(min(9, batch_images.shape[0])):
                plt.subplot(3, 3, i + 1)
                plt.imshow(batch_images[i].numpy().astype("uint8"))
                title = f"pred: {class_names[pred_classes[i]]}\ntrue: {class_names[int(batch_labels[i].numpy())]}"
                plt.title(title); plt.axis("off")
            plt.tight_layout(); plt.savefig("sample_predictions.png")
            print("Saved sample_predictions.png")
            break


# ---------------------------------------------------------
# Tests (fast, synthetic data; no internet required)
# ---------------------------------------------------------

def run_tests():
    import unittest

    class TestKerasCNN(unittest.TestCase):
        @unittest.skipIf(_try_import_tf()[0] is None, "TensorFlow not installed")
        def test_model_build_and_forward(self):
            tf, _, _ = TF()
            model = build_model(weight_decay=1e-4, quick=True)
            x = tf.random.uniform((8, 32, 32, 3))
            y = model(x, training=False)
            self.assertEqual(tuple(y.shape), (8, 10))

        @unittest.skipIf(_try_import_tf()[0] is None, "TensorFlow not installed")
        def test_single_train_step(self):
            tf, _, _ = TF()
            model = build_model(weight_decay=1e-4, quick=True)
            compile_model(model, learning_rate=1e-3)
            # tiny synthetic dataset
            x = tf.random.uniform((64, 32, 32, 3))
            y = tf.random.uniform((64,), minval=0, maxval=10, dtype=tf.int32)
            hist = model.fit(x, y, epochs=1, batch_size=16, verbose=0)
            self.assertIn("loss", hist.history)

        @unittest.skipIf(_try_import_tf()[0] is None, "TensorFlow not installed")
        def test_callbacks_init(self):
            cbs = get_callbacks("/tmp/model.keras")
            self.assertGreaterEqual(len(cbs), 3)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestKerasCNN)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


# ---------------------------------------------------------
# CLI / Main
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.learning_rate)
    parser.add_argument("--val_split", type=float, default=Config.validation_split)
    parser.add_argument("--model_path", type=str, default=Config.model_path)
    parser.add_argument("--auto_install_tf", type=int, default=1, help="1 to auto-install TF if missing, 0 to disable")
    parser.add_argument("--dataset", type=str, default=Config.dataset, choices=["cifar10", "synthetic"])
    parser.add_argument("--quick", type=int, default=0, help="1 = smaller model for slow CPUs")
    parser.add_argument("--run_tests", type=int, default=0, help="1 = run fast self-tests and exit if they fail")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        validation_split=args.val_split,
        model_path=args.model_path,
        auto_install_tf=bool(args.auto_install_tf),
        dataset=args.dataset,
        quick=bool(args.quick),
    )

    # Ensure TF is available (auto-install if requested)
    tf, keras, layers = require_tf(auto_install=cfg.auto_install_tf)
    tf.random.set_seed(SEED)  # after TF is available

    if args.run_tests:
        print("Running self-tests…")
        run_tests()
        print("Self-tests completed successfully.\n")

    print("TensorFlow version:", tf.__version__)

    # Load data
    train_ds, val_ds, test_ds = make_datasets(cfg.batch_size, cfg.validation_split, dataset=cfg.dataset)

    # CIFAR-10 class names
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    # Build & compile model
    model = build_model(input_shape=(32, 32, 3), num_classes=len(class_names),
                        weight_decay=cfg.weight_decay, quick=cfg.quick)
    model.summary()
    compile_model(model, cfg.learning_rate)

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=get_callbacks(cfg.model_path),
        verbose=2,
    )

    # Save curves
    plot_history(history)

    # Evaluate & visualize predictions
    evaluate_and_show_samples(model, test_ds, class_names)

    print(f"Best model (by val_accuracy) saved to: {cfg.model_path}")

    # Notes:
    # - If your environment blocks pip or you lack internet, run with --dataset synthetic
    #   and install TensorFlow manually in a prepared environment.
    # - For a super-fast demo: --quick 1 --epochs 1 --dataset synthetic --run_tests 1
