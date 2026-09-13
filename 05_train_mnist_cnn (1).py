"""
05_train_mnist_cnn.py
Task 3: Handwritten Character Recognition (digits) using a CNN on MNIST.

MNIST is built directly into TensorFlow -- no download or upload needed,
it fetches automatically the first time this runs (requires internet,
which Colab has by default).

Steps:
  1. Load MNIST (60,000 train images, 10,000 test images, 28x28 grayscale)
  2. Preprocess (normalize pixel values, reshape for CNN input, split off
     a validation set from the training data)
  3. Build a CNN (Conv2D -> Pool -> Conv2D -> Pool -> Dense -> Dropout -> output)
  4. Train with early stopping
  5. Evaluate on the held-out test set (accuracy, classification report,
     confusion matrix)
  6. Save the trained model and training curves
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
out_dir = "outputs/mnist_model"
os.makedirs(out_dir, exist_ok=True)

random_state = 42
val_fraction = 0.1     # fraction of the 60,000 training images held out for validation
batch_size = 128
epochs = 20             # early stopping will likely stop before this
num_classes = 10

tf.random.set_seed(random_state)
np.random.seed(random_state)


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_data():
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
    print(f"Full training set: {x_train_full.shape}, Test set: {x_test.shape}")
    return x_train_full, y_train_full, x_test, y_test


# ---------------------------------------------------------------------------
# 2. Preprocess
# ---------------------------------------------------------------------------
def preprocess(x_train_full, y_train_full, x_test, y_test):
    # Pixel values are 0-255 integers; scale to 0-1 floats so the network
    # trains faster and more stably.
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # CNNs expect a channel dimension: (height, width, channels).
    # MNIST is grayscale, so channels=1.
    x_train_full = np.expand_dims(x_train_full, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Hold out a validation set from the training data, so we never tune
    # or make decisions based on the real test set until the very end.
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full,
        test_size=val_fraction, random_state=random_state, stratify=y_train_full,
    )

    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    return x_train, y_train, x_val, y_val, x_test, y_test


# ---------------------------------------------------------------------------
# 3. Build the CNN
# ---------------------------------------------------------------------------
def build_model():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),

        layers.Conv2D(64, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  # labels are plain integers (0-9), not one-hot
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
def train_model(model, x_train, y_train, x_val, y_val):
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True,
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stop],
        verbose=1,
    )
    return history


# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------
def evaluate_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss:     {test_loss:.4f}")

    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(y_test, y_pred, digits=4)
    print("\nClassification report:\n", report)

    cm = confusion_matrix(y_test, y_pred)
    return test_acc, test_loss, report, cm, y_pred


def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss over epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=120)
    plt.close(fig)


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)

    # Annotate each cell with its count
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=120)
    plt.close(fig)


def plot_sample_predictions(x_test, y_test, y_pred, n=12):
    """Show a grid of test images with their true vs predicted labels --
    useful to visually sanity-check the model and for your video."""
    fig, axes = plt.subplots(2, 6, figsize=(14, 5))
    axes = axes.flatten()
    idxs = np.random.choice(len(x_test), n, replace=False)

    for ax, idx in zip(axes, idxs):
        ax.imshow(x_test[idx].squeeze(), cmap="gray")
        correct = y_test[idx] == y_pred[idx]
        color = "green" if correct else "red"
        ax.set_title(f"True: {y_test[idx]} / Pred: {y_pred[idx]}", color=color, fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sample_predictions.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    x_train_full, y_train_full, x_test, y_test = load_data()
    x_train, y_train, x_val, y_val, x_test, y_test = preprocess(
        x_train_full, y_train_full, x_test, y_test
    )

    model = build_model()
    model.summary()

    history = train_model(model, x_train, y_train, x_val, y_val)

    test_acc, test_loss, report, cm, y_pred = evaluate_model(model, x_test, y_test)

    plot_training_curves(history)
    plot_confusion_matrix(cm)
    plot_sample_predictions(x_test, y_test, y_pred)

    model_path = os.path.join(out_dir, "mnist_cnn.keras")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    metadata = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "epochs_trained": len(history.history["loss"]),
        "classification_report": report,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {os.path.join(out_dir, 'metadata.json')}")


if __name__ == "__main__":
    main()
