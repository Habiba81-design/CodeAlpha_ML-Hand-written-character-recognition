# CodeAlpha_HandwrittenCharacterRecognition

Machine Learning internship project — CodeAlpha (Task 3: Handwritten Character Recognition).

## Problem

Classify handwritten digit images (0-9) using a Convolutional Neural Network
(CNN), trained on the MNIST dataset (60,000 training images, 10,000 test
images, 28x28 grayscale).

## Project structure

```
CodeAlpha_HandwrittenCharacterRecognition/
├── 05_train_mnist_cnn.py
├── requirements.txt
├── README.md
└── outputs/
    └── mnist_model/
        ├── mnist_cnn.keras
        ├── metadata.json
        ├── training_curves.png
        ├── confusion_matrix.png
        └── sample_predictions.png
```

## Data

MNIST is built directly into TensorFlow/Keras (`keras.datasets.mnist`) and
downloads automatically the first time the script runs — no manual download
or upload required.

## Approach

1. **Preprocessing** — normalized pixel values from 0-255 integers to 0-1
   floats; reshaped images to include a channel dimension (28, 28, 1) for
   CNN input; held out 10% of the training set as a validation set (the
   official test set was kept completely untouched until final evaluation).
2. **Model** — a CNN with two Conv2D + MaxPooling blocks (32 and 64 filters)
   to learn visual patterns, followed by a Dense layer with Dropout (0.5)
   for regularization, and a 10-way softmax output layer.
3. **Training** — Adam optimizer, sparse categorical crossentropy loss,
   trained with early stopping (patience=3 on validation loss) to prevent
   overfitting.
4. **Evaluation** — accuracy, per-class precision/recall/F1, and a
   confusion matrix on the held-out test set.

## Results

**Test accuracy: 99.13%** | **Test loss: 0.0234**

| Digit | Precision | Recall | F1-score |
|---|---|---|---|
| 0 | 0.9909 | 0.9969 | 0.9939 |
| 1 | 0.9956 | 0.9921 | 0.9938 |
| 2 | 0.9903 | 0.9922 | 0.9913 |
| 3 | 0.9892 | 0.9931 | 0.9911 |
| 4 | 0.9959 | 0.9888 | 0.9923 |
| 5 | 0.9844 | 0.9910 | 0.9877 |
| 6 | 0.9947 | 0.9833 | 0.9890 |
| 7 | 0.9893 | 0.9903 | 0.9898 |
| 8 | 0.9938 | 0.9918 | 0.9928 |
| 9 | 0.9882 | 0.9931 | 0.9906 |

**Interpretation:** every digit class scores above 98% on all three metrics,
with no systematic confusion between any pair of digits. This result is in
line with published CNN benchmarks for MNIST (typically 99-99.5%),
confirming the model generalizes well rather than overfitting to the
training data.

## How to run

```bash
pip install -r requirements.txt
python3 05_train_mnist_cnn.py
```

## Notes

- The task brief also lists EMNIST (letters) as an extension option; this
  project used MNIST (digits) for a clean, reliable first build. EMNIST
  could be added as a follow-up using the same architecture.
- No GPU is required, though training is noticeably faster with one
  (a few minutes on CPU, under a minute on GPU).
