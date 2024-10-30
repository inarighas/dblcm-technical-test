# Voice Activity Detection (VAD) System

This project implements and evaluates a Voice Activity Detection (VAD) system from scratch. The system uses a rule-based approach combining energy and zero-crossing rate features to distinguish between speech and non-speech segments in audio files.

## Project Overview

### Objective

Implement and evaluate/test a VAD system from scratch, using either a machine learning or rule-based approach (or a combination of both).

### Approach

We chose a rule-based approach for its simplicity and interpretability. The system uses two key features:

1. Energy: to detect the presence of high-energy segments typically associated with speech.
2. Zero-Crossing Rate (ZCR): to differentiate between voiced and unvoiced sounds.

### Datasets

We used two datasets to develop and evaluate our VAD system:

1. French Conversational Corpus: For testing on continuous speech in a conversational context.
2. Speech Commands Dataset: For evaluating on short, isolated utterances and background noises.

## Implementation

The VAD system is implemented in Python and consists of the following key components:

1. Audio framing
2. Feature extraction (Energy and ZCR)
3. Feature smoothing using moving average
4. Speech detection using adaptive thresholding

## Evaluation

### Metrics

We used the following metrics to evaluate our VAD system:

- Precision: Measures the accuracy of positive predictions.
- Recall: Measures the ability to find all positive instances.
- F1 Score: The harmonic mean of precision and recall.

### Results

(Include a summary of your evaluation results here)

## Usage

To use the VAD system:

1. Install the required dependencies:

  ```bash
  sudo dnf install libsndfile-devel &&
  python -m venv .venv && poetry init vad-system && poetry install --no-root &&
  cd vad_system
  ```

2. Run the demo script: `python vad.py`
3. Run the evaluation script: `python eval.py`

## Future Improvements

- The script is not fully functional and not enough tested and evaluated (time was not enough)
- Implement more sophisticated features (e.g., spectral entropy, pitch).
- Explore machine learning approaches (e.g., GMM, SVM, or neural networks).
- Implement adaptive thresholding to handle varying noise conditions.
- Add noise reduction techniques as a preprocessing step.
