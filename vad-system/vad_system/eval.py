import numpy as np
from datasets import load_dataset
from scipy.signal import lfilter
from sklearn.metrics import precision_recall_fscore_support
from vad import VAD


def evaluate_vad(vad, test_files, ground_truth):
    total_tp, total_fp, total_fn = 0, 0, 0

    for file_path, truth in zip(test_files, ground_truth):
        speech_mask, sample_rate = vad.process(file_path)
        predicted_segments = vad.get_speech_segments(speech_mask, sample_rate)

        for pred_start, pred_end in predicted_segments:
            overlap = any(
                (truth_start <= pred_start < truth_end)
                or (truth_start < pred_end <= truth_end)
                or (pred_start <= truth_start and pred_end >= truth_end)
                for truth_start, truth_end in truth
            )

            if overlap:
                total_tp += 1
            else:
                total_fp += 1

        total_fn += sum(
            1
            for truth_start, truth_end in truth
            if all(
                pred_end <= truth_start or pred_start >= truth_end
                for pred_start, pred_end in predicted_segments
            )
        )

    precision = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    )
    recall = (
        total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {"Precision": precision, "Recall": recall, "F1 Score": f1_score}


def evaluate_vad_on_dataset(vad, dataset, num_samples=1000):
    y_true = []
    y_pred = []

    for i, example in enumerate(dataset):
        if i >= num_samples:
            break

        audio = example["audio"]["array"]
        sample_rate = example["audio"]["sampling_rate"]
        label = example["label"]

        # Process audio with VAD
        speech_mask = vad.process(audio=audio, sample_rate=sample_rate)

        # Consider the audio as speech if any frame is detected as speech
        is_speech_pred = np.any(speech_mask)

        # Consider the audio as speech if it's not "silence" or "background_noise"
        is_speech_true = label not in ["silence", "background_noise"]

        y_true.append(is_speech_true)
        y_pred.append(is_speech_pred)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )

    return {"Precision": precision, "Recall": recall, "F1 Score": f1}


# Load the Speech Commands dataset
dataset = load_dataset("speech_commands", "v0.02")
train_dataset = dataset["train"]

# Create VAD instance
vad = VAD()

# Evaluate VAD on the dataset
results = evaluate_vad_on_dataset(vad, train_dataset)
print("Evaluation results:", results)
