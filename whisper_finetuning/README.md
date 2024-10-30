# Whisper Fine-tuning for German Speech Recognition

This project fine-tunes the Whisper small model for German speech recognition using the Common Voice dataset.

## Requirements

- Check if python, python-venv and poetry are installed in your system
- Install the required packages using:

```bash
python -m venv .venv && source .venv/bin/activate && poetry install --no-root
```

## Project Structure

```plain
whisper-german-finetuning/
│
├── train.py                 # Main script for data processing and model training
├── requirements.txt         # List of Python dependencies
├── README.md                # This file
└── whisper-small-de/        # Directory where the fine-tuned model will be saved
└── ...
```

- `train.py`: Script for fine-tuning the Whisper model on German speech data.
- `whisper-german-finetuned/`: Directory containing the fine-tuned model and processor.

## Overview

The script performs the following main steps:

1. Load and preprocess the Common Voice German dataset
2. Set up the Whisper model, tokenizer, and feature extractor
3. Prepare the dataset for training
4. Define a custom data collator
5. Set up evaluation metrics
6. Configure and run the tra

## Technical Choices

1. **Dataset**: We use the Common Voice 7.0 German dataset. This is a high-quality, open-source dataset suitable for speech recognition tasks.

2. **Model**: We use the "whisper-small" model as a starting point. This balances performance and computational requirements, making it suitable for fine-tuning on modest hardware.

3. **Audio Preprocessing**:
   - We resample the audio to 16kHz to match Whisper's requirements.
   - We use the WhisperFeatureExtractor to compute log-Mel input features, which are the expected input for the Whisper model.

4. **Tokenizer**: We use the WhisperTokenizer to tokenize the target text. This ensures compatibility with the Whisper model architecture.

5. **Data Collator**: We implement a custom DataCollatorSpeechSeq2SeqWithPadding. This handles padding of both input features and labels, which is necessary for batch processing.

6. **Evaluation Metric**: We use Word Error Rate (WER) as our evaluation metric, which is standard for speech recognition tasks.

7. **Training Arguments**:
   - We use a learning rate of 1e-5, which is typically suitable for fine-tuning pre-trained models.
   - We enable gradient checkpointing and FP16 training for memory efficiency and speed.
   - We use a warmup period of 500 steps and train for a total of 5000 steps.

## GPU Optimization

The script already includes some GPU optimizations, but here are additional suggestions:

1. **Batch Size**: Increase `per_device_train_batch_size` and `per_device_eval_batch_size` to the maximum that your GPU memory allows. This will improve GPU utilization.

2. **Gradient Accumulation**: If you can't increase batch size further, increase `gradient_accumulation_steps` to simulate larger batch sizes.

3. **Mixed Precision Training**: The script already uses FP16, which is good for GPU performance. Ensure your GPU supports it.

4. **DataLoader Workers**: Increase `num_proc` in the dataset mapping step to use more CPU cores for data preprocessing, potentially speeding up data loading.

5. **Caching**: Consider adding caching to the dataset preprocessing step to speed up subsequent runs.

6. **Model Parallelism**: For very large models or multi-GPU setups, consider using techniques like model parallelism or distributed training.

7. **GPU-specific Optimizations**: If using NVIDIA GPUs, consider using NVIDIA's Apex library for further optimizations.

## Running the Script

1. Ensure all required libraries are installed.
2. Set the `N_MAX_DATA` variable to control the amount of data used.
3. Run the script `python train.py`. The model will be saved in the "./whisper-small-de" directory.

Note: This script requires access to the Common Voice dataset, which may require authentication.

This will start the training process. The model checkpoints and training logs will be saved in the "./whisper-small-de" directory.

Note: This script requires access to the Common Voice dataset, which may require authentication.

## Monitoring Training

The script is configured to use TensorBoard for monitoring training progress. To view the training metrics:

1. Ensure TensorBoard is installed:
2. Start TensorBoard: `tensorboard --logdir=./whisper-small-de`
3. Open a web browser and go to `http://localhost:6006` to view the TensorBoard dashboard.

You can monitor the following metrics:

- Training loss
- Evaluation loss
- Word Error Rate (WER) on the evaluation set
- Learning rate

The script is also configured to log training progress every 25 steps and evaluate the model every 1000 steps. You can adjust these frequencies in the `Seq2SeqTrainingArguments` in the script.

## Customization

Various aspects of the training process cab modified through the `Seq2SeqTrainingArguments` in the script. Some key parameters to adjust are:

- `per_device_train_batch_size` and `per_device_eval_batch_size`: Adjust based on your GPU memory. More GPU memory allows to use bigger batches.
- `learning_rate`: Change if you want to experiment with different learning rates. High rates makes fitting faster but can yield overfitted models, lower rates make training slower and more vulnerable to local minima.
- `max_steps`: Increase or decrease the total number of training steps.
- `warmup_steps`: Adjust the number of warmup steps for the learning rate scheduler.

these parameters need to be adjusted carefully, as they can significantly impact the training process and the final model performance.

## Note

The scripts are set up to use a limited subset of the dataset for faster execution.
For full training, remove or adjust the dataset size limits.
