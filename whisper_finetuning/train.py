#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import torch
from datasets import Audio, DatasetDict, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

N_MAX_TRAIN_DATA = 80  # train on the first 800 samples, for faster training
N_MAX_TEST_DATA = 10  # test on the first 80 samples, for faster evaluation
SAMPLING_RATE = 16_000  # audio sampling rate in Hz (whisper requires 16kHz)
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-5
WARMUP_STEPS = 20  # 500 would be more suitabl
MAX_STEPS = 50  # 5000 would be more suitabl
LOGGING_STEPS = 25
SAVE_STEPS = 10
EVAL_STEPS = 10


# utils
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths
        # and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]}
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [
            {"input_ids": feature["labels"]} for feature in features
        ]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# Loading data from HF

common_voice = DatasetDict()

common_voice["train"] = load_dataset(
    "mozilla-foundation/common_voice_7_0",
    "de",
    split="train+validation",
    token=True,
)
common_voice["test"] = load_dataset(
    "mozilla-foundation/common_voice_7_0", "de", split="test", token=True
)

print("Dataset content: ", common_voice)


common_voice = common_voice.remove_columns(
    [
        "accent",
        "age",
        "client_id",
        "down_votes",
        "gender",
        "locale",
        "path",
        "segment",
        "up_votes",
    ]
)

# Getting model from hugging face
feature_extractor = WhisperFeatureExtractor.from_pretrained(
    "openai/whisper-small"
)
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Check that tokenizer and model match for the 5 first sentences of
# the train dataset
for i in range(5):
    input_str = common_voice["train"][i]["sentence"]
    labels = tokenizer(input_str).input_ids
    decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
    decoded_str = tokenizer.decode(labels, skip_special_tokens=True)
    assert decoded_str == input_str, (
        f"Tokenizer decoding output must match input."
        f"Got decoded='{decoded_str}' vs input='{input_str}'."
    )

processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# Whisper accepts only 16kHz
common_voice = common_voice.cast_column(
    "audio", Audio(sampling_rate=SAMPLING_RATE)
)


common_voice_reduced = DatasetDict(
    {
        "train": common_voice["train"].select(
            range(min(len(common_voice["train"]), N_MAX_TRAIN_DATA))
        ),
        "test": common_voice["test"].select(
            range(min(len(common_voice["test"]), N_MAX_TEST_DATA))
        ),
    }
)

del common_voice

# Data perepration

common_voice = common_voice_reduced.map(
    prepare_dataset,
    remove_columns=common_voice_reduced.column_names["train"],
)


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)


# Evaluation metrics (WER)
metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# configure trainer
# Note:
#   increase `gradient_accumulation_steps` by 2x
#   for every 2x decrease in batch size

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-de",  # repo name
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,
    # gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    logging_steps=LOGGING_STEPS,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)


# Run model training

trainer.train()
