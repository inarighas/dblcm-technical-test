# dblcm:

## Assignment

This exercise is divided into the following objectives (please read everything well before starting) :

- **Obj 1:** implement and evaluate/test a VAD system from scratch : you have the choice of using a machine learning or rule-based  approach (or a combination of both). You also have to evaluate the implemented system. For this, take care of carefully selecting the metrics, dataset, etc.

- **Obj 2:** Implement a training script with the goal to fine-tune a whisper based on a specific language of your choice, you may pick the dataset that you see fit. The goal here is not to actually train a model and obtain a final model that's ready, it is only to have a functioning script without code and that is optimised to run as efficiently as possible on GPU. Implement also a script that can evaluate the model trained on datasets of your choice and with relevant metrics of your choosing. Here too, the goal is not to actually obtain evaluation results, but to have a functioning working script.

- **Obj 3:** Using Whisper or Canary 1B, implement a real-time transcription system that should be able to transcribe speech while a user is speaking with the least delay possible and with the least transcription errors possible. Properly evaluate the system in all important dimensions and in minimum one language (runtime, accuracy, etc.). The evaluation should be made using audio files.
Implement a multi-process or multi-threaded version of this (real-time) system that is able to process several (a minimum of 2) audio files at the same time in real-time each.

## Setting dev environment

- Make sure that you have installed the following requirements:
  - python 3.12
  - python-venv
  - poetry > 1.7
  - libsndfile on RPM-based distros (libsndfile1 on debian-based distros)
  - sox

### Voice Activity Detector: `vad-system`

- see [vad-system](vad-system/README.md)  
- **Note**: This exercise is not fully functional and not enough tested and evaluated.

### Finetuning Whisper

- see [whisper_finetuning](whisper_finetuning/README.md)  

### Real-time STT using Whisper

- see [real-time-transcription](real-time-transcription/README.md)

## Credits

- Ali Saghiran (<a.saghiran@gmail.com>)
