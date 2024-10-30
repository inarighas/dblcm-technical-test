# Real time Transcription using Whisper

Real-time transcription using the Whisper model.

## Requirements

- Check if python, python-venv and poetry are installed in your system
- Install the required packages using:

```bash
sudo dnf install portaudio-devel alsa-lib-devel libsoundfile-devel
python -m venv .venv && source .venv/bin/activate && poetry install --no-root
```

## Run in the command line

```shell
python server.py
python client.py
```

## Output example

- Here is an example of a real time transcription of the firs seconds of President Macron Speech in 2020 (source ).

```plain
>> python client.py 
ALSA lib pcm_dsnoop.c:567:(snd_pcm_dsnoop_open) unable to open slave
ALSA lib pcm_dmix.c:1000:(snd_pcm_dmix_open) unable to open slave
ALSA lib pcm.c:2722:(snd_pcm_open_noupdate) Unknown PCM cards.pcm.rear
ALSA lib pcm.c:2722:(snd_pcm_open_noupdate) Unknown PCM cards.pcm.center_lfe
ALSA lib pcm.c:2722:(snd_pcm_open_noupdate) Unknown PCM cards.pcm.side
ALSA lib pcm_dmix.c:1000:(snd_pcm_dmix_open) unable to open slave
BDB1539 Build signature doesn't match environment
Cannot open DB environment: BDB0091 DB_VERSION_MISMATCH: Database environment version mismatch
Cannot lock down 107341340 byte memory area (Cannot allocate memory)
--------------------------------------------------------------------------------
Microphone initialized, recording started...
--------------------------------------------------------------------------------
TRANSCRIPTION
--------------------------------------------------------------------------------
|fr| François, François.                                                        
|fr| mes chers compatriotes, depuis quelques semaines notre pays fait face à la propagation d'un vieil.
|fr| le Covid-19 qui a touché plusieurs milliers de nos compatriotes.           
|fr| J'ai bien entendu ce soir, avant tout chose une pensée ému et chaleureuse. 
|fr| famille et les proches de nos victimes, cette épidémie qui effecte.        
|fr| Tous les continents. Et frappe tous les pays européens et la plus grave crise sanitaire.
|fr| qu'est connu la France depuis un siècle ?                                  
^CExiting...                                                                    
Number of processed chunks:  44
Overall time: avg: 0.7511s, std: 0.4490s
Transcription time: avg: 0.7511s, std: 0.4490s
Postprocessing time: avg: 0.0000s, std: 0.0000s
The average latency is 1.7511s
```

After a keyboard interrupt, the transcriber will exit and print some overall statistics about performance and transcription delay.

## Improvements

- Transcribe shorter timeframes to get more instant transcriptions.
- Include more parallelism (parallel processing) in the transcriber.

## Troubleshooting

- If you encounter jack audio error, :
  - check if all necessary dependencies are installed and execute `jack_control exit` then `jack_control start`.
  - You can also check pulseaudio using `pavucontrol` to check if the python app is correctly linked to a PC input or output device.
- This code has been tested on Fedora 40.
