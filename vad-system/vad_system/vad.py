import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter


class VAD:
    def __init__(self, frame_duration=0.02, frame_shift=0.01):
        self.frame_duration = frame_duration
        self.frame_shift = frame_shift

    def read_wav(self, file_path):
        sample_rate, audio = wavfile.read(file_path)
        audio = audio.astype(float) / np.iinfo(audio.dtype).max
        return sample_rate, audio

    def frame_audio(self, audio, sample_rate):
        frame_length = int(self.frame_duration * sample_rate)
        frame_step = int(self.frame_shift * sample_rate)

        frames = []
        for i in range(0, len(audio) - frame_length + 1, frame_step):
            frames.append(audio[i : i + frame_length])

        return np.array(frames)

    def compute_energy(self, frames):
        return np.sum(frames**2, axis=1)

    def compute_zcr(self, frames):
        signs = np.sign(frames)
        signs[signs == 0] = -1
        return np.sum(np.abs(signs[:, 1:] - signs[:, :-1]), axis=1) / (
            2 * frames.shape[1]
        )

    def apply_moving_average(self, x, window_size=5):
        return lfilter(np.ones(window_size) / window_size, 1, x)

    def detect_speech(
        self, energy, zcr, energy_threshold=0.05, zcr_threshold=0.2
    ):

        # print(f"Energy value: {energy}, ZCR value: {zcr}")
        speech_mask = (energy > energy_threshold) & (zcr > zcr_threshold)
        return speech_mask

    def process(self, audio=None, sample_rate=None, file_path=None):
        if file_path is not None and audio is None:
            sample_rate, audio = self.read_wav(file_path)
            print(f"Loaded audio with sample rate: {sample_rate}")
        frames = self.frame_audio(audio, sample_rate)
        print(f"Frame shape: {frames.shape}")

        # compute features
        energy = self.compute_energy(frames)
        zcr = self.compute_zcr(frames)

        # Apply moving average to smooth the features
        energy_smooth = self.apply_moving_average(energy)
        zcr_smooth = self.apply_moving_average(zcr)

        # Normalize features
        energy_norm = (energy_smooth - np.min(energy_smooth)) / (
            np.max(energy_smooth) - np.min(energy_smooth)
        )
        zcr_norm = (zcr_smooth - np.min(zcr_smooth)) / (
            np.max(zcr_smooth) - np.min(zcr_smooth)
        )

        speech_mask = self.detect_speech(energy_norm, zcr_norm)
        print("len(speech_mask) =", len(speech_mask))
        print("speech_mask[:100] =", speech_mask[:100])
        return speech_mask, sample_rate

    def get_speech_segments(self, speech_mask, sample_rate):
        frame_shift_samples = int(self.frame_shift * sample_rate)
        segments = []
        start = None

        for i, is_speech in enumerate(speech_mask):
            if is_speech and start is None:
                start = i * frame_shift_samples
            elif not is_speech and start is not None:
                end = i * frame_shift_samples
                segments.append((start / sample_rate, end / sample_rate))
                start = None

        if start is not None:
            segments.append(
                (
                    start / sample_rate,
                    len(speech_mask) * frame_shift_samples / sample_rate,
                )
            )

        return segments


# Usage example
vad = VAD()
file_path = "./data/raw/custom/A0001_S001_0_G0001_G0002.wav"
speech_mask, sample_rate = vad.process(file_path=file_path)
speech_segments = vad.get_speech_segments(speech_mask, sample_rate)

print("Speech segments:")
for start, end in speech_segments:
    print(f"Start: {start:.2f}s, End: {end:.2f}s")
