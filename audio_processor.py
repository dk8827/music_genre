import numpy as np
import librosa
import librosa.display

DEFAULT_SR = 22050

class AudioProcessor:
    def __init__(self, n_mels, n_fft, hop_length, segment_length):
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment_length = segment_length

    def _create_mel_spectrogram(self, y_segment, sr, expected_frames):
        mel_spec = librosa.feature.melspectrogram(
            y=y_segment, sr=sr, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        min_db, max_db = mel_spec_db.min(), mel_spec_db.max()
        if max_db == min_db: # Handle silent or constant segments
            mel_spec_norm = np.zeros_like(mel_spec_db)
        else:
            mel_spec_norm = (mel_spec_db - min_db) / (max_db - min_db)

        if mel_spec_norm.shape[1] > expected_frames:
            mel_spec_norm = mel_spec_norm[:, :expected_frames]
        elif mel_spec_norm.shape[1] < expected_frames:
            pad_width = expected_frames - mel_spec_norm.shape[1]
            mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, pad_width)), mode='constant')
        return mel_spec_norm

    def extract_features(self, audio_path, duration=30, use_segments=True, sr=DEFAULT_SR):
        try:
            y, loaded_sr = librosa.load(audio_path, duration=None if use_segments else duration, sr=sr)

            if use_segments:
                segment_samples = int(self.segment_length * sr)
                expected_frames_per_segment = int((self.segment_length * sr) / self.hop_length) + 1
                segments_audio = []

                num_full_segments = len(y) // segment_samples
                for i in range(num_full_segments):
                    start_idx = i * segment_samples
                    end_idx = start_idx + segment_samples
                    segments_audio.append(y[start_idx:end_idx])

                remaining_samples = len(y) % segment_samples
                if remaining_samples > 0:
                    last_segment_audio = y[num_full_segments * segment_samples:]
                    if len(last_segment_audio) < segment_samples:
                            last_segment_audio = np.pad(last_segment_audio, (0, segment_samples - len(last_segment_audio)), mode='constant')
                    segments_audio.append(last_segment_audio)

                if not segments_audio:
                    y_padded = np.pad(y, (0, segment_samples - len(y)), mode='constant')
                    segments_audio.append(y_padded)

                segment_features = []
                for i, segment_audio_data in enumerate(segments_audio):
                    if np.random.random() < 0.2:
                        segment_audio_data = self.apply_audio_augmentation(segment_audio_data, sr)

                    mel_spec_norm = self._create_mel_spectrogram(segment_audio_data, sr, expected_frames_per_segment)
                    segment_features.append(mel_spec_norm)
                return segment_features

            else:
                if np.random.random() < 0.2:
                    y = self.apply_audio_augmentation(y, sr)

                target_length_samples = duration * sr
                if len(y) < target_length_samples:
                    y = np.pad(y, (0, target_length_samples - len(y)), mode='constant')
                else:
                    y = y[:target_length_samples]

                expected_frames = int((duration * sr) / self.hop_length) + 1
                mel_spec_norm = self._create_mel_spectrogram(y, sr, expected_frames)
                return mel_spec_norm

        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None

    def apply_audio_augmentation(self, y, sr):
        augmented = y.copy()

        if np.random.random() < 0.3:
            stretch_factor = np.random.uniform(0.8, 1.2)
            try:
                augmented = librosa.effects.time_stretch(augmented, rate=stretch_factor)
            except Exception as e:
                print(f"Warning: Time stretching failed: {e}. Skipping.")

        if np.random.random() < 0.3:
            n_steps = np.random.uniform(-2, 2)
            try:
                augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=n_steps)
            except Exception as e:
                print(f"Warning: Pitch shifting failed: {e}. Skipping.")

        if np.random.random() < 0.2:
            noise_factor = np.random.uniform(0.005, 0.02)
            noise = np.random.normal(0, noise_factor, len(augmented))
            augmented = augmented + noise

        if np.random.random() < 0.3:
            gain_factor = np.random.uniform(0.7, 1.3)
            augmented = augmented * gain_factor

        return augmented 