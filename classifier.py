import os
import numpy as np
import joblib
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

from audio_processor import AudioProcessor, DEFAULT_SR
from data_manager import DataManager
from trainer import ModelTrainer
from plotting_utils import visualize_augmentations_standalone, visualize_segment_predictions_standalone
# Assuming _create_tf_augmentation_layers is needed by visualize_augmentations_standalone indirectly
from augmentation_layers import _create_tf_augmentation_layers 

class MusicGenreClassifier:
    """
    Main class for music genre classification.

    Orchestrates audio processing, data management, model training, and prediction.
    Handles model saving and loading, including necessary metadata.
    """
    def __init__(self, data_path="music_data", n_mels=128, n_fft=2048, hop_length=512, segment_length=4):
        """
        Initializes the classifier with audio processing parameters and data paths.

        Args:
            data_path (str): Path to the root directory containing music data.
            n_mels (int): Number of Mel bands for spectrograms.
            n_fft (int): FFT window size.
            hop_length (int): Hop length for STFT.
            segment_length (int): Length of audio segments in seconds for feature extraction.
        """
        self.data_path = data_path
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment_length = segment_length
        self.cache_file = f"processed_features_seg{segment_length}s.npz"

        self.label_encoder = LabelEncoder()
        self.genres = [] # Will be populated by DataManager
        self.model = None
        self.train_mean = None
        self.train_std = None

        # Initialize components
        self.audio_processor = AudioProcessor(n_mels, n_fft, hop_length, segment_length)
        self.data_manager = DataManager(data_path, self.cache_file, n_mels, segment_length, self.audio_processor)
        self.model_trainer = ModelTrainer(self.label_encoder) # Pass label_encoder instance

        print(f"Initialized MusicGenreClassifier:")
        print(f"  Data path: {data_path}")
        print(f"  Mel bands: {n_mels}")
        print(f"  FFT size: {n_fft}")
        print(f"  Hop length: {hop_length}")
        print(f"  Segment length: {segment_length} seconds")
        print(f"  Cache file: {self.cache_file}")

    def create_dataset(self):
        """Creates or loads the dataset using the DataManager."""
        X, y, self.genres = self.data_manager.create_dataset()
        return X, y

    def train_model(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """
        Trains the genre classification model using the ModelTrainer.

        Args:
            X (np.ndarray): Input features (spectrograms).
            y (np.ndarray): Corresponding labels (genre names).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            validation_split (float): Fraction of data to use for validation.

        Returns:
            tensorflow.keras.callbacks.History: Training history object.
        """
        history, model, train_mean, train_std = self.model_trainer.train_model(
            X, y, epochs, batch_size, validation_split
        )
        self.model = model
        self.train_mean = train_mean
        self.train_std = train_std
        # self.label_encoder is already fitted by ModelTrainer
        return history

    def _get_metadata_path(self, model_filepath):
        """Generates the metadata file path from a model file path."""
        return model_filepath.replace(".h5", "_metadata.pkl").replace(".keras", "_metadata.pkl")

    def save_model(self, filepath="music_genre_model.keras"):
        """
        Saves the trained Keras model and associated metadata.

        Metadata includes label encoder, normalization stats, and audio processing params.

        Args:
            filepath (str): Path to save the Keras model.
        """
        if self.model:
            self.model.save(filepath)
            metadata = {
                "label_encoder": self.label_encoder, # Already fitted
                "train_mean": self.train_mean,
                "train_std": self.train_std,
                "genres": self.genres,
                "n_mels": self.n_mels,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "segment_length": self.segment_length
            }
            metadata_path = self._get_metadata_path(filepath)
            joblib.dump(metadata, metadata_path)
            print(f"Model saved to {filepath}")
            print(f"Metadata saved to {metadata_path}")
        else:
            print("No model to save. Train the model first.")

    def load_model(self, filepath="music_genre_model.keras"):
        """
        Loads a Keras model and its associated metadata.

        Re-initializes internal components (AudioProcessor, DataManager, ModelTrainer)
        with the loaded parameters to ensure consistency.

        Args:
            filepath (str): Path to the Keras model file.
        """
        try:
            self.model = keras.models.load_model(filepath)
            metadata_path = self._get_metadata_path(filepath)
            metadata = joblib.load(metadata_path)
            self.label_encoder = metadata["label_encoder"]
            self.train_mean = metadata["train_mean"]
            self.train_std = metadata["train_std"]
            self.genres = metadata.get("genres", [])
            self.n_mels = metadata.get("n_mels", self.n_mels)
            self.n_fft = metadata.get("n_fft", self.n_fft)
            self.hop_length = metadata.get("hop_length", self.hop_length)
            self.segment_length = metadata.get("segment_length", self.segment_length)
            
            # Re-initialize audio_processor with loaded params if they differ
            self.audio_processor = AudioProcessor(self.n_mels, self.n_fft, self.hop_length, self.segment_length)
            # Re-initialize data_manager with loaded params (cache_file might change if segment_length changes)
            self.cache_file = f"processed_features_seg{self.segment_length}s.npz"
            self.data_manager = DataManager(self.data_path, self.cache_file, self.n_mels, self.segment_length, self.audio_processor)
            # Re-initialize model_trainer with the loaded label_encoder
            self.model_trainer = ModelTrainer(self.label_encoder)

            print(f"Model loaded from {filepath}")
            print(f"Metadata loaded from {metadata_path}")
            print(f"Loaded model with segment length: {self.segment_length}s")

        except Exception as e:
            print(f"Error loading model or metadata: {e}")
            self.model = None

    def predict_genre(self, audio_path, use_segments=True, aggregation_method='average'):
        """
        Predicts the genre of an audio file.

        Can perform prediction on the full audio or on segments with aggregation.

        Args:
            audio_path (str): Path to the audio file.
            use_segments (bool): If True, uses segment-based prediction. Otherwise, uses full audio.
            aggregation_method (str): Method to aggregate segment predictions ('average', 'majority_vote', etc.).
                                       Used only if use_segments is True.

        Returns:
            tuple or None: (genre_name, confidence, prediction_probabilities) or None if error.
                           If use_segments is True, also returns segment_predictions in the tuple.
        """
        if self.model is None or self.train_mean is None or self.train_std is None:
            print("Model or normalization stats not loaded. Train or load a model first.")
            return None

        if use_segments:
            return self.predict_genre_with_segments(audio_path, aggregation_method)
        else:
            features = self.audio_processor.extract_features(audio_path, use_segments=False, sr=DEFAULT_SR)
            if features is None: return None

            features_normalized = (features - self.train_mean) / (self.train_std + 1e-6) # Added epsilon for std
            features_reshaped = features_normalized.reshape(1, *features_normalized.shape, 1)

            prediction = self.model.predict(features_reshaped)
            predicted_class_idx = np.argmax(prediction)
            confidence = prediction[0][predicted_class_idx]
            genre = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            return genre, confidence, prediction[0]

    def predict_genre_with_segments(self, audio_path, aggregation_method='average'):
        """
        Predicts genre by processing audio in segments and aggregating predictions.

        Args:
            audio_path (str): Path to the audio file.
            aggregation_method (str): Method to combine segment predictions.
                Supported: 'average', 'majority_vote', 'max_confidence', 'weighted_average'.

        Returns:
            tuple or None: (genre_name, confidence, final_prediction_probabilities, segment_raw_predictions) or None.
        """
        print(f"Predicting genre for: {os.path.basename(audio_path)} using {self.segment_length}s segments, {aggregation_method} aggregation.")

        segments_features_list = self.audio_processor.extract_features(audio_path, use_segments=True, sr=DEFAULT_SR)
        if not segments_features_list:
            print("Could not extract features.")
            return None

        segments_array = np.array(segments_features_list)
        segments_normalized = (segments_array - self.train_mean) / (self.train_std + 1e-6) # Added epsilon for std
        segments_reshaped = segments_normalized.reshape(*segments_normalized.shape, 1)

        print(f"Predicting on {len(segments_features_list)} segments...")
        segment_predictions = self.model.predict(segments_reshaped, verbose=0)

        if aggregation_method == 'average':
            final_prediction_probs = np.mean(segment_predictions, axis=0)
        elif aggregation_method == 'majority_vote':
            segment_classes = np.argmax(segment_predictions, axis=1)
            most_common_class = np.bincount(segment_classes).argmax()
            final_prediction_probs = np.zeros(segment_predictions.shape[1])
            final_prediction_probs[most_common_class] = 1.0
        elif aggregation_method == 'max_confidence':
            max_conf_idx = np.argmax(np.max(segment_predictions, axis=1))
            final_prediction_probs = segment_predictions[max_conf_idx]
        elif aggregation_method == 'weighted_average':
            confidences = np.max(segment_predictions, axis=1)
            weights = confidences / (np.sum(confidences) + 1e-6)
            final_prediction_probs = np.average(segment_predictions, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        predicted_class_idx = np.argmax(final_prediction_probs)
        confidence = final_prediction_probs[predicted_class_idx]
        genre = self.label_encoder.inverse_transform([predicted_class_idx])[0]

        print(f"  Predicted genre: {genre}, Confidence: {confidence:.4f}")
        return genre, confidence, final_prediction_probs, segment_predictions

    def compare_aggregation_methods(self, audio_path):
        """Compares different segment aggregation methods for a given audio file."""
        methods = ['average', 'majority_vote', 'max_confidence', 'weighted_average']
        print(f"Comparing aggregation methods for: {os.path.basename(audio_path)}\n" + "-" * 60)
        results = {}
        for method in methods:
            print(f"Testing {method} aggregation:")
            result_tuple = self.predict_genre_with_segments(audio_path, aggregation_method=method)
            if result_tuple:
                genre, confidence, _, _ = result_tuple
                results[method] = {'genre': genre, 'confidence': confidence}
                print(f"  Result: {genre} (confidence: {confidence:.4f})")

        print(f"\nSUMMARY FOR {os.path.basename(audio_path)}\n" + "-" * 40)
        for method, res in results.items():
            print(f"{method:18}: {res['genre']:10} ({res['confidence']:.4f})")
        return results

    def clear_cache(self):
        """Clears the cached processed features file."""
        self.data_manager.clear_cache()

    def visualize_augmentations(self, audio_path):
        """Visualizes various audio augmentations on a sample audio file."""
        visualize_augmentations_standalone(
            audio_path,
            self.audio_processor._create_mel_spectrogram, # Pass the method from audio_processor
            _create_tf_augmentation_layers, # This is the function from augmentation_layers
            self.n_mels, self.n_fft, self.hop_length, sr=DEFAULT_SR
        )

    def visualize_segment_predictions(self, audio_path, max_segments_to_show=10):
        """Visualizes segment-level predictions and their aggregation for an audio file."""
        print(f"Visualizing segment predictions for: {os.path.basename(audio_path)}")
        result_tuple = self.predict_genre_with_segments(audio_path, aggregation_method='average')
        if not result_tuple: return

        genre, confidence, final_pred_probs, segment_predictions = result_tuple
        visualize_segment_predictions_standalone(
            segment_predictions, final_pred_probs, self.label_encoder.classes_, genre, max_segments_to_show
        ) 