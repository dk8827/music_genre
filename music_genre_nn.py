import os
import warnings
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

DEFAULT_SR = 22050

class PlotCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.train_acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

        if (epoch + 1) % 5 == 0:
            self.plot_metrics()

    def plot_metrics(self):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_acc, label='Training Accuracy', color='blue')
        plt.plot(self.val_acc, label='Validation Accuracy', color='red')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.train_loss, label='Training Loss', color='blue')
        plt.plot(self.val_loss, label='Validation Loss', color='red')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


class MusicGenreClassifier:
    def __init__(self, data_path="music_data", n_mels=128, n_fft=2048, hop_length=512, segment_length=4):
        self.data_path = data_path
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment_length = segment_length
        self.model = None
        self.label_encoder = LabelEncoder()
        self.genres = []
        self.cache_file = f"processed_features_seg{segment_length}s.npz"
        self.train_mean = None
        self.train_std = None

        print(f"Initialized MusicGenreClassifier:")
        print(f"  Data path: {data_path}")
        print(f"  Mel bands: {n_mels}")
        print(f"  FFT size: {n_fft}")
        print(f"  Hop length: {hop_length}")
        print(f"  Segment length: {segment_length} seconds")
        print(f"  Cache file: {self.cache_file}")

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

    def create_dataset(self):
        print("Creating dataset from audio files...")
        if os.path.exists(self.cache_file):
            print("Loading cached features...")
            try:
                cached_data = np.load(self.cache_file, allow_pickle=True)
                X = cached_data['features']
                y = cached_data['labels']
                self.genres = cached_data['genres'].tolist() 
                print("Successfully loaded cached features")
                print(f"Dataset shape: {X.shape}")
                unique_genres, counts = np.unique(y, return_counts=True)
                for genre, count in zip(unique_genres, counts):
                    print(f"  {genre}: {count} samples")
                return X, y
            except Exception as e:
                print(f"Error loading cached data: {str(e)}. Processing files from scratch...")
        
        X, y = self.load_audio_dataset()
        
        if X is not None and y is not None and len(X) > 0:
            print("Saving features to cache...")
            try:
                np.savez(self.cache_file, features=X, labels=y, genres=np.array(self.genres))
                print("Successfully cached features")
            except Exception as e:
                print(f"Error caching features: {str(e)}")
        else:
            print("No data to cache.")
        return X, y

    def _create_tf_augmentation_layers(self):
        @tf.function
        def add_noise_tf(spectrogram, noise_factor=0.005):
            noise = tf.random.normal(tf.shape(spectrogram), stddev=noise_factor)
            return spectrogram + noise
        
        @tf.function 
        def spec_augment_tf(spectrogram, freq_mask_param=15, time_mask_param=35, num_freq_masks=1, num_time_masks=1):
            spec_shape = tf.shape(spectrogram)
            num_mel_bins = spec_shape[1] 
            num_time_steps = spec_shape[2]
            
            for _ in range(num_freq_masks):
                f = tf.random.uniform([], 0, freq_mask_param, dtype=tf.int32)
                f = tf.minimum(f, num_mel_bins)
                f0 = tf.random.uniform([], 0, num_mel_bins - f, dtype=tf.int32)
                mask = tf.concat([tf.ones((spec_shape[0], f0, num_time_steps, spec_shape[3])),
                                  tf.zeros((spec_shape[0], f, num_time_steps, spec_shape[3])),
                                  tf.ones((spec_shape[0], num_mel_bins - f0 - f, num_time_steps, spec_shape[3]))], axis=1)
                spectrogram = spectrogram * mask

            for _ in range(num_time_masks):
                t = tf.random.uniform([], 0, time_mask_param, dtype=tf.int32)
                t = tf.minimum(t, num_time_steps)
                t0 = tf.random.uniform([], 0, num_time_steps - t, dtype=tf.int32)
                mask = tf.concat([tf.ones((spec_shape[0], num_mel_bins, t0, spec_shape[3])),
                                  tf.zeros((spec_shape[0], num_mel_bins, t, spec_shape[3])),
                                  tf.ones((spec_shape[0], num_mel_bins, num_time_steps - t0 - t, spec_shape[3]))], axis=2)
                spectrogram = spectrogram * mask
            return spectrogram
        
        @tf.function
        def random_gain_tf(spectrogram, min_gain_db=-6, max_gain_db=6):
            gain_db = tf.random.uniform([], min_gain_db, max_gain_db)
            gain_linear = tf.pow(10.0, gain_db / 20.0)
            return spectrogram * gain_linear
        
        @tf.function
        def mixup_batch_tf(batch_x, batch_y, alpha=0.2):
            batch_size = tf.shape(batch_x)[0]
            lambda_val = tf.random.uniform([], 0, alpha) 
            lambda_val = tf.maximum(lambda_val, 1 - lambda_val)
            
            indices = tf.random.shuffle(tf.range(batch_size))
            
            mixed_x = lambda_val * batch_x + (1 - lambda_val) * tf.gather(batch_x, indices)
            
            batch_y_float = tf.cast(batch_y, dtype=tf.float32)
            shuffled_y = tf.gather(batch_y_float, indices)
            mixed_y = lambda_val * batch_y_float + (1 - lambda_val) * shuffled_y
            return mixed_x, mixed_y
        
        return add_noise_tf, spec_augment_tf, random_gain_tf, mixup_batch_tf

    def _create_data_augmentation_pipeline(self):
        add_noise_fn, spec_augment_fn, random_gain_fn, mixup_fn = self._create_tf_augmentation_layers()
        
        class AudioAugmentationLayer(layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.add_noise = add_noise_fn
                self.spec_augment = spec_augment_fn
                self.random_gain = random_gain_fn
                
            def call(self, inputs, training=None):
                if not training:
                    return inputs
                
                x = inputs
                
                # Use tf.cond for conditional application of augmentations
                x = tf.cond(
                    tf.random.uniform([]) < 0.5,
                    lambda: self.add_noise(x, noise_factor=0.01),
                    lambda: x
                )
                x = tf.cond(
                    tf.random.uniform([]) < 0.8,
                    lambda: self.spec_augment(x, freq_mask_param=10, time_mask_param=20),
                    lambda: x
                )
                x = tf.cond(
                    tf.random.uniform([]) < 0.3,
                    lambda: self.random_gain(x, min_gain_db=-3, max_gain_db=3),
                    lambda: x
                )
                return x

        data_augmentation_pipeline = keras.Sequential([
            layers.RandomTranslation(height_factor=0.03, width_factor=0.03),
            layers.RandomZoom(0.03),
            AudioAugmentationLayer(),
        ], name="audio_augmentation_pipeline")
        
        return data_augmentation_pipeline, mixup_fn

    def build_model(self, input_shape, num_classes, use_mixup=False):
        inputs = layers.Input(shape=input_shape)
        x = layers.BatchNormalization()(inputs) 
        
        def conv_block(input_tensor, filters, kernel_size=(3,3), strides=(1,1), stage_name=""):
            x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', name=f"{stage_name}_conv1")(input_tensor)
            x = layers.BatchNormalization(name=f"{stage_name}_bn1")(x)
            x = layers.Activation('relu', name=f"{stage_name}_relu1")(x)
            
            x = layers.Conv2D(filters, kernel_size, padding='same', name=f"{stage_name}_conv2")(x)
            x = layers.BatchNormalization(name=f"{stage_name}_bn2")(x)
            
            if strides != (1,1) or input_tensor.shape[-1] != filters:
                shortcut = layers.Conv2D(filters, (1,1), strides=strides, padding='same', name=f"{stage_name}_shortcut_conv")(input_tensor)
                shortcut = layers.BatchNormalization(name=f"{stage_name}_shortcut_bn")(shortcut)
            else:
                shortcut = input_tensor
            
            x = layers.Add(name=f"{stage_name}_add")([x, shortcut])
            x = layers.Activation('relu', name=f"{stage_name}_relu2")(x)
            return x

        x = conv_block(x, 32, stage_name="block1")
        x = layers.MaxPooling2D((2, 2), name="block1_pool")(x)
        x = layers.Dropout(0.25, name="block1_drop")(x)

        x = conv_block(x, 64, stage_name="block2")
        x = layers.MaxPooling2D((2, 2), name="block2_pool")(x)
        x = layers.Dropout(0.25, name="block2_drop")(x)

        x = conv_block(x, 128, stage_name="block3")
        x = layers.MaxPooling2D((2, 2), name="block3_pool")(x)
        x = layers.Dropout(0.25, name="block3_drop")(x)
        
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        
        x = layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.001), name="dense1")(x)
        x = layers.BatchNormalization(name="dense1_bn")(x)
        x = layers.Activation('relu', name="dense1_relu")(x)
        x = layers.Dropout(0.4, name="dense1_drop")(x)
        
        outputs = layers.Dense(num_classes, activation='softmax', name="output")(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        loss = 'categorical_crossentropy' if use_mixup else 'sparse_categorical_crossentropy'
        
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        return model

    def _prepare_tf_datasets(self, X_train, y_train_encoded, X_val, y_val_encoded,
                              batch_size, num_classes, data_augmentation_pipeline, mixup_fn):
        
        y_train_onehot = tf.keras.utils.to_categorical(y_train_encoded, num_classes)
        y_val_onehot = tf.keras.utils.to_categorical(y_val_encoded, num_classes)

        def apply_mixup_randomly(batch_x, batch_y):
            return tf.cond(
                tf.random.uniform([]) < 0.25, 
                lambda: mixup_fn(batch_x, batch_y, alpha=0.2),
                lambda: (batch_x, tf.cast(batch_y, tf.float32)) 
            )

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_onehot))
        train_dataset = train_dataset.shuffle(buffer_size=len(X_train))
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.map(
            lambda x, y: (data_augmentation_pipeline(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        train_dataset = train_dataset.map(apply_mixup_randomly, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val_onehot))
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset, y_train_onehot, y_val_onehot


    def train_model(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        y_encoded = self.label_encoder.fit_transform(y)
        num_classes = len(self.label_encoder.classes_)
        
        X_train, X_val, y_train_enc, y_val_enc = train_test_split(
            X, y_encoded, test_size=validation_split, stratify=y_encoded, 
            random_state=42, shuffle=True
        )
        
        self.train_mean = np.mean(X_train, axis=(0,1,2), keepdims=True) 
        self.train_std = np.std(X_train, axis=(0,1,2), keepdims=True)
        
        if self.train_std < 1e-6: self.train_std = 1.0

        X_train = (X_train - self.train_mean) / self.train_std
        X_val = (X_val - self.train_mean) / self.train_std
        
        print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
        
        data_augmentation_pipeline, mixup_fn = self._create_data_augmentation_pipeline()
        
        self.model = self.build_model(X_train.shape[1:], num_classes, use_mixup=True)
        print(f"Model built for {num_classes} classes. Input shape: {X_train.shape[1:]}")
        
        train_dataset, val_dataset, y_train_onehot, y_val_onehot = self._prepare_tf_datasets(
            X_train, y_train_enc, X_val, y_val_enc, batch_size, num_classes,
            data_augmentation_pipeline, mixup_fn
        )

        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-6, verbose=1),
            PlotCallback()
        ]
        
        print("Starting training...")
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Evaluating model...")
        train_loss, train_acc = self.model.evaluate(X_train, y_train_onehot, verbose=0)
        val_loss, val_acc = self.model.evaluate(X_val, y_val_onehot, verbose=0)
        
        print(f"Final Training Accuracy: {train_acc:.4f}, Loss: {train_loss:.4f}")
        print(f"Final Validation Accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}")
        
        self.plot_final_results(history)
        
        y_pred_probs = self.model.predict(X_val)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        
        print("Classification Report:")
        print(classification_report(y_val_enc, y_pred_classes, target_names=self.label_encoder.classes_))
        self.plot_confusion_matrix(y_val_enc, y_pred_classes)
        
        return history

    def plot_final_results(self, history):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        
        plt.tight_layout(); plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('Actual')
        plt.show()

    def _get_metadata_path(self, model_filepath):
        return model_filepath.replace(".h5", "_metadata.pkl").replace(".keras", "_metadata.pkl")

    def save_model(self, filepath="music_genre_model.keras"):
        if self.model:
            self.model.save(filepath)
            metadata = {
                "label_encoder": self.label_encoder,
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

            print(f"Model loaded from {filepath}")
            print(f"Metadata loaded from {metadata_path}")
            print(f"Loaded model with segment length: {self.segment_length}s")

        except Exception as e:
            print(f"Error loading model or metadata: {e}")
            self.model = None 

    def predict_genre(self, audio_path, use_segments=True, aggregation_method='average'):
        if self.model is None or self.train_mean is None or self.train_std is None:
            print("Model or normalization stats not loaded. Train or load a model first.")
            return None
        
        if use_segments:
            return self.predict_genre_with_segments(audio_path, aggregation_method)
        else: 
            features = self.extract_features(audio_path, use_segments=False)
            if features is None: return None
            
            features_normalized = (features - self.train_mean) / self.train_std
            features_reshaped = features_normalized.reshape(1, *features_normalized.shape, 1)
            
            prediction = self.model.predict(features_reshaped)
            predicted_class_idx = np.argmax(prediction)
            confidence = prediction[0][predicted_class_idx]
            genre = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            return genre, confidence, prediction[0]

    def predict_genre_with_segments(self, audio_path, aggregation_method='average'):
        print(f"Predicting genre for: {os.path.basename(audio_path)} using {self.segment_length}s segments, {aggregation_method} aggregation.")
        
        segments_features_list = self.extract_features(audio_path, use_segments=True)
        if not segments_features_list:
            print("Could not extract features.")
            return None
        
        segments_array = np.array(segments_features_list)
        segments_normalized = (segments_array - self.train_mean) / self.train_std
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

    def load_audio_dataset(self):
        start_time = time.time()
        print(f"Loading audio dataset with {self.segment_length}s segments...")
        X_data, y_labels = [], []
        
        genres_root_path = os.path.join(self.data_path, 'genres_original')
        if not os.path.exists(genres_root_path):
            print(f"Error: Directory not found: {genres_root_path}")
            return None, None
        
        self.genres = [d for d in os.listdir(genres_root_path) if os.path.isdir(os.path.join(genres_root_path, d))]
        if not self.genres:
            print("Error: No genre subdirectories found.")
            return None, None
        
        print(f"Found {len(self.genres)} genres: {', '.join(self.genres)}")
        
        total_files_to_process = sum(len(files) for _, _, files in os.walk(genres_root_path) if any(f.endswith(('.au', '.wav')) for f in files))
        processed_files_count = 0
        total_segments_created = 0

        for genre_name in self.genres:
            genre_dir_path = os.path.join(genres_root_path, genre_name)
            print(f"Processing {genre_name} files...")
            genre_segments_count = 0
            for filename in os.listdir(genre_dir_path):
                if filename.endswith(('.au', '.wav')):
                    file_full_path = os.path.join(genre_dir_path, filename)
                    segment_features_list = self.extract_features(file_full_path, use_segments=True)
                    if segment_features_list:
                        for feature_matrix in segment_features_list:
                            X_data.append(feature_matrix)
                            y_labels.append(genre_name)
                            total_segments_created += 1
                            genre_segments_count +=1
                        processed_files_count += 1
                    
                    if processed_files_count > 0 and processed_files_count % 50 == 0 and total_files_to_process > 0 :
                        progress = (processed_files_count / total_files_to_process) * 100
                        print(f"Progress: {processed_files_count}/{total_files_to_process} files ({progress:.1f}%) - Segments: {total_segments_created}")
            print(f"Completed {genre_name}: {genre_segments_count} segments from its files.")

        if not X_data:
            print("Error: No audio files processed successfully.")
            return None, None
        
        X_np = np.array(X_data).reshape(len(X_data), self.n_mels, -1, 1) 
        y_np = np.array(y_labels)
        
        print(f"Dataset creation completed in {(time.time() - start_time):.2f}s. Shape: {X_np.shape}")
        print(f"Total files processed: {processed_files_count}, Total segments: {total_segments_created}")
        return X_np, y_np

    def clear_cache(self):
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
                print("Cache cleared successfully.")
            except Exception as e:
                print(f"Error clearing cache: {str(e)}")
        else:
            print("No cache file found.")

    def visualize_augmentations(self, audio_path, sr=DEFAULT_SR):
        print(f"Visualizing augmentations for: {os.path.basename(audio_path)}")
        y_original, _ = librosa.load(audio_path, duration=10, sr=sr)
        
        vis_duration_sec = 5 
        expected_frames_vis = int((vis_duration_sec * sr) / self.hop_length) + 1
        
        if len(y_original) < vis_duration_sec * sr:
             y_original = np.pad(y_original, (0, vis_duration_sec * sr - len(y_original)), mode='constant')
        y_original_segment = y_original[:vis_duration_sec * sr]


        mel_orig_norm = self._create_mel_spectrogram(y_original_segment, sr, expected_frames_vis)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Audio Augmentation Visualization: {os.path.basename(audio_path)}', fontsize=16)

        def plot_spec(ax, data, title):
            img = librosa.display.specshow(data, sr=sr, hop_length=self.hop_length, 
                                           x_axis='time', y_axis='mel', ax=ax)
            ax.set_title(title)

        plot_spec(axes[0,0], librosa.power_to_db(librosa.feature.melspectrogram(y=y_original_segment, sr=sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length)), 'Original (dB)')


        y_stretched = librosa.effects.time_stretch(y_original_segment, rate=0.8)
        plot_spec(axes[0,1], self._create_mel_spectrogram(y_stretched, sr, expected_frames_vis), 'Time Stretched (0.8x)')
        
        y_pitched = librosa.effects.pitch_shift(y_original_segment, sr=sr, n_steps=2)
        plot_spec(axes[0,2], self._create_mel_spectrogram(y_pitched, sr, expected_frames_vis), 'Pitch Shifted (+2 semitones)')

        noise = np.random.normal(0, 0.01, len(y_original_segment))
        y_noisy = y_original_segment + noise
        plot_spec(axes[1,0], self._create_mel_spectrogram(y_noisy, sr, expected_frames_vis), 'Noise Added')

        spec_tf_input = tf.constant(mel_orig_norm.reshape(1, *mel_orig_norm.shape, 1), dtype=tf.float32)
        
        add_noise_fn, spec_augment_fn, _, _ = self._create_tf_augmentation_layers()

        spec_freq_masked = spec_augment_fn(spec_tf_input, freq_mask_param=20, time_mask_param=0, num_freq_masks=1, num_time_masks=0) 
        plot_spec(axes[1,1], spec_freq_masked.numpy().squeeze(), 'Frequency Masked (SpecAugment)')
        
        spec_time_masked = spec_augment_fn(spec_tf_input, freq_mask_param=0, time_mask_param=40, num_freq_masks=0, num_time_masks=1) 
        plot_spec(axes[1,2], spec_time_masked.numpy().squeeze(), 'Time Masked (SpecAugment)')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def visualize_segment_predictions(self, audio_path, max_segments_to_show=10):
        print(f"Visualizing segment predictions for: {os.path.basename(audio_path)}")
        result_tuple = self.predict_genre_with_segments(audio_path, aggregation_method='average')
        if not result_tuple: return
        
        genre, confidence, final_pred_probs, segment_predictions = result_tuple
        
        num_segments_actual = segment_predictions.shape[0]
        num_segments_to_plot = min(num_segments_actual, max_segments_to_show)
        segment_preds_subset = segment_predictions[:num_segments_to_plot]
        
        plt.figure(figsize=(15, 10))
        genre_names = self.label_encoder.classes_
        final_predicted_class_idx = np.argmax(final_pred_probs)

        ax1 = plt.subplot(2, 2, 1)
        confidences_subset = np.max(segment_preds_subset, axis=1)
        predicted_classes_subset = np.argmax(segment_preds_subset, axis=1)
        bar_colors = ['red' if pred_idx == final_predicted_class_idx else 'blue' for pred_idx in predicted_classes_subset]
        ax1.bar(range(num_segments_to_plot), confidences_subset, color=bar_colors, alpha=0.7)
        ax1.set_title(f'Segment Confidences (Max {max_segments_to_show})\n(Red = Final Prediction: {genre})')
        ax1.set_xlabel('Segment Number'); ax1.set_ylabel('Confidence'); ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(2, 2, 2)
        unique_classes_subset, counts_subset = np.unique(predicted_classes_subset, return_counts=True)
        unique_names_subset = [genre_names[cls_idx] for cls_idx in unique_classes_subset]
        ax2.pie(counts_subset, labels=unique_names_subset, autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'Distribution of Segment Predictions (Max {max_segments_to_show} Segments)')

        ax3 = plt.subplot(2, 2, 3)
        ax3.bar(genre_names, final_pred_probs, alpha=0.7)
        ax3.set_title('Final Aggregated Probabilities')
        ax3.set_xlabel('Genre'); ax3.set_ylabel('Probability')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
        ax3.grid(True, alpha=0.3)
        
        ax4 = plt.subplot(2, 2, 4)
        if num_segments_to_plot <= 20: 
            sns.heatmap(segment_preds_subset.T, 
                       xticklabels=[f'Seg{i+1}' for i in range(num_segments_to_plot)],
                       yticklabels=genre_names, annot=False, cmap='Blues', cbar=True, ax=ax4)
            ax4.set_title('Prediction Probabilities Heatmap')
        else:
            ax4.text(0.5, 0.5, f'Too many segments ({num_segments_to_plot})\nto display heatmap', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Segment Heatmap')
        ax4.set_xlabel('Segment'); ax4.set_ylabel('Genre')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main_start_time = time.time()
    print("Music Genre Classification Neural Network")
    print("Started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    classifier = MusicGenreClassifier(segment_length=4)
    
    X_data, y_data = classifier.create_dataset()
    
    if X_data is None or y_data is None or len(X_data) == 0:
        print("Dataset creation failed or resulted in no data. Exiting.")
        exit(1)
    
    print(f"Dataset shape: {X_data.shape}")
    print(f"Unique Genres found in data: {np.unique(y_data)}")
    
    print("Starting model training...")
    history = classifier.train_model(X_data, y_data, epochs=30, batch_size=32)
    
    classifier.save_model()
    
    total_exec_time = time.time() - main_start_time
    print(f"Total execution time: {total_exec_time/60:.2f} minutes")
    print("Training completed! Model saved.")