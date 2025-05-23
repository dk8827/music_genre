import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Try to import tensorflow-addons, but make it optional
try:
    import tensorflow_addons as tfa
    TFA_AVAILABLE = True
    print("TensorFlow Addons imported successfully")
except ImportError as e:
    TFA_AVAILABLE = False
    print("TensorFlow Addons not available, using custom implementations")

import warnings
import time
from datetime import datetime
import joblib
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class MusicGenreClassifier:
    def __init__(self, data_path="music_data", n_mels=128, n_fft=2048, hop_length=512, segment_length=4):
        self.data_path = data_path
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment_length = segment_length  # Length of each segment in seconds
        self.model = None
        self.label_encoder = LabelEncoder()
        self.genres = []
        self.cache_file = f"processed_features_seg{segment_length}s.npz"
        print(f"Initialized MusicGenreClassifier:")
        print(f"  Data path: {data_path}")
        print(f"  Mel bands: {n_mels}")
        print(f"  FFT size: {n_fft}")
        print(f"  Hop length: {hop_length}")
        print(f"  Segment length: {segment_length} seconds")
        print(f"  Cache file: {self.cache_file}")
        
    def extract_features(self, audio_path, duration=30, use_segments=True):
        """Extract mel-spectrogram features from audio file with optional segmentation"""
        try:
            start_time = time.time()
            
            # Load audio file
            y, sr = librosa.load(audio_path, duration=duration, sr=22050)
            
            if use_segments:
                # Break audio into segments
                segment_samples = int(self.segment_length * sr)
                segments = []
                
                # Calculate number of segments that fit in the audio
                num_segments = len(y) // segment_samples
                
                if num_segments == 0:
                    # Audio is shorter than segment length, pad it
                    y_padded = np.pad(y, (0, segment_samples - len(y)), mode='constant')
                    segments.append(y_padded)
                else:
                    # Extract non-overlapping segments
                    for i in range(num_segments):
                        start_idx = i * segment_samples
                        end_idx = start_idx + segment_samples
                        segment = y[start_idx:end_idx]
                        segments.append(segment)
                    
                    # Handle remaining audio if it's more than half a segment
                    remaining_samples = len(y) % segment_samples
                    if remaining_samples > segment_samples // 2:
                        remaining_audio = y[-remaining_samples:]
                        # Pad to full segment length
                        padded_remaining = np.pad(remaining_audio, 
                                                (0, segment_samples - len(remaining_audio)), 
                                                mode='constant')
                        segments.append(padded_remaining)
                
                # Process each segment
                segment_features = []
                for i, segment in enumerate(segments):
                    # Apply random audio augmentation during loading (20% chance)
                    if np.random.random() < 0.2:
                        segment = self.apply_audio_augmentation(segment, sr)
                    
                    # Extract mel-spectrogram for this segment
                    mel_spec = librosa.feature.melspectrogram(
                        y=segment, sr=sr, n_mels=self.n_mels, 
                        n_fft=self.n_fft, hop_length=self.hop_length
                    )
                    
                    # Convert to dB scale
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    
                    # Normalize
                    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
                    
                    # Ensure consistent shape
                    expected_frames = int((self.segment_length * sr) / self.hop_length) + 1
                    if mel_spec_norm.shape[1] > expected_frames:
                        mel_spec_norm = mel_spec_norm[:, :expected_frames]
                    elif mel_spec_norm.shape[1] < expected_frames:
                        pad_width = expected_frames - mel_spec_norm.shape[1]
                        mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, pad_width)), mode='constant')
                    
                    segment_features.append(mel_spec_norm)
                
                processing_time = time.time() - start_time
                
                return segment_features
            
            else:
                # Original single-file processing (for backward compatibility)
                # Apply random audio augmentation during loading (20% chance)
                if np.random.random() < 0.2:
                    y = self.apply_audio_augmentation(y, sr)
                
                # Ensure consistent length
                target_length = duration * sr
                if len(y) < target_length:
                    y = np.pad(y, (0, target_length - len(y)))
                else:
                    y = y[:target_length]
                
                # Extract mel-spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=y, sr=sr, n_mels=self.n_mels, 
                    n_fft=self.n_fft, hop_length=self.hop_length
                )
                
                # Convert to dB scale
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Normalize
                mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
                
                processing_time = time.time() - start_time
                
                # Ensure the output shape is consistent
                expected_frames = int((duration * sr) / self.hop_length) + 1
                if mel_spec_norm.shape[1] > expected_frames:
                    mel_spec_norm = mel_spec_norm[:, :expected_frames]
                elif mel_spec_norm.shape[1] < expected_frames:
                    pad_width = expected_frames - mel_spec_norm.shape[1]
                    mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, pad_width)), mode='constant')
                
                return mel_spec_norm
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None
    
    def apply_audio_augmentation(self, y, sr):
        """Apply librosa-based audio augmentation techniques"""
        augmented = y.copy()
        
        # Time stretching (80% to 120% speed)
        if np.random.random() < 0.3:
            stretch_factor = np.random.uniform(0.8, 1.2)
            try:
                augmented = librosa.effects.time_stretch(augmented, rate=stretch_factor)
            except:
                pass  # Fall back to original if time stretching fails
        
        # Pitch shifting (-2 to +2 semitones)
        if np.random.random() < 0.3:
            n_steps = np.random.uniform(-2, 2)
            try:
                augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=n_steps)
            except:
                pass  # Fall back to original if pitch shifting fails
        
        # Add background noise
        if np.random.random() < 0.2:
            noise_factor = np.random.uniform(0.005, 0.02)
            noise = np.random.normal(0, noise_factor, len(augmented))
            augmented = augmented + noise
        
        # Random gain
        if np.random.random() < 0.3:
            gain_factor = np.random.uniform(0.7, 1.3)
            augmented = augmented * gain_factor
        
        return augmented
    
    def create_dataset(self):
        """Create dataset from audio files with caching"""
        print("Creating dataset from audio files...")
        
        # Check if cached data exists
        if os.path.exists(self.cache_file):
            print("Loading cached features...")
            try:
                cached_data = np.load(self.cache_file, allow_pickle=True)
                X = cached_data['features']
                y = cached_data['labels']
                self.genres = cached_data['genres']
                print("Successfully loaded cached features")
                print(f"Dataset shape: {X.shape}")
                print("Number of samples per genre:")
                unique_genres, counts = np.unique(y, return_counts=True)
                for genre, count in zip(unique_genres, counts):
                    print(f"  {genre}: {count} samples")
                return X, y
            except Exception as e:
                print(f"Error loading cached data: {str(e)}")
                print("Processing files from scratch...")
        
        # If no cache or error, process the files
        X, y = self.load_audio_dataset()
        
        # Save processed features to cache
        if X is not None and y is not None:
            print("Saving features to cache...")
            try:
                np.savez(
                    self.cache_file,
                    features=X,
                    labels=y,
                    genres=self.genres
                )
                print("Successfully cached features")
            except Exception as e:
                print(f"Error caching features: {str(e)}")
        
        return X, y
    
    def create_audio_augmentation_layers(self):
        """Create custom audio-specific augmentation layers"""
        
        @tf.function
        def add_noise(spectrogram, noise_factor=0.005):
            """Add Gaussian noise to spectrogram"""
            noise = tf.random.normal(tf.shape(spectrogram), stddev=noise_factor)
            return spectrogram + noise
        
        @tf.function 
        def spec_augment(spectrogram, freq_mask_param=15, time_mask_param=35, num_freq_masks=1, num_time_masks=1):
            """Apply SpecAugment: frequency and time masking"""
            spec_shape = tf.shape(spectrogram)
            freq_bins = spec_shape[1]
            time_steps = spec_shape[2]
            
            # Frequency masking
            for _ in range(num_freq_masks):
                f = tf.random.uniform([], 0, freq_mask_param, dtype=tf.int32)
                f = tf.minimum(f, freq_bins)
                f0 = tf.random.uniform([], 0, freq_bins - f, dtype=tf.int32)
                
                # Create mask
                freq_mask = tf.ones([freq_bins])
                freq_mask = tf.tensor_scatter_nd_update(
                    freq_mask,
                    tf.reshape(tf.range(f0, f0 + f), [-1, 1]),
                    tf.zeros([f])
                )
                freq_mask = tf.reshape(freq_mask, [1, freq_bins, 1, 1])
                spectrogram = spectrogram * freq_mask
            
            # Time masking  
            for _ in range(num_time_masks):
                t = tf.random.uniform([], 0, time_mask_param, dtype=tf.int32)
                t = tf.minimum(t, time_steps)
                t0 = tf.random.uniform([], 0, time_steps - t, dtype=tf.int32)
                
                # Create mask
                time_mask = tf.ones([time_steps])
                time_mask = tf.tensor_scatter_nd_update(
                    time_mask,
                    tf.reshape(tf.range(t0, t0 + t), [-1, 1]),
                    tf.zeros([t])
                )
                time_mask = tf.reshape(time_mask, [1, 1, time_steps, 1])
                spectrogram = spectrogram * time_mask
            
            return spectrogram
        
        @tf.function
        def random_gain(spectrogram, min_gain_db=-6, max_gain_db=6):
            """Apply random gain to spectrogram"""
            gain_db = tf.random.uniform([], min_gain_db, max_gain_db)
            gain_linear = tf.pow(10.0, gain_db / 20.0)
            return spectrogram * gain_linear
        
        @tf.function
        def mixup_batch(batch_x, batch_y, alpha=0.2):
            """Apply mixup augmentation to a batch"""
            batch_size = tf.shape(batch_x)[0]
            
            # Generate lambda from beta distribution
            lambda_val = tf.random.uniform([], 0, alpha)
            lambda_val = tf.maximum(lambda_val, 1 - lambda_val)
            
            # Shuffle indices for mixing
            indices = tf.random.shuffle(tf.range(batch_size))
            
            # Mix inputs
            mixed_x = lambda_val * batch_x + (1 - lambda_val) * tf.gather(batch_x, indices)
            
            # Mix labels (already one-hot encoded)
            batch_y = tf.cast(batch_y, dtype=tf.float32)
            shuffled_y = tf.gather(batch_y, indices)
            shuffled_y = tf.cast(shuffled_y, dtype=tf.float32)
            mixed_y = lambda_val * batch_y + (1 - lambda_val) * shuffled_y
            
            return mixed_x, mixed_y
        
        return add_noise, spec_augment, random_gain, mixup_batch
    
    def create_advanced_augmentation_pipeline(self):
        """Create sophisticated audio-specific data augmentation pipeline"""
        
        # Get custom augmentation functions
        add_noise, spec_augment, random_gain, mixup_batch = self.create_audio_augmentation_layers()
        
        # Create augmentation layer
        class AudioAugmentationLayer(layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.add_noise = add_noise
                self.spec_augment = spec_augment
                self.random_gain = random_gain
                
            def call(self, inputs, training=None):
                if not training:
                    return inputs
                
                # Apply augmentations randomly with certain probabilities using tf.cond
                # Add noise with 50% chance
                inputs = tf.cond(
                    tf.random.uniform([]) < 0.5,
                    lambda: self.add_noise(inputs, noise_factor=0.01),
                    lambda: inputs
                )
                
                # Apply SpecAugment with 80% chance
                inputs = tf.cond(
                    tf.random.uniform([]) < 0.8,
                    lambda: self.spec_augment(
                        inputs, 
                        freq_mask_param=10, 
                        time_mask_param=20,
                        num_freq_masks=1,
                        num_time_masks=1
                    ),
                    lambda: inputs
                )
                
                # Apply random gain with 30% chance
                inputs = tf.cond(
                    tf.random.uniform([]) < 0.3,
                    lambda: self.random_gain(inputs, min_gain_db=-3, max_gain_db=3),
                    lambda: inputs
                )
                
                return inputs
        
        # Create complete augmentation pipeline
        data_augmentation = keras.Sequential([
            # Traditional image augmentations (still useful for spectrograms)
            layers.RandomTranslation(height_factor=0.03, width_factor=0.03),
            layers.RandomZoom(0.03),
            
            # Audio-specific augmentations
            AudioAugmentationLayer(),
        ], name="audio_augmentation_pipeline")
        
        return data_augmentation, mixup_batch
    
    def build_model(self, input_shape, num_classes, use_mixup=False):
        """Build CNN model for music genre classification"""
        # Input layer with batch normalization
        inputs = layers.Input(shape=input_shape)
        x = layers.BatchNormalization()(inputs)
        
        # First convolutional block with residual connection
        conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        conv1 = layers.BatchNormalization()(conv1)
        conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        conv1 = layers.BatchNormalization()(conv1)
        pool1 = layers.MaxPooling2D((2, 2))(conv1)
        drop1 = layers.Dropout(0.25)(pool1)
        
        # Second convolutional block with residual connection
        conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(drop1)
        conv2 = layers.BatchNormalization()(conv2)
        conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        conv2 = layers.BatchNormalization()(conv2)
        pool2 = layers.MaxPooling2D((2, 2))(conv2)
        drop2 = layers.Dropout(0.25)(pool2)
        
        # Third convolutional block with residual connection
        conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(drop2)
        conv3 = layers.BatchNormalization()(conv3)
        conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        conv3 = layers.BatchNormalization()(conv3)
        pool3 = layers.MaxPooling2D((2, 2))(conv3)
        drop3 = layers.Dropout(0.25)(pool3)
        
        # Global pooling and feature extraction
        gap = layers.GlobalAveragePooling2D()(drop3)
        
        # Dense layers with stronger regularization
        dense1 = layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.001))(gap)
        dense1 = layers.BatchNormalization()(dense1)
        dense1 = layers.Activation('relu')(dense1)
        dense1 = layers.Dropout(0.4)(dense1)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax')(dense1)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Use a simple learning rate with Adam optimizer
        optimizer = keras.optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0
        )
        
        # Choose loss function based on whether mixup is used
        if use_mixup:
            loss = 'categorical_crossentropy'  # For mixup with one-hot labels
            print("Using categorical crossentropy for mixup compatibility")
        else:
            loss = 'sparse_categorical_crossentropy'  # For integer labels
            print("Using sparse categorical crossentropy for standard training")
        
        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """Train the model with real-time plotting"""
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Normalize input data
        X = (X - X.mean()) / X.std()
        
        # Split data with shuffling
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=validation_split, stratify=y_encoded, 
            random_state=42, shuffle=True
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        
        # Create advanced audio-specific data augmentation pipeline
        print("Creating audio augmentation pipeline...")
        data_augmentation, mixup_batch = self.create_advanced_augmentation_pipeline()
        print("Augmentation pipeline created")
        
        # Build model
        input_shape = X_train.shape[1:]
        num_classes = len(np.unique(y_encoded))
        self.model = self.build_model(input_shape, num_classes, use_mixup=True)
        
        print(f"Model built for {num_classes} classes")
        print(f"Input shape: {input_shape}")
        print("Model configured for audio augmentation with mixup support")
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
        
        # Custom callback for real-time plotting
        class PlotCallback(keras.callbacks.Callback):
            def __init__(self):
                self.train_acc = []
                self.val_acc = []
                self.train_loss = []
                self.val_loss = []
                
            def on_epoch_end(self, epoch, logs=None):
                self.train_acc.append(logs.get('accuracy'))
                self.val_acc.append(logs.get('val_accuracy'))
                self.train_loss.append(logs.get('loss'))
                self.val_loss.append(logs.get('val_loss'))
                
                # Plot every 5 epochs
                if (epoch + 1) % 5 == 0:
                    self.plot_metrics()
            
            def plot_metrics(self):
                plt.figure(figsize=(15, 5))
                
                # Accuracy plot
                plt.subplot(1, 2, 1)
                plt.plot(self.train_acc, label='Training Accuracy', color='blue')
                plt.plot(self.val_acc, label='Validation Accuracy', color='red')
                plt.title('Model Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True)
                
                # Loss plot
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
        
        plot_callback = PlotCallback()
        
        # Train model with data augmentation
        print("Starting training with audio augmentation...")
        print("Augmentation techniques enabled:")
        print("  - SpecAugment (Time & Frequency Masking)")
        print("  - Gaussian Noise Addition")
        print("  - Random Gain")
        print("  - Spatial Transformations")
        print("  - Mixup (25% of batches)")
        
        # Create training dataset with sophisticated augmentation
        def apply_mixup_randomly(batch_x, batch_y):
            """Apply mixup to random 25% of batches"""
            return tf.cond(
                tf.random.uniform([]) < 0.25,  # 25% chance for mixup
                lambda: mixup_batch(batch_x, batch_y, alpha=0.2),
                lambda: (batch_x, tf.cast(batch_y, tf.float32))  # Ensure false_fn also returns float32 for labels
            )
        
        # Convert labels to one-hot format for consistent handling
        num_classes = len(np.unique(y_train))
        y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)
        y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_onehot))
        train_dataset = train_dataset.shuffle(buffer_size=len(X_train))
        train_dataset = train_dataset.batch(batch_size)
        
        # Apply augmentation pipeline
        train_dataset = train_dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Apply mixup occasionally - now both branches return one-hot labels
        train_dataset = train_dataset.map(
            apply_mixup_randomly,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Create validation dataset with one-hot labels
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val_onehot))
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Train the model
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=[early_stopping, reduce_lr, plot_callback],
            verbose=1
        )
        
        # Final evaluation
        print("Evaluating model...")
        train_loss, train_acc = self.model.evaluate(X_train, y_train_onehot, verbose=0)
        val_loss, val_acc = self.model.evaluate(X_val, y_val_onehot, verbose=0)
        
        print(f"Final Training Accuracy: {train_acc:.4f}")
        print(f"Final Validation Accuracy: {val_acc:.4f}")
        
        # Plot final results
        self.plot_final_results(history)
        
        # Classification report - convert back to class indices for sklearn
        y_pred = self.model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_val_classes = np.argmax(y_val_onehot, axis=1)
        
        print("Classification Report:")
        print(classification_report(
            y_val_classes, y_pred_classes, 
            target_names=self.label_encoder.classes_
        ))
        
        # Confusion matrix
        self.plot_confusion_matrix(y_val_classes, y_pred_classes)
        
        return history
    
    def plot_final_results(self, history):
        """Plot final training results"""
        plt.figure(figsize=(15, 5))
        
        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def save_model(self, filepath="music_genre_model.h5"):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(filepath)
            
            # Save label encoder
            joblib.dump(self.label_encoder, "label_encoder.pkl")
            
            print(f"Model saved to {filepath}")
            print("Label encoder saved to label_encoder.pkl")
        else:
            print("No model to save. Train the model first.")
    
    def load_model(self, filepath="music_genre_model.h5"):
        """Load a saved model"""
        self.model = keras.models.load_model(filepath)
        
        # Load label encoder
        self.label_encoder = joblib.load("label_encoder.pkl")
        
        print(f"Model loaded from {filepath}")
    
    def predict_genre(self, audio_path, use_segments=True, aggregation_method='average'):
        """Predict genre for a single audio file using segment-based prediction"""
        if self.model is None:
            print("No model loaded. Train or load a model first.")
            return None
        
        if use_segments:
            return self.predict_genre_with_segments(audio_path, aggregation_method)
        else:
            # Original single prediction (for backward compatibility)
            features = self.extract_features(audio_path, use_segments=False)
            if features is None:
                return None
            
            # Reshape for prediction
            features = features.reshape(1, features.shape[0], features.shape[1], 1)
            
            # Predict
            prediction = self.model.predict(features)
            predicted_class = np.argmax(prediction)
            confidence = prediction[0][predicted_class]
            
            genre = self.label_encoder.inverse_transform([predicted_class])[0]
            
            return genre, confidence, prediction[0]
    
    def predict_genre_with_segments(self, audio_path, aggregation_method='average'):
        """Predict genre using multiple segments with aggregation"""
        print(f"Predicting genre for: {os.path.basename(audio_path)}")
        print(f"Using {self.segment_length}-second segments with {aggregation_method} aggregation")
        
        # Extract features for all segments
        segments_features = self.extract_features(audio_path, use_segments=True)
        if segments_features is None:
            return None
        
        # Prepare segments for prediction
        segments_array = np.array(segments_features)
        segments_array = segments_array.reshape(
            segments_array.shape[0], segments_array.shape[1], segments_array.shape[2], 1
        )
        
        print(f"Predicting on {len(segments_features)} segments...")
        
        # Predict on all segments
        segment_predictions = self.model.predict(segments_array, verbose=0)
        
        # Apply aggregation strategy
        if aggregation_method == 'average':
            # Average the probabilities across all segments
            final_prediction = np.mean(segment_predictions, axis=0)
            predicted_class = np.argmax(final_prediction)
            confidence = final_prediction[predicted_class]
            
        elif aggregation_method == 'majority_vote':
            # Majority vote on predicted classes
            segment_classes = np.argmax(segment_predictions, axis=1)
            predicted_class = np.bincount(segment_classes).argmax()
            
            # Calculate confidence as the proportion of segments that voted for this class
            confidence = np.sum(segment_classes == predicted_class) / len(segment_classes)
            final_prediction = segment_predictions[0] * 0  # Initialize with zeros
            final_prediction[predicted_class] = confidence
            
        elif aggregation_method == 'max_confidence':
            # Use the prediction with the highest confidence
            max_confidence_idx = np.argmax(np.max(segment_predictions, axis=1))
            final_prediction = segment_predictions[max_confidence_idx]
            predicted_class = np.argmax(final_prediction)
            confidence = final_prediction[predicted_class]
            
        elif aggregation_method == 'weighted_average':
            # Weight predictions by their confidence
            confidences = np.max(segment_predictions, axis=1)
            weights = confidences / np.sum(confidences)
            final_prediction = np.average(segment_predictions, axis=0, weights=weights)
            predicted_class = np.argmax(final_prediction)
            confidence = final_prediction[predicted_class]
            
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        genre = self.label_encoder.inverse_transform([predicted_class])[0]
        
        # Print detailed results
        print(f"Segment prediction details:")
        print(f"  Number of segments: {len(segments_features)}")
        print(f"  Aggregation method: {aggregation_method}")
        print(f"  Predicted genre: {genre}")
        print(f"  Confidence: {confidence:.4f}")
        
        # Show individual segment predictions
        if len(segments_features) <= 10:  # Only show details for reasonable number of segments
            print(f"  Individual segment votes:")
            for i, pred in enumerate(segment_predictions):
                seg_class = np.argmax(pred)
                seg_genre = self.label_encoder.inverse_transform([seg_class])[0]
                seg_conf = pred[seg_class]
                print(f"    Segment {i+1}: {seg_genre} ({seg_conf:.3f})")
        
        return genre, confidence, final_prediction, segment_predictions
    
    def compare_aggregation_methods(self, audio_path):
        """Compare different aggregation methods on the same audio file"""
        methods = ['average', 'majority_vote', 'max_confidence', 'weighted_average']
        
        print(f"Comparing aggregation methods for: {os.path.basename(audio_path)}")
        print("-" * 60)
        
        results = {}
        
        for method in methods:
            print(f"Testing {method} aggregation:")
            result = self.predict_genre_with_segments(audio_path, aggregation_method=method)
            if result is not None:
                genre, confidence, final_pred, segment_preds = result
                results[method] = {
                    'genre': genre,
                    'confidence': confidence,
                    'prediction': final_pred
                }
                print(f"  Result: {genre} (confidence: {confidence:.4f})")
        
        # Summary
        print(f"SUMMARY FOR {os.path.basename(audio_path)}:")
        print("-" * 40)
        for method, result in results.items():
            print(f"{method:15}: {result['genre']:10} ({result['confidence']:.4f})")
        
        return results
    
    def load_audio_dataset(self):
        """Load and process real audio files from the dataset with segmentation"""
        start_time = time.time()
        print("Loading audio dataset with segmentation...")
        
        X = []
        y = []
        
        # Path to the genres directory
        genres_path = os.path.join(self.data_path, 'genres_original')
        print(f"Looking for audio files in: {genres_path}")
        
        if not os.path.exists(genres_path):
            print(f"Error: Directory not found: {genres_path}")
            print("Please make sure you have downloaded and extracted the GTZAN dataset correctly.")
            return None, None
        
        # Get list of genres from directory names
        self.genres = [d for d in os.listdir(genres_path) 
                      if os.path.isdir(os.path.join(genres_path, d))]
        
        if not self.genres:
            print("Error: No genre directories found!")
            return None, None
        
        print(f"Found {len(self.genres)} genres: {', '.join(self.genres)}")
        print(f"Using {self.segment_length}-second segments for training data augmentation")
        
        total_files = 0
        processed_files = 0
        total_segments = 0
        
        # Count total files first
        for genre in self.genres:
            genre_path = os.path.join(genres_path, genre)
            total_files += len([f for f in os.listdir(genre_path) 
                              if f.endswith(('.au', '.wav'))])
        
        # Process each genre
        for genre in self.genres:
            genre_path = os.path.join(genres_path, genre)
            genre_start_time = time.time()
            genre_segments = 0
            print(f"Processing {genre} files...")
            
            # Process each audio file in the genre folder
            genre_files = [f for f in os.listdir(genre_path) 
                         if f.endswith(('.au', '.wav'))]
            
            for filename in genre_files:
                file_path = os.path.join(genre_path, filename)
                
                # Extract features (returns list of segments)
                segments_features = self.extract_features(file_path, use_segments=True)
                if segments_features is not None:
                    # Add each segment as a separate training sample
                    for segment_feature in segments_features:
                        X.append(segment_feature)
                        y.append(genre)
                        total_segments += 1
                        genre_segments += 1
                    
                    processed_files += 1
                
                # Show progress every 50 files
                if processed_files % 50 == 0:
                    progress = (processed_files / total_files) * 100
                    print(f"Progress: {processed_files}/{total_files} files ({progress:.1f}%)")
                    print(f"Total segments created: {total_segments}")
            
            genre_time = time.time() - genre_start_time
            print(f"Completed {genre}: {len(genre_files)} files -> {genre_segments} segments in {genre_time:.2f} seconds")
        
        if not X:
            print("Error: No files were processed successfully!")
            return None, None
        
        X = np.array(X)
        y = np.array(y)
        
        # Add channel dimension for CNN
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        
        total_time = time.time() - start_time
        print(f"Dataset creation completed in {total_time:.2f} seconds")
        print(f"Final dataset shape: {X.shape}")
        print(f"Data augmentation through segmentation:")
        print(f"  Original files: {processed_files}")
        print(f"  Training segments: {total_segments}")
        print(f"  Augmentation factor: {total_segments/processed_files:.1f}x")
        print("Number of segments per genre:")
        for genre in self.genres:
            count = np.sum(y == genre)
            print(f"  {genre}: {count} segments")
        
        return X, y
    
    def clear_cache(self):
        """Clear the cached features"""
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
                print("Cache cleared successfully")
            except Exception as e:
                print(f"Error clearing cache: {str(e)}")
        else:
            print("No cache file found.")
    
    def visualize_augmentations(self, audio_path, num_augmentations=4):
        """Visualize different augmentation techniques on a sample audio file"""
        try:
            print(f"Visualizing augmentations for: {os.path.basename(audio_path)}")
            
            # Load original audio
            y_original, sr = librosa.load(audio_path, duration=10, sr=22050)
            
            # Create original spectrogram
            mel_spec_original = librosa.feature.melspectrogram(
                y=y_original, sr=sr, n_mels=self.n_mels, 
                n_fft=self.n_fft, hop_length=self.hop_length
            )
            mel_spec_original_db = librosa.power_to_db(mel_spec_original, ref=np.max)
            
            # Create figure
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Audio Augmentation Visualization: {os.path.basename(audio_path)}', fontsize=16)
            
            # Original
            librosa.display.specshow(mel_spec_original_db, sr=sr, hop_length=self.hop_length, 
                                   x_axis='time', y_axis='mel', ax=axes[0,0])
            axes[0,0].set_title('Original')
            axes[0,0].set_xlabel('Time')
            axes[0,0].set_ylabel('Mel Frequency')
            
            # Time stretching
            try:
                y_stretched = librosa.effects.time_stretch(y_original, rate=0.8)
                mel_spec_stretched = librosa.feature.melspectrogram(
                    y=y_stretched[:len(y_original)], sr=sr, n_mels=self.n_mels, 
                    n_fft=self.n_fft, hop_length=self.hop_length
                )
                mel_spec_stretched_db = librosa.power_to_db(mel_spec_stretched, ref=np.max)
                librosa.display.specshow(mel_spec_stretched_db, sr=sr, hop_length=self.hop_length, 
                                       x_axis='time', y_axis='mel', ax=axes[0,1])
                axes[0,1].set_title('Time Stretched (0.8x)')
            except:
                axes[0,1].text(0.5, 0.5, 'Time Stretch\nNot Available', ha='center', va='center')
                axes[0,1].set_title('Time Stretched (Failed)')
            
            # Pitch shifting
            try:
                y_pitched = librosa.effects.pitch_shift(y_original, sr=sr, n_steps=2)
                mel_spec_pitched = librosa.feature.melspectrogram(
                    y=y_pitched, sr=sr, n_mels=self.n_mels, 
                    n_fft=self.n_fft, hop_length=self.hop_length
                )
                mel_spec_pitched_db = librosa.power_to_db(mel_spec_pitched, ref=np.max)
                librosa.display.specshow(mel_spec_pitched_db, sr=sr, hop_length=self.hop_length, 
                                       x_axis='time', y_axis='mel', ax=axes[0,2])
                axes[0,2].set_title('Pitch Shifted (+2 semitones)')
            except:
                axes[0,2].text(0.5, 0.5, 'Pitch Shift\nNot Available', ha='center', va='center')
                axes[0,2].set_title('Pitch Shifted (Failed)')
            
            # Noise addition
            noise = np.random.normal(0, 0.01, len(y_original))
            y_noisy = y_original + noise
            mel_spec_noisy = librosa.feature.melspectrogram(
                y=y_noisy, sr=sr, n_mels=self.n_mels, 
                n_fft=self.n_fft, hop_length=self.hop_length
            )
            mel_spec_noisy_db = librosa.power_to_db(mel_spec_noisy, ref=np.max)
            librosa.display.specshow(mel_spec_noisy_db, sr=sr, hop_length=self.hop_length, 
                                   x_axis='time', y_axis='mel', ax=axes[1,0])
            axes[1,0].set_title('Noise Added')
            
            # SpecAugment simulation (frequency masking)
            mel_spec_freq_masked = mel_spec_original_db.copy()
            freq_mask_start = np.random.randint(0, mel_spec_freq_masked.shape[0] - 10)
            mel_spec_freq_masked[freq_mask_start:freq_mask_start+10, :] = mel_spec_freq_masked.min()
            librosa.display.specshow(mel_spec_freq_masked, sr=sr, hop_length=self.hop_length, 
                                   x_axis='time', y_axis='mel', ax=axes[1,1])
            axes[1,1].set_title('Frequency Masked (SpecAugment)')
            
            # SpecAugment simulation (time masking)
            mel_spec_time_masked = mel_spec_original_db.copy()
            time_mask_start = np.random.randint(0, mel_spec_time_masked.shape[1] - 20)
            mel_spec_time_masked[:, time_mask_start:time_mask_start+20] = mel_spec_time_masked.min()
            librosa.display.specshow(mel_spec_time_masked, sr=sr, hop_length=self.hop_length, 
                                   x_axis='time', y_axis='mel', ax=axes[1,2])
            axes[1,2].set_title('Time Masked (SpecAugment)')
            
            plt.tight_layout()
            plt.show()
            
            print("Augmentation visualization completed")
            
        except Exception as e:
            print(f"Error visualizing augmentations: {str(e)}")
    
    def visualize_segment_predictions(self, audio_path, max_segments_to_show=10):
        """Visualize the prediction confidence for each segment"""
        print(f"Visualizing segment predictions for: {os.path.basename(audio_path)}")
        
        # Get segment predictions
        result = self.predict_genre_with_segments(audio_path, aggregation_method='average')
        if result is None:
            return
        
        genre, confidence, final_prediction, segment_predictions = result
        
        # Create visualization
        num_segments = min(len(segment_predictions), max_segments_to_show)
        segment_predictions_subset = segment_predictions[:num_segments]
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Confidence over segments
        plt.subplot(2, 2, 1)
        confidences = np.max(segment_predictions_subset, axis=1)
        predicted_classes = np.argmax(segment_predictions_subset, axis=1)
        colors = ['red' if pred == np.argmax(final_prediction) else 'blue' 
                 for pred in predicted_classes]
        
        plt.bar(range(num_segments), confidences, color=colors, alpha=0.7)
        plt.title(f'Segment Confidences\n(Red = Final Prediction: {genre})')
        plt.xlabel('Segment Number')
        plt.ylabel('Confidence')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Class predictions per segment
        plt.subplot(2, 2, 2)
        segment_classes = np.argmax(segment_predictions_subset, axis=1)
        class_names = [self.label_encoder.inverse_transform([cls])[0] for cls in segment_classes]
        
        unique_classes, counts = np.unique(segment_classes, return_counts=True)
        unique_names = [self.label_encoder.inverse_transform([cls])[0] for cls in unique_classes]
        
        plt.pie(counts, labels=unique_names, autopct='%1.1f%%', startangle=90)
        plt.title('Distribution of Segment Predictions')
        
        # Plot 3: Average probability distribution
        plt.subplot(2, 2, 3)
        genre_names = self.label_encoder.classes_
        plt.bar(genre_names, final_prediction, alpha=0.7)
        plt.title('Final Aggregated Probabilities')
        plt.xlabel('Genre')
        plt.ylabel('Probability')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Segment-by-segment heatmap
        plt.subplot(2, 2, 4)
        if num_segments <= 20:  # Only show heatmap for reasonable number of segments
            heatmap_data = segment_predictions_subset.T
            sns.heatmap(heatmap_data, 
                       xticklabels=[f'Seg{i+1}' for i in range(num_segments)],
                       yticklabels=genre_names,
                       annot=False, cmap='Blues', cbar=True)
            plt.title('Prediction Probabilities Heatmap')
            plt.xlabel('Segment')
            plt.ylabel('Genre')
        else:
            plt.text(0.5, 0.5, f'Too many segments ({len(segment_predictions)})\nto display heatmap', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Segment Heatmap (Too Many Segments)')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("Segment Analysis Summary:")
        print(f"  Total segments: {len(segment_predictions)}")
        print(f"  Final prediction: {genre} ({confidence:.4f})")
        print(f"  Average confidence: {np.mean(confidences):.4f}")
        print(f"  Most confident segment: Segment {np.argmax(confidences)+1} ({np.max(confidences):.4f})")
        print(f"  Least confident segment: Segment {np.argmin(confidences)+1} ({np.min(confidences):.4f})")

# Main execution
if __name__ == "__main__":
    start_time = time.time()
    print("Music Genre Classification Neural Network")
    print("Started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Initialize classifier with segmentation (default 4-second segments)
    classifier = MusicGenreClassifier(segment_length=4)
    
    # Create dataset from real audio files
    X, y = classifier.create_dataset()
    
    if X is None or y is None:
        print("Dataset creation failed. Exiting.")
        exit(1)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Genres: {classifier.genres}")
    
    # Train model with advanced augmentation
    print("Starting model training with audio augmentation and segmentation...")
    history = classifier.train_model(X, y, epochs=30, batch_size=32)
    
    # Save model
    classifier.save_model()
    
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time/60:.2f} minutes")
    print("Training completed! Model saved.")
    print("\nAdvanced Features Implemented:")
    print("  - Audio Segmentation (4-second segments)")
    print("  - Multiple Aggregation Strategies")
    print("  - SpecAugment (Time & Frequency Masking)")
    print("  - Librosa-based Audio Augmentation")
    print("  - Mixup with Adaptive Loss Functions")
    print("  - Multi-stage Augmentation Pipeline")
    print("  - Optimized tf.data Pipeline")
    print("\nUsage Examples:")
    print("  # Basic prediction with segmentation")
    print("  classifier.predict_genre('path/to/song.wav')")
    print("  ")
    print("  # Compare different aggregation methods")
    print("  classifier.compare_aggregation_methods('path/to/song.wav')")
    print("  ")
    print("  # Visualize segment predictions")
    print("  classifier.visualize_segment_predictions('path/to/song.wav')")
    print("  ")
    print("  # Use different segment length")
    print("  classifier = MusicGenreClassifier(segment_length=3)  # 3-second segments")
    print("\nYou can now use the Gradio interface to test the model.")
    print("The model now uses segment-based prediction for improved accuracy!")
