import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from plotting_utils import PlotCallback, plot_final_results, plot_confusion_matrix_standalone
from augmentation_layers import create_data_augmentation_pipeline
from model_builder import build_model

class ModelTrainer:
    """
    Handles the model training process, including data preparation, augmentation, and evaluation.
    """
    def __init__(self, label_encoder):
        """
        Initializes the ModelTrainer.

        Args:
            label_encoder (sklearn.preprocessing.LabelEncoder): A LabelEncoder instance,
                                                                 which will be fitted during training.
        """
        self.model = None
        self.train_mean = None
        self.train_std = None
        self.label_encoder = label_encoder

    def _prepare_tf_datasets(self, X_train, y_train_encoded, X_val, y_val_encoded,
                               batch_size, num_classes, data_augmentation_pipeline, mixup_fn):
        """
        Prepares TensorFlow Dataset objects for training and validation.

        This includes one-hot encoding labels, batching, applying data augmentation
        (via the provided pipeline), and applying Mixup randomly.

        Args:
            X_train, X_val (np.ndarray): Training and validation features.
            y_train_encoded, y_val_encoded (np.ndarray): Encoded training and validation labels.
            batch_size (int): Batch size for the datasets.
            num_classes (int): Total number of unique classes.
            data_augmentation_pipeline (keras.Sequential): Keras pipeline for augmentation.
            mixup_fn (function): Function to apply Mixup augmentation.

        Returns:
            tuple: (train_dataset, val_dataset, y_train_onehot, y_val_onehot)
                   TensorFlow datasets and one-hot encoded labels.
        """

        # One-hot encode labels for categorical crossentropy (especially with Mixup)
        y_train_onehot = tf.keras.utils.to_categorical(y_train_encoded, num_classes)
        y_val_onehot = tf.keras.utils.to_categorical(y_val_encoded, num_classes)

        def apply_mixup_randomly(batch_x, batch_y):
            """Applies Mixup to a batch with a certain probability."""
            return tf.cond(
                tf.random.uniform([]) < 0.25, # Apply Mixup to 25% of batches
                lambda: mixup_fn(batch_x, batch_y, alpha=0.2),
                lambda: (batch_x, tf.cast(batch_y, tf.float32))
            )

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_onehot))
        train_dataset = train_dataset.shuffle(buffer_size=len(X_train)) # Shuffle training data
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.map(
            lambda x, y: (data_augmentation_pipeline(x, training=True), y), # Apply augmentations
            num_parallel_calls=tf.data.AUTOTUNE
        )
        train_dataset = train_dataset.map(apply_mixup_randomly, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val_onehot))
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset, y_train_onehot, y_val_onehot

    def train_model(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """
        Trains and evaluates the music genre classification model.

        Steps include:
        1. Encoding labels.
        2. Splitting data into training and validation sets.
        3. Normalizing features (spectrograms) based on training set statistics.
        4. Building the Keras model.
        5. Preparing TensorFlow Datasets with augmentations and Mixup.
        6. Training the model with callbacks (EarlyStopping, ReduceLROnPlateau).
        7. Evaluating the model and printing classification reports.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Input labels (genre names).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size.
            validation_split (float): Fraction of data for validation.

        Returns:
            tuple: (history, model, train_mean, train_std)
                - Keras training history.
                - Trained Keras model.
                - Mean used for training data normalization.
                - Standard deviation used for training data normalization.
        """
        # Encode string labels to integers
        y_encoded = self.label_encoder.fit_transform(y)
        num_classes = len(self.label_encoder.classes_)

        # Split data into training and validation sets, stratified by class
        X_train, X_val, y_train_enc, y_val_enc = train_test_split(
            X, y_encoded, test_size=validation_split, stratify=y_encoded,
            random_state=42, shuffle=True
        )

        # Calculate mean and std for normalization from the training set ONLY
        self.train_mean = np.mean(X_train, axis=(0,1,2), keepdims=True)
        self.train_std = np.std(X_train, axis=(0,1,2), keepdims=True)

        if self.train_std < 1e-6: self.train_std = 1.0 # Avoid division by zero if std is too small

        # Normalize training and validation data
        X_train_norm = (X_train - self.train_mean) / self.train_std
        X_val_norm = (X_val - self.train_mean) / self.train_std

        print(f"Training set: {X_train_norm.shape}, Validation set: {X_val_norm.shape}")

        data_augmentation_pipeline, mixup_fn = create_data_augmentation_pipeline()

        # Build the CNN model
        self.model = build_model(X_train_norm.shape[1:], num_classes, use_mixup=True)
        print(f"Model built for {num_classes} classes. Input shape: {X_train_norm.shape[1:]}")

        # Prepare tf.data.Dataset objects for efficient training
        train_dataset, val_dataset, y_train_onehot, y_val_onehot = self._prepare_tf_datasets(
            X_train_norm, y_train_enc, X_val_norm, y_val_enc, batch_size, num_classes,
            data_augmentation_pipeline, mixup_fn
        )

        # Define callbacks for training
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-6, verbose=1),
            # PlotCallback() # This is from plotting_utils - uncomment to plot during training
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
        # Use normalized data for evaluation
        train_loss, train_acc = self.model.evaluate(X_train_norm, y_train_onehot, verbose=0)
        val_loss, val_acc = self.model.evaluate(X_val_norm, y_val_onehot, verbose=0)

        print(f"Final Training Accuracy: {train_acc:.4f}, Loss: {train_loss:.4f}")
        print(f"Final Validation Accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}")

        # plot_final_results(history) # This is from plotting_utils

        y_pred_probs = self.model.predict(X_val_norm)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)

        print("Classification Report:")
        print(classification_report(y_val_enc, y_pred_classes, target_names=self.label_encoder.classes_))
        # plot_confusion_matrix_standalone(y_val_enc, y_pred_classes, self.label_encoder.classes_) # This is from plotting_utils

        return history, self.model, self.train_mean, self.train_std 