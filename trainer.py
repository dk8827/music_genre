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
    def __init__(self, label_encoder):
        self.model = None
        self.train_mean = None
        self.train_std = None
        self.label_encoder = label_encoder

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

        X_train_norm = (X_train - self.train_mean) / self.train_std
        X_val_norm = (X_val - self.train_mean) / self.train_std

        print(f"Training set: {X_train_norm.shape}, Validation set: {X_val_norm.shape}")

        data_augmentation_pipeline, mixup_fn = create_data_augmentation_pipeline()

        self.model = build_model(X_train_norm.shape[1:], num_classes, use_mixup=True)
        print(f"Model built for {num_classes} classes. Input shape: {X_train_norm.shape[1:]}")

        train_dataset, val_dataset, y_train_onehot, y_val_onehot = self._prepare_tf_datasets(
            X_train_norm, y_train_enc, X_val_norm, y_val_enc, batch_size, num_classes,
            data_augmentation_pipeline, mixup_fn
        )

        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-6, verbose=1),
            # PlotCallback() # This is from plotting_utils
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