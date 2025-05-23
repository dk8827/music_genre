import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import os

from audio_processor import DEFAULT_SR # Assuming DEFAULT_SR is in config.py

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

def plot_final_results(history):
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

def plot_confusion_matrix_standalone(y_true, y_pred, label_encoder_classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder_classes,
                yticklabels=label_encoder_classes)
    plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.show()

def visualize_augmentations_standalone(audio_path, create_mel_spectrogram_func, create_tf_augmentation_layers_func, n_mels, n_fft, hop_length, sr=DEFAULT_SR):
    print(f"Visualizing augmentations for: {os.path.basename(audio_path)}")
    y_original, _ = librosa.load(audio_path, duration=10, sr=sr)

    vis_duration_sec = 5
    expected_frames_vis = int((vis_duration_sec * sr) / hop_length) + 1

    if len(y_original) < vis_duration_sec * sr:
            y_original = np.pad(y_original, (0, vis_duration_sec * sr - len(y_original)), mode='constant')
    y_original_segment = y_original[:vis_duration_sec * sr]

    mel_orig_norm = create_mel_spectrogram_func(y_original_segment, sr, expected_frames_vis)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Audio Augmentation Visualization: {os.path.basename(audio_path)}', fontsize=16)

    def plot_spec(ax, data, title):
        img = librosa.display.specshow(data, sr=sr, hop_length=hop_length,
                                        x_axis='time', y_axis='mel', ax=ax)
        ax.set_title(title)

    plot_spec(axes[0,0], librosa.power_to_db(librosa.feature.melspectrogram(y=y_original_segment, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)), 'Original (dB)')

    y_stretched = librosa.effects.time_stretch(y_original_segment, rate=0.8)
    plot_spec(axes[0,1], create_mel_spectrogram_func(y_stretched, sr, expected_frames_vis), 'Time Stretched (0.8x)')

    y_pitched = librosa.effects.pitch_shift(y_original_segment, sr=sr, n_steps=2)
    plot_spec(axes[0,2], create_mel_spectrogram_func(y_pitched, sr, expected_frames_vis), 'Pitch Shifted (+2 semitones)')

    noise = np.random.normal(0, 0.01, len(y_original_segment))
    y_noisy = y_original_segment + noise
    plot_spec(axes[1,0], create_mel_spectrogram_func(y_noisy, sr, expected_frames_vis), 'Noise Added')

    spec_tf_input = tf.constant(mel_orig_norm.reshape(1, *mel_orig_norm.shape, 1), dtype=tf.float32)

    _, spec_augment_fn, _, _ = create_tf_augmentation_layers_func() # Unpack only needed function

    spec_freq_masked = spec_augment_fn(spec_tf_input, freq_mask_param=20, time_mask_param=0, num_freq_masks=1, num_time_masks=0)
    plot_spec(axes[1,1], spec_freq_masked.numpy().squeeze(), 'Frequency Masked (SpecAugment)')

    spec_time_masked = spec_augment_fn(spec_tf_input, freq_mask_param=0, time_mask_param=40, num_freq_masks=0, num_time_masks=1)
    plot_spec(axes[1,2], spec_time_masked.numpy().squeeze(), 'Time Masked (SpecAugment)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def visualize_segment_predictions_standalone(segment_predictions, final_pred_probs, label_encoder_classes, genre, max_segments_to_show=10):
    num_segments_actual = segment_predictions.shape[0]
    num_segments_to_plot = min(num_segments_actual, max_segments_to_show)
    segment_preds_subset = segment_predictions[:num_segments_to_plot]

    plt.figure(figsize=(15, 10))
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
    unique_names_subset = [label_encoder_classes[cls_idx] for cls_idx in unique_classes_subset]
    ax2.pie(counts_subset, labels=unique_names_subset, autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'Distribution of Segment Predictions (Max {max_segments_to_show} Segments)')

    ax3 = plt.subplot(2, 2, 3)
    ax3.bar(label_encoder_classes, final_pred_probs, alpha=0.7)
    ax3.set_title('Final Aggregated Probabilities')
    ax3.set_xlabel('Genre'); ax3.set_ylabel('Probability')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 2, 4)
    if num_segments_to_plot <= 20:
        sns.heatmap(segment_preds_subset.T,
                    xticklabels=[f'Seg{i+1}' for i in range(num_segments_to_plot)],
                    yticklabels=label_encoder_classes, annot=False, cmap='Blues', cbar=True, ax=ax4)
        ax4.set_title('Prediction Probabilities Heatmap')
    else:
        ax4.text(0.5, 0.5, f'Too many segments ({num_segments_to_plot})\nto display heatmap',
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Segment Heatmap')
    ax4.set_xlabel('Segment'); ax4.set_ylabel('Genre')

    plt.tight_layout()
    plt.show() 