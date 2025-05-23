import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def _create_tf_augmentation_layers():
    """
    Creates and returns TensorFlow functions for various spectrogram augmentations.

    These functions are designed to be used within a TensorFlow graph (e.g., tf.data.Dataset.map).
    The returned functions are: add_noise_tf, spec_augment_tf, random_gain_tf, mixup_batch_tf.
    """
    @tf.function
    def add_noise_tf(spectrogram, noise_factor=0.005):
        """Adds random Gaussian noise to a spectrogram."""
        noise = tf.random.normal(tf.shape(spectrogram), stddev=noise_factor)
        return spectrogram + noise

    @tf.function
    def spec_augment_tf(spectrogram, freq_mask_param=15, time_mask_param=35, num_freq_masks=1, num_time_masks=1):
        """Applies SpecAugment (frequency and time masking) to a batch of spectrograms."""
        spec_shape = tf.shape(spectrogram)
        num_mel_bins = spec_shape[1]
        num_time_steps = spec_shape[2]

        for _ in range(num_freq_masks):
            f = tf.random.uniform([], 0, freq_mask_param, dtype=tf.int32)
            f = tf.minimum(f, num_mel_bins)
            f0 = tf.random.uniform([], 0, num_mel_bins - f, dtype=tf.int32)
            mask = tf.concat([
                tf.ones((spec_shape[0], f0, num_time_steps, spec_shape[3])),
                tf.zeros((spec_shape[0], f, num_time_steps, spec_shape[3])),
                tf.ones((spec_shape[0], num_mel_bins - f0 - f, num_time_steps, spec_shape[3]))
            ], axis=1)
            spectrogram = spectrogram * mask

        for _ in range(num_time_masks):
            t = tf.random.uniform([], 0, time_mask_param, dtype=tf.int32)
            t = tf.minimum(t, num_time_steps)
            t0 = tf.random.uniform([], 0, num_time_steps - t, dtype=tf.int32)
            mask = tf.concat([
                tf.ones((spec_shape[0], num_mel_bins, t0, spec_shape[3])),
                tf.zeros((spec_shape[0], num_mel_bins, t, spec_shape[3])),
                tf.ones((spec_shape[0], num_mel_bins, num_time_steps - t0 - t, spec_shape[3]))
            ], axis=2)
            spectrogram = spectrogram * mask
        return spectrogram

    @tf.function
    def random_gain_tf(spectrogram, min_gain_db=-6, max_gain_db=6):
        """Applies a random gain (in dB) to a spectrogram."""
        gain_db = tf.random.uniform([], min_gain_db, max_gain_db)
        gain_linear = tf.pow(10.0, gain_db / 20.0)
        return spectrogram * gain_linear

    @tf.function
    def mixup_batch_tf(batch_x, batch_y, alpha=0.2):
        """
        Applies Mixup augmentation to a batch of data (spectrograms and labels).

        Mixup linearly interpolates between pairs of examples and their labels.

        Args:
            batch_x: Input features (e.g., spectrograms).
            batch_y: Corresponding labels (expected to be one-hot encoded or castable to float).
            alpha (float): Mixup hyperparameter controlling the interpolation strength.

        Returns:
            tuple: Mixed features and mixed labels.
        """
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

def create_data_augmentation_pipeline():
    """
    Creates a Keras Sequential model for spectrogram augmentation and a Mixup function.

    The pipeline includes random translation, random zoom, and a custom layer
    applying noise, SpecAugment, and random gain with specified probabilities.

    Returns:
        tuple: (keras.Sequential, function)
            - The Keras Sequential model for data augmentation.
            - The Mixup function (mixup_batch_tf).
    """
    add_noise_fn, spec_augment_fn, random_gain_fn, mixup_fn = _create_tf_augmentation_layers()

    class AudioAugmentationLayer(layers.Layer):
        """
        Custom Keras layer that applies a sequence of audio augmentations.

        Includes: add_noise, spec_augment, random_gain.
        These are applied stochastically during training.
        """
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.add_noise = add_noise_fn
            self.spec_augment = spec_augment_fn
            self.random_gain = random_gain_fn

        def call(self, inputs, training=None):
            if not training:
                return inputs

            x = inputs

            # Apply noise with 50% probability
            x = tf.cond(
                tf.random.uniform([]) < 0.5,
                lambda: self.add_noise(x, noise_factor=0.01),
                lambda: x
            )
            # Apply SpecAugment with 80% probability
            x = tf.cond(
                tf.random.uniform([]) < 0.8,
                lambda: self.spec_augment(x, freq_mask_param=10, time_mask_param=20),
                lambda: x
            )
            # Apply random gain with 30% probability
            x = tf.cond(
                tf.random.uniform([]) < 0.3,
                lambda: self.random_gain(x, min_gain_db=-3, max_gain_db=3),
                lambda: x
            )
            return x

    data_augmentation_pipeline = keras.Sequential([
        layers.RandomTranslation(height_factor=0.03, width_factor=0.03), # Small random shifts
        layers.RandomZoom(0.03), # Small random zoom
        AudioAugmentationLayer(),
    ], name="audio_augmentation_pipeline")

    return data_augmentation_pipeline, mixup_fn 