import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def _create_tf_augmentation_layers():
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

def create_data_augmentation_pipeline():
    add_noise_fn, spec_augment_fn, random_gain_fn, mixup_fn = _create_tf_augmentation_layers()

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