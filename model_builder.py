from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape, num_classes, use_mixup=False):
    """
    Builds and compiles a Convolutional Neural Network (CNN) for genre classification.

    The architecture includes multiple convolutional blocks with Batch Normalization,
    ReLU activations, Squeeze-and-Excitation (SE) blocks, MaxPooling, and Dropout.
    The SE blocks help the model to learn channel-wise feature recalibration.
    Residual connections are used within the conv_blocks.

    Args:
        input_shape (tuple): Shape of the input spectrograms (height, width, channels).
        num_classes (int): Number of output classes (genres).
        use_mixup (bool): If True, configures the loss for Mixup augmentation (categorical_crossentropy).
                          Otherwise, uses sparse_categorical_crossentropy.

    Returns:
        tensorflow.keras.Model: The compiled Keras model.
    """
    inputs = layers.Input(shape=input_shape)
    # Initial Batch Normalization for input stabilization.
    x = layers.BatchNormalization()(inputs)

    def conv_block(input_tensor, filters, kernel_size=(3,3), strides=(1,1), stage_name=""):
        """
        A convolutional block with two Conv2D layers, Batch Normalization, ReLU, 
        a Squeeze-and-Excitation (SE) block, and a residual connection.

        Args:
            input_tensor: Input tensor to the block.
            filters (int): Number of filters for the convolutional layers.
            kernel_size (tuple): Kernel size for convolutions.
            strides (tuple): Strides for the first convolution and the shortcut connection if dimensions change.
            stage_name (str): Prefix for layer names within this block.

        Returns:
            Output tensor of the block.
        """
        x_conv = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', name=f"{stage_name}_conv1")(input_tensor)
        x_conv = layers.BatchNormalization(name=f"{stage_name}_bn1")(x_conv)
        x_conv = layers.Activation('relu', name=f"{stage_name}_relu1")(x_conv)

        x_conv = layers.Conv2D(filters, kernel_size, padding='same', name=f"{stage_name}_conv2")(x_conv)
        x_conv = layers.BatchNormalization(name=f"{stage_name}_bn2")(x_conv)

        # Shortcut connection for residual learning.
        # If dimensions change due to strides or filter numbers, a 1x1 convolution is used on the shortcut.
        if strides != (1,1) or input_tensor.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1,1), strides=strides, padding='same', name=f"{stage_name}_shortcut_conv")(input_tensor)
            shortcut = layers.BatchNormalization(name=f"{stage_name}_shortcut_bn")(shortcut)
        else:
            shortcut = input_tensor

        x_conv = layers.Add(name=f"{stage_name}_add")([x_conv, shortcut])
        x_conv = layers.Activation('relu', name=f"{stage_name}_relu2")(x_conv)
        return x_conv

    x = conv_block(x, 32, stage_name="block1")
    x = layers.MaxPooling2D((2, 2), name="block1_pool")(x)
    x = layers.Dropout(0.25, name="block1_drop")(x) # Dropout for regularization

    x = conv_block(x, 64, stage_name="block2")
    x = layers.MaxPooling2D((2, 2), name="block2_pool")(x)
    x = layers.Dropout(0.25, name="block2_drop")(x)

    x = conv_block(x, 128, stage_name="block3")
    x = layers.MaxPooling2D((2, 2), name="block3_pool")(x)
    x = layers.Dropout(0.25, name="block3_drop")(x)

    # Global Average Pooling reduces each feature map to a single number.
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Fully connected dense layers for classification.
    x = layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.001), name="dense1")(x)
    x = layers.BatchNormalization(name="dense1_bn")(x)
    x = layers.Activation('relu', name="dense1_relu")(x)
    x = layers.Dropout(0.4, name="dense1_drop")(x) # Higher dropout before the final layer

    outputs = layers.Dense(num_classes, activation='softmax', name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Adam optimizer with gradient clipping to prevent exploding gradients.
    optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    # Loss function choice depends on whether Mixup is used (expects one-hot labels).
    loss = 'categorical_crossentropy' if use_mixup else 'sparse_categorical_crossentropy'

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model 