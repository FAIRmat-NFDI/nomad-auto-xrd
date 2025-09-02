import tensorflow as tf


class CustomDropout(tf.keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def get_config(self):
        config = super().get_config()
        config.update({'rate': self.rate})
        return config

    # Always apply dropout
    def call(self, inputs, training=None):
        return tf.nn.dropout(inputs, rate=self.rate)


def build_model(input_shape, n_phases, is_pdf, n_dense=[3100, 1200], dropout_rate=0.7):
    """
    Builds the CNN model based on whether it is for PDF or XRD analysis.
    """
    layers = []
    if is_pdf:
        # Architecture for PDF analysis
        layers.extend(
            [
                tf.keras.layers.Conv1D(
                    64, 60, activation='relu', padding='same', input_shape=input_shape
                ),
                tf.keras.layers.MaxPooling1D(3, strides=2, padding='same'),
                tf.keras.layers.MaxPooling1D(3, strides=2, padding='same'),
                tf.keras.layers.MaxPooling1D(2, strides=2, padding='same'),
                tf.keras.layers.MaxPooling1D(1, strides=2, padding='same'),
                tf.keras.layers.MaxPooling1D(1, strides=2, padding='same'),
                tf.keras.layers.MaxPooling1D(1, strides=2, padding='same'),
            ]
        )
    else:
        # Architecture for XRD analysis
        layers.extend(
            [
                tf.keras.layers.Conv1D(
                    64, 35, activation='relu', padding='same', input_shape=input_shape
                ),
                tf.keras.layers.MaxPooling1D(3, strides=2, padding='same'),
                tf.keras.layers.Conv1D(64, 30, activation='relu', padding='same'),
                tf.keras.layers.MaxPooling1D(3, strides=2, padding='same'),
                tf.keras.layers.Conv1D(64, 25, activation='relu', padding='same'),
                tf.keras.layers.MaxPooling1D(2, strides=2, padding='same'),
                tf.keras.layers.Conv1D(64, 20, activation='relu', padding='same'),
                tf.keras.layers.MaxPooling1D(1, strides=2, padding='same'),
                tf.keras.layers.Conv1D(64, 15, activation='relu', padding='same'),
                tf.keras.layers.MaxPooling1D(1, strides=2, padding='same'),
                tf.keras.layers.Conv1D(64, 10, activation='relu', padding='same'),
                tf.keras.layers.MaxPooling1D(1, strides=2, padding='same'),
            ]
        )

    # Common layers
    layers.extend(
        [
            tf.keras.layers.Flatten(),
            CustomDropout(dropout_rate),
            tf.keras.layers.Dense(n_dense[0], activation='relu'),
            tf.keras.layers.BatchNormalization(),
            CustomDropout(dropout_rate),
            tf.keras.layers.Dense(n_dense[1], activation='relu'),
            tf.keras.layers.BatchNormalization(),
            CustomDropout(dropout_rate),
            tf.keras.layers.Dense(n_phases, activation='softmax'),
        ]
    )

    model = tf.keras.Sequential(layers)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['categorical_accuracy'],
    )
    return model
