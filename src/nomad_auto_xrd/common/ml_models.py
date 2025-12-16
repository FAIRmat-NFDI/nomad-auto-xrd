from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Flatten,
    Layer,
    MaxPooling1D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.nn import dropout


class CustomDropout(Layer):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def get_config(self):
        config = super().get_config()
        config.update({'rate': self.rate})
        return config

    # Always apply dropout
    def call(self, inputs, training=None):
        return dropout(inputs, rate=self.rate)


def build_model(
    input_shape,
    n_phases,
    is_pdf,
    learning_rate=0.001,
):
    """
    Builds the CNN model based on whether it is for PDF or XRD analysis.
    """
    layers = []
    if is_pdf:
        # Architecture for PDF analysis
        layers.extend(
            [
                Conv1D(
                    64, 60, activation='relu', padding='same', input_shape=input_shape
                ),
                MaxPooling1D(3, strides=2, padding='same'),
                MaxPooling1D(3, strides=2, padding='same'),
                MaxPooling1D(2, strides=2, padding='same'),
                MaxPooling1D(1, strides=2, padding='same'),
                MaxPooling1D(1, strides=2, padding='same'),
                MaxPooling1D(1, strides=2, padding='same'),
            ]
        )
    else:
        # Architecture for XRD analysis
        layers.extend(
            [
                Conv1D(
                    64, 35, activation='relu', padding='same', input_shape=input_shape
                ),
                MaxPooling1D(3, strides=2, padding='same'),
                Conv1D(64, 30, activation='relu', padding='same'),
                MaxPooling1D(3, strides=2, padding='same'),
                Conv1D(64, 25, activation='relu', padding='same'),
                MaxPooling1D(2, strides=2, padding='same'),
                Conv1D(64, 20, activation='relu', padding='same'),
                MaxPooling1D(1, strides=2, padding='same'),
                Conv1D(64, 15, activation='relu', padding='same'),
                MaxPooling1D(1, strides=2, padding='same'),
                Conv1D(64, 10, activation='relu', padding='same'),
                MaxPooling1D(1, strides=2, padding='same'),
            ]
        )

    # Common layers
    n_dense = [3100, 1200]
    dropout_rate = 0.7
    layers.extend(
        [
            Flatten(),
            CustomDropout(dropout_rate),
            Dense(n_dense[0], activation='relu'),
            BatchNormalization(),
            CustomDropout(dropout_rate),
            Dense(n_dense[1], activation='relu'),
            BatchNormalization(),
            CustomDropout(dropout_rate),
            Dense(n_phases, activation='softmax'),
        ]
    )
    optimizer = Adam(learning_rate=learning_rate)

    model = Sequential(layers)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['categorical_accuracy'],
    )
    return model
