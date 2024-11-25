import tensorflow as tf
from typing import Tuple

class TensorFlowModel:
    def __init__(self, input_shape: Tuple[int], num_classes: int):
        self.model = self._build_model(input_shape, num_classes)

    def _build_model(self, input_shape: Tuple[int], num_classes: int):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size
        )

    def save(self, filepath: str):
        """
        Save the model to the specified filepath in H5 format.
        """
        self.model.save(f"{filepath}.h5")
