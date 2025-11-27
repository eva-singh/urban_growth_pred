"""ConvLSTM model definition for image-sequence -> next-image prediction."""
import tensorflow as tf
from tensorflow.keras import layers, models

def build_convlstm_model(in_seq=5, h=256, w=256, filters=32, kernel_size=(3,3)):
    # Input shape: (in_seq, H, W, 1)
    inputs = layers.Input(shape=(in_seq, h, w, 1))
    x = layers.ConvLSTM2D(filters=filters, kernel_size=kernel_size, padding='same', return_sequences=True, activation='tanh')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(filters=filters//2, kernel_size=kernel_size, padding='same', return_sequences=False, activation='tanh')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid', padding='same')(x)  # predict single-channel probability map
    model = models.Model(inputs, x)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    m = build_convlstm_model(in_seq=5, h=64, w=64)
    m.summary()
