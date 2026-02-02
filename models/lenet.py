from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense

def build_lenet(input_shape):
    model = Sequential([
        Conv2D(6, kernel_size=5, activation='relu', input_shape=input_shape),
        AveragePooling2D(),

        Conv2D(16, kernel_size=5, activation='relu'),
        AveragePooling2D(),

        Flatten(),
        Dense(120, activation='relu'),
        Dense(84, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model
