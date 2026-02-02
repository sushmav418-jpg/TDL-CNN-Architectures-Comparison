from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def build_alexnet(input_shape):
    model = Sequential([
        Conv2D(96, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(),

        Conv2D(256, (3,3), activation='relu'),
        MaxPooling2D(),

        Conv2D(384, (3,3), activation='relu'),
        Conv2D(384, (3,3), activation='relu'),
        Conv2D(256, (3,3), activation='relu'),
        MaxPooling2D(),

        Flatten(),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model
