import time
from models.lenet import build_lenet
from models.alexnet import build_alexnet
from models.resnet import build_resnet
from utils.preprocess import load_mnist

models = {
    "LeNet": build_lenet((28,28,1)),
    "AlexNet": build_alexnet((28,28,1)),
    "ResNet": build_resnet((28,28,1))
}

x_train, y_train, x_test, y_test = load_mnist()

for name, model in models.items():
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    start = time.time()
    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=5,
                        batch_size=64)
    end = time.time()

    print(f"{name} | Params: {model.count_params()} | Time: {end-start:.2f}s")
