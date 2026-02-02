import time
import pandas as pd
import tensorflow as tf

from models.lenet import build_lenet
from models.alexnet import build_alexnet
from models.resnet import build_resnet
from utils.preprocess import load_cifar10

# --------------------------------------------------
# 1. Load CIFAR-10 dataset
# --------------------------------------------------
x_train, y_train, x_test, y_test = load_cifar10()

# CIFAR-10 input shape (IMPORTANT)
input_shape = (32, 32, 3)

# --------------------------------------------------
# 2. Initialize models
# --------------------------------------------------
models = {
    "LeNet": build_lenet(input_shape),
    "AlexNet": build_alexnet(input_shape),
    "ResNet": build_resnet(input_shape)
}

results = []

# --------------------------------------------------
# 3. Train each model
# --------------------------------------------------
for model_name, model in models.items():

    print(f"\nTraining {model_name} on CIFAR-10")

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    start_time = time.time()

    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=5,              # 5 or 10 as per assignment
        batch_size=64,
        verbose=1
    )

    end_time = time.time()

    training_time = end_time - start_time
    total_params = model.count_params()
    final_val_accuracy = history.history['val_accuracy'][-1]

    # Store results
    results.append([
        model_name,
        "CIFAR-10",
        round(final_val_accuracy * 100, 2),
        total_params,
        round(training_time, 2)
    ])

# --------------------------------------------------
# 4. Save results to CSV
# --------------------------------------------------
df = pd.DataFrame(
    results,
    columns=["Model", "Dataset", "Validation Accuracy (%)", "Parameters", "Training Time (s)"]
)

df.to_csv("results/cifar10_results.csv", index=False)

print("\nTraining completed. Results saved to results/cifar10_results.csv")
print(df)
