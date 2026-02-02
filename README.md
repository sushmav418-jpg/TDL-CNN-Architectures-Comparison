# TDL-CNN-Architectures-Comparison
Comparison of LeNet, AlexNet and ResNet on MNIST and CIFAR-10
#  CNN Architecture Comparison: LeNet, AlexNet & ResNet

This project implements and compares three popular Convolutional Neural Network (CNN) architectures — **LeNet**, **AlexNet**, and **ResNet** — on two benchmark image datasets: **MNIST** and **CIFAR-10**.

The goal is to analyze how architectural complexity impacts **accuracy**, **number of parameters**, and **training time**.



##  Datasets Used

### 1️ MNIST
- Grayscale images (28×28)
- Handwritten digits (0–9)
- Simple dataset with low visual complexity

###  CIFAR-10
- RGB images (32×32×3)
- 10 object classes (airplane, car, bird, etc.)
- Higher visual and feature complexity

---

## CNN Architectures Implemented

### 🔹 LeNet
- Shallow CNN architecture
- Fewer parameters
- Fast training
- Suitable for simple datasets like MNIST

### 🔹 AlexNet
- Deeper architecture with more filters
- Large fully connected layers
- High number of parameters
- Performs well on complex datasets

### 🔹 ResNet
- Uses residual (skip) connections
- Helps reduce vanishing gradient problem
- Parameter-efficient compared to depth

---

##  Project Folder Structure
```
TDL_CNN_Assignment/
│
├── models/
│ ├── lenet.py
│ ├── alexnet.py
│ └── resnet.py
│
├── train/
│ ├── train_mnist.py
│ └── train_cifar10.py
│
├── utils/
│ └── preprocess.py
│
├── results/
│ └── cifar10_results.csv
│
├── requirements.txt
└── README.md


---

##  How to Run the Project

### 🔹 Install dependencies
```bash
pip install tensorflow pandas numpy matplotlib

🔹 Train models on MNIST
python -m train.train_mnist

🔹 Train models on CIFAR-10
python -m train.train_cifar10

Experimental Results
🔸 Accuracy, Parameters & Training Time
Model	Dataset	Accuracy (%)	Parameters	Training Time (s)
LeNet	MNIST	98.68	44,426	22.14
AlexNet	MNIST	99.07	2,273,802	52.52
ResNet	MNIST	99.07	224,394	119.25
LeNet	CIFAR-10	55.03	62,006	22.19
AlexNet	CIFAR-10	75.33	3,192,458	53.33
ResNet	CIFAR-10	41.53	225,546	138.37
 Key Observations

MNIST achieves high accuracy for all models due to its simplicity.

AlexNet performs best on CIFAR-10 as it can capture complex RGB features.

ResNet requires more training epochs to converge effectively on CIFAR-10.

Increasing parameters improves learning capacity but increases computation cost.

 Trade-offs Identified

LeNet: Fast and efficient, but limited accuracy on complex datasets.

AlexNet: Best balance between accuracy and performance on CIFAR-10.

ResNet: Powerful architecture, but computationally expensive.

 Technologies Used

Python

TensorFlow / Keras

NumPy

Pandas

Matplotlib

Git & GitHub

 Author

Sushma Venkatesh
B.Tech CSE, PES University
Deep Learning / CNN Assignment

 License

This project is for academic and educational purposes.


---

##  FINAL STEPS

After saving `README.md`, run:

```bash
git add README.md
git commit -m "Added detailed project README"
git push

```


