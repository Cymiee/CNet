# CNet — Neural Network Framework in C

A deep learning framework built from scratch in pure C,
with no external libraries. Implements tensors with N-dimensional
stride-based indexing, automatic differentiation via a computation
graph, and a training loop targeting MNIST classification.

Built as a systems + ML portfolio project to understand what sits
beneath frameworks like PyTorch — memory layout, gradient flow,
and computation graphs implemented explicitly.

## Architecture
- **Tensor engine** — N-D tensors, flat memory, stride arithmetic
- **Autograd** — DAG-based computation graph, `backward()` via function pointers
- **Python bindings** — ctypes interface to the C core (`python/cnet.py`)
- **MNIST loader** — IDX binary parser, normalizes pixels to [0, 1]
- **Training** — SGD, cross-entropy loss, save/load weights

## Status
- [x] Tensor engine — N-D tensors, stride arithmetic, 9 ops
- [x] Autograd — BackwardContext, backward() for all ops, gradient check passing
- [x] Python bindings — ctypes wrapper, Tensor class, operator overloading
- [x] MNIST training — 2-layer MLP (784→128→10), ~97% train accuracy
- [x] Save/load weights
- [x] Digit prediction from image
- [ ] CNNs

---

## Build

```bash
# C test binary
gcc -Wall -Wextra -g -o tensor core/tensor.c core/main.c -lm

# Python shared library (required before running any Python scripts)
gcc -shared -fPIC -o python/libcnet.so core/tensor.c core/mnist.c -lm
```

---

## Run C Tests

```bash
./tensor ops        # forward pass tests
./tensor autograd   # gradient check + backward tests
```

---

## MNIST Dataset

Download the 4 IDX binary files from [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist)
or mirror at [ossci-datasets.s3.amazonaws.com](https://ossci-datasets.s3.amazonaws.com/mnist/):

| File | Contents |
|------|----------|
| `train-images-idx3-ubyte` | 60,000 training images |
| `train-labels-idx1-ubyte` | 60,000 training labels |
| `t10k-images-idx3-ubyte`  | 10,000 test images    |
| `t10k-labels-idx1-ubyte`  | 10,000 test labels    |

---

## Training

```bash
# Train for 3 epochs and save weights
python3 python/demo.py train-images-idx3-ubyte train-labels-idx1-ubyte --save weights.bin

# Train without saving
python3 python/demo.py train-images-idx3-ubyte train-labels-idx1-ubyte
```

Progress is printed every 5,000 steps:
```
  epoch 1 [ 5000/60000]  loss=0.8312  acc=0.764
  epoch 1 [10000/60000]  loss=0.6201  acc=0.823
  ...
Epoch 1: loss=0.3744  acc=0.893
```

---

## Evaluation

```bash
# Evaluate saved weights on the test set
python3 python/demo.py t10k-images-idx3-ubyte t10k-labels-idx1-ubyte --load weights.bin
```

---

## Predict from Image

Requires [Pillow](https://pillow.readthedocs.io):
```bash
pip install Pillow
```

Draw a digit (white or dark background both work — auto-detected) and run:
```bash
python3 python/predict.py weights.bin your_digit.png
```

Output:
```
Prediction: 3
Top 3:
  1. digit 3  (94.2%)
  2. digit 8  (4.1%)
  3. digit 5  (1.3%)
```

**Tips for best accuracy:**
- Draw on a ~200×200+ canvas, centered, thick strokes
- Save as PNG
- The image is auto-resized to 28×28 internally

---

## Python API

```python
from cnet import Tensor, matmul, relu, zeros, randn, softmax
from cnet import cross_entropy_loss, save_weights, load_weights, MNISTData

# Create tensors
a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = zeros((2, 2), requires_grad=True)
c = randn((784, 128), requires_grad=True, scale=0.01)

# Ops (operator overloading supported: +, -, *, @)
h = relu(a @ b)
loss = cross_entropy_loss(h, label=0)

# Autograd
h.backward_grad_set()   # when grad is pre-set (e.g. after cross_entropy_loss)
h.backward()            # standard backward from scalar loss

# SGD
w.sgd_step(lr=0.01)
w.zero_grad()

# Weights
save_weights("weights.bin", [w1, w2])
load_weights("weights.bin", [w1, w2])

# MNIST loader
data = MNISTData("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
img  = data.get_image_tensor(0)   # Tensor, shape (1, 784)
lbl  = data.get_label(0)          # int 0-9
```
