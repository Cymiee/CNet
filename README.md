# CNet — A Neural Network Framework in C

A deep learning framework built from scratch in pure C, with no external libraries.
N-dimensional tensors, automatic differentiation, and a full training loop — then
wrapped in a Python API so you can define and train a network in a few lines.

Trains a 2-layer MLP on MNIST to **97.7% test accuracy**, and ships a live drawing
app where you scribble a digit and watch the prediction update in real time.

Built to understand what sits beneath frameworks like PyTorch — memory layout,
gradient flow, and computation graphs implemented explicitly rather than hidden
behind `loss.backward()`.

> I wrote every line myself and used an AI assistant as a reviewer and rubber duck,
> not a code generator — every design decision here is one I can explain.

---

## Demo


```
Epoch 1:  loss 0.23   train acc 93.4%
Epoch 2:  loss 0.09   train acc 97.4%
Epoch 3:  loss 0.06   train acc 98.2%

Test set: 97.7%   (10,000 held-out images)
```

---

## What's inside

- **Tensor engine** — N-dimensional tensors, flat memory, stride-based indexing
- **Autograd** — a dynamically-built computation graph; `backward()` walks it in
  reverse via per-op function pointers (C has no closures, so the "context" each
  backward needs is captured by hand in a `BackwardContext` struct)
- **Gradient checking** — every backward function is verified numerically against
  finite differences; this is what makes the rest trustworthy
- **Operations** — add, sub, mul, matmul, ReLU, sigmoid, log, sum, with broadcasting
- **Training** — batched forward/backward, cross-entropy loss, **Adam** optimizer,
  He initialization, save/load weights
- **MNIST loader** — IDX binary parser in C (handles big-endian, normalizes to [0,1])
- **Python bindings** — a ctypes wrapper with a `Tensor` class and operator
  overloading, so training code reads almost like PyTorch
- **Inference** — predict from an image file, or draw live in a GUI canvas

## Status
- [x] Tensor engine — N-D tensors, stride arithmetic
- [x] Autograd — `BackwardContext`, backward for all ops, **gradient check passing**
- [x] Python bindings — ctypes wrapper, `Tensor` class, operator overloading
- [x] Batching + broadcasting + bias layers
- [x] MNIST training — 784→512→10 MLP, Adam, **97.7% test accuracy**
- [x] Save / load weights
- [x] Live drawing app + image prediction
- [ ] Convolutional layers (next)

---

## Build

```bash
# C test binary (forward-pass tests + gradient check)
gcc -Wall -Wextra -g -o tensor core/tensor.c core/mnist.c core/main.c -lm

# Python shared library — required before running any Python script
gcc -O2 -shared -fPIC -o python/libcnet.so core/tensor.c core/mnist.c -lm
```

Recompile the shared library any time you change the C source.

---

## Get the MNIST data

The original LeCun mirror is unreliable. Easiest path:

```bash
mkdir -p data && cd data
curl -O https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz
curl -O https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz
curl -O https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz
curl -O https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz
gunzip *.gz
cd ..
```

| File | Contents |
|------|----------|
| `train-images-idx3-ubyte` | 60,000 training images |
| `train-labels-idx1-ubyte` | 60,000 training labels |
| `t10k-images-idx3-ubyte`  | 10,000 test images    |
| `t10k-labels-idx1-ubyte`  | 10,000 test labels    |

---

## Train

```bash
python3 python/demo.py data/train-images-idx3-ubyte data/train-labels-idx1-ubyte --save weights.bin
```

Progress prints periodically:
```
  epoch 1 [ 5056/60000]  loss=0.6749  acc=0.813
  ...
Epoch 1: loss=0.2289  acc=0.934
```

## Evaluate on the test set

```bash
python3 python/demo.py data/t10k-images-idx3-ubyte data/t10k-labels-idx1-ubyte --load weights.bin
# Accuracy: 9770/10000 = 0.977
```

---

## Draw a digit (live)

Scribble with your mouse/trackpad and watch the prediction and per-digit
confidence update in real time:

```bash
pip install Pillow        # tkinter ships with Python on macOS
python3 python/draw.py weights.bin
```

## Predict from an image file

```bash
python3 python/predict.py weights.bin your_digit.png
```
```
Prediction: 3
Top 3:
  1. digit 3  (94.2%)
  2. digit 8  (4.1%)
  3. digit 5  (1.3%)
```

Light or dark background both work (auto-detected). The image is cropped to the
digit, padded, and resized to 28×28 to match MNIST's preprocessing — without that
step, hand-drawn input doesn't match what the network trained on.

---

## C tests

```bash
./tensor ops        # forward-pass tests
./tensor autograd   # gradient check + backward tests
./tensor mnist      # loader sanity check
```

---

## Python API

```python
from cnet import (Tensor, matmul, relu, zeros, randn, softmax,
                  cross_entropy_loss_batch, save_weights, load_weights,
                  Adam, MNISTData)

# Tensors — operator overloading for +, -, *, @
a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
w = randn((784, 512), requires_grad=True, scale=(2.0/784)**0.5)  # He init

# Forward
h      = relu(matmul(imgs, w1) + b1)
logits = matmul(h, w2) + b2
loss   = cross_entropy_loss_batch(logits, labels)   # sets logits.grad

# Backward + Adam
opt = Adam([w1, b1, w2, b2], lr=0.001)
logits.backward_grad_set()   # grad already set by the loss
opt.step()
opt.zero_grad()

# Weights
save_weights("weights.bin", [w1, b1, w2, b2])
load_weights("weights.bin", [w1, b1, w2, b2])

# MNIST
data = MNISTData("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte")
imgs, labels = data.get_batch_tensor(range(64))   # (64, 784) Tensor + labels
```

---

## Why C

The constraints are the point. No garbage collector means deciding who owns every
tensor in the graph. No closures means the backward pass for each op has to capture
its context by hand. No operator overloading on the C side means the math is fully
explicit. If you can implement backprop under those constraints — and prove it with
a gradient check — you understand it at a level that calling `loss.backward()` never
teaches.

The companion write-up goes deeper: (I will put a link to my blog here soon).