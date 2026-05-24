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
- **Autograd** — DAG-based computation graph, backward() via function pointers
- **Python bindings** — ctypes interface to the C core
- **Training** — SGD/Adam in C, MLP/ConvNet in Python (upcoming)

## Status
- [x] Tensor engine — N-D tensors, stride arithmetic, 9 ops
- [x] Autograd — BackwardContext, backward() for all ops, gradient check passing
- [x] Python bindings — ctypes wrapper, Tensor class, operator overloading
- [ ] MNIST training
- [ ] CNNs

## Build
```bash
# C test binary
gcc -Wall -Wextra -g -o tensor core/tensor.c core/main.c -lm

# Python shared library
gcc -shared -fPIC -o python/libcnet.so core/tensor.c -lm
```

## Run Tests
```bash
./tensor ops        # forward pass tests
./tensor autograd   # gradient check + backward tests
python3 python/cnet.py  # Python wrapper tests
```