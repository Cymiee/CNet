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
- **Training** — SGD/Adam in C, MLP/ConvNet in Python

## Status
- [x] Tensor engine
- [ ] Autograd (in progress)
- [ ] Python bindings
- [ ] MNIST training

## Build
gcc -Wall -Wextra -g -o tensor tensor.c main.c -lm
