import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
from cnet import Tensor, add, sub, mul, matmul, log, sigmoid, relu, sum, zeros

inp = Tensor([[4.0, 1.0]])

w1 = zeros((2, 4), requires_grad=True)
w2 = zeros((4, 1), requires_grad=True)

for i in range(2):
    for j in range(4):
        w1.set((i, j), random.uniform(-1, 1))

for i in range(4):
    for j in range(1):
        w2.set((i, j), random.uniform(-1, 1))

print("w1: ")
w1.print()
print("h1 = input matmul w1: ")
h1 = matmul(inp, w1)
h1.print()
print("relu of h1: ")
h1_relu = relu(h1)
h1_relu.print()
print("h2 = h1_relu matmul w2")
h2 = matmul(h1_relu, w2)
h2.print()
print("loss: ")
loss = sum(h2)
loss.print()

loss.backward()
print("backward loss results: ")


for i in range(2):
    for j in range(4):
        print(w1.get_grad(i, j))

for i in range(4):
    for j in range(1):
        print(w2.get_grad(i, j))