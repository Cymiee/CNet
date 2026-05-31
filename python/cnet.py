import builtins
import ctypes
import math
import os
import random
import struct

_libdir = os.path.dirname(os.path.abspath(__file__))
cnet = ctypes.CDLL(os.path.join(_libdir, 'libcnet.so'))

# tensor operations
cnet.tensor_create.restype  = ctypes.c_void_p
cnet.tensor_create.argtypes = [
    ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]

cnet.tensor_free.restype = None
cnet.tensor_free.argtypes = [ctypes.c_void_p]

cnet.tensor_get_grad.restype = ctypes.c_float
cnet.tensor_get_grad.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]

cnet.tensor_set_grad_scalar.restype = None
cnet.tensor_set_grad_scalar.argtypes = [
    ctypes.c_void_p, ctypes.c_float]

cnet.tensor_set_grad_flat.restype = None
cnet.tensor_set_grad_flat.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_float]

cnet.tensor_get.restype = ctypes.c_float
cnet.tensor_get.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]

cnet.tensor_get_flat.restype  = ctypes.c_float
cnet.tensor_get_flat.argtypes = [ctypes.c_void_p, ctypes.c_int]

cnet.tensor_set.restype = None
cnet.tensor_set.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_float]

cnet.tensor_set_flat.restype = None
cnet.tensor_set_flat.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_float]

cnet.tensor_zero_grad.restype = None
cnet.tensor_zero_grad.argtypes = [ctypes.c_void_p]

cnet.tensor_print.restype = None
cnet.tensor_print.argtypes = [ctypes.c_void_p]

cnet.tensor_print_grad.restype = None
cnet.tensor_print_grad.argtypes = [ctypes.c_void_p]

cnet.tensor_backward_step.restype = None
cnet.tensor_backward_step.argtypes = [ctypes.c_void_p]

cnet.tensor_backward.restype = None
cnet.tensor_backward.argtypes = [ctypes.c_void_p]

# math operations
cnet.tensor_add.restype = ctypes.c_void_p
cnet.tensor_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

cnet.tensor_sub.restype = ctypes.c_void_p
cnet.tensor_sub.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

cnet.tensor_mul.restype = ctypes.c_void_p
cnet.tensor_mul.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

cnet.tensor_relu.restype = ctypes.c_void_p
cnet.tensor_relu.argtypes = [ctypes.c_void_p]

cnet.tensor_sigmoid.restype = ctypes.c_void_p
cnet.tensor_sigmoid.argtypes = [ctypes.c_void_p]

cnet.tensor_log.restype = ctypes.c_void_p
cnet.tensor_log.argtypes = [ctypes.c_void_p]

cnet.tensor_matmul.restype = ctypes.c_void_p
cnet.tensor_matmul.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

cnet.tensor_sum.restype = ctypes.c_void_p
cnet.tensor_sum.argtypes = [ctypes.c_void_p]

cnet.tensor_copy_data_to_buffer.restype = None
cnet.tensor_copy_data_to_buffer.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]

cnet.tensor_set_data_from_buffer.restype = None
cnet.tensor_set_data_from_buffer.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]

cnet.tensor_get_grad_flat.restype  = ctypes.c_float
cnet.tensor_get_grad_flat.argtypes = [ctypes.c_void_p, ctypes.c_int]

cnet.tensor_sgd_step.restype  = None
cnet.tensor_sgd_step.argtypes = [ctypes.c_void_p, ctypes.c_float]

# MNIST
class MNISTDataC(ctypes.Structure):
    _fields_ = [
        ("images", ctypes.POINTER(ctypes.c_float)),
        ("labels", ctypes.POINTER(ctypes.c_int)),
        ("count",  ctypes.c_int),
        ("rows",   ctypes.c_int),
        ("cols",   ctypes.c_int),
    ]

cnet.mnist_load.restype  = ctypes.POINTER(MNISTDataC)
cnet.mnist_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p]

cnet.mnist_free.restype  = None
cnet.mnist_free.argtypes = [ctypes.POINTER(MNISTDataC)]


def broadcast_shape(a_shape, b_shape):
    out_ndim = max(len(a_shape), len(b_shape))
    out = []
    for i in range(out_ndim):
        ai   = len(a_shape) - out_ndim + i
        bi   = len(b_shape) - out_ndim + i
        adim = a_shape[ai] if ai >= 0 else 1
        bdim = b_shape[bi] if bi >= 0 else 1
        if   adim == bdim: out.append(adim)
        elif adim == 1:    out.append(bdim)
        elif bdim == 1:    out.append(adim)
        else: raise ValueError(f"shapes {a_shape} and {b_shape} are not broadcastable")
    return tuple(out)

def to_c_int_arr(*vals):
    arr = (ctypes.c_int * len(vals))(*vals)
    return arr

def infer_shape(data, shape = ()):
    if isinstance(data, (int, float)):
        return shape
    
    return infer_shape(data[0], shape + (len(data),))

def flatten(data):
    for item in data:
        if isinstance(item, (int, float)):
            yield item
        else:
            yield from flatten(item)

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.shape = infer_shape(data)
        flat = list(flatten(data))

        c_shape = to_c_int_arr(*self.shape)
        self.ptr = cnet.tensor_create(len(self.shape), c_shape, int(requires_grad))

        for i, val in enumerate(flat):
            cnet.tensor_set_flat(self.ptr, i, val)

    def __del__(self):
        if (self.ptr is not None):
            cnet.tensor_free(self.ptr)

    def __add__(self, other):
        return add(self, other)
    
    def __sub__(self, other):
        return sub(self, other)
    
    def __mul__(self, other):
        return mul(self, other)
    
    def __matmul__(self, other):
        return matmul(self, other)
    
    def set(self, indices, val):
        c_indices = to_c_int_arr(*indices)
        cnet.tensor_set(self.ptr, c_indices, val)

    def get(self, indices):
        c_indices = to_c_int_arr(*indices)
        return cnet.tensor_get(self.ptr, c_indices)

    def get_grad(self, indices):
        c_indices = to_c_int_arr(*indices)
        return cnet.tensor_get_grad(self.ptr, c_indices)

    def set_grad_flat(self, index, val):
        cnet.tensor_set_grad_flat(self.ptr, index, val)

    def zero_grad(self):
        cnet.tensor_zero_grad(self.ptr)
    
    def print(self):
        cnet.tensor_print(self.ptr)
    
    def print_grad(self):
        cnet.tensor_print_grad(self.ptr)

    def sgd_step(self, lr):
        cnet.tensor_sgd_step(self.ptr, lr)

    def argmax(self):
        n = math.prod(self.shape)
        buf = (ctypes.c_float * n)()
        cnet.tensor_copy_data_to_buffer(self.ptr, buf)
        return max(range(n), key=lambda i: buf[i])

    def backward_grad_set(self):
        """Backward pass using self.grad as-is (don't reset to 1.0)."""
        def build_topo(t, visited, order):
            if id(t) in visited:
                return
            visited.add(id(t))
            for child in getattr(t, '_children', []):
                build_topo(child, visited, order)
            order.append(t)
        visited, order = set(), []
        build_topo(self, visited, order)
        for t in reversed(order):
            cnet.tensor_backward_step(t.ptr)

    def backward(self):
        def build_topo(t, visited, order):
            if id(t) in visited:
                return
            visited.add(id(t))
            for child in getattr(t, '_children', []):
                build_topo(child, visited, order)
            order.append(t)

        visited, order = set(), []
        build_topo(self, visited, order)

        cnet.tensor_set_grad_scalar(self.ptr, 1.0)
        for t in reversed(order):
            cnet.tensor_backward_step(t.ptr)


def add(a, b):
    out = Tensor.__new__(Tensor)
    out.ptr = cnet.tensor_add(a.ptr, b.ptr)
    out.shape = broadcast_shape(a.shape, b.shape)
    out._children = [a, b]
    return out

def sub(a, b):
    out = Tensor.__new__(Tensor)
    out.ptr = cnet.tensor_sub(a.ptr, b.ptr)
    out.shape = broadcast_shape(a.shape, b.shape)
    out._children = [a, b]
    return out
    
def mul(a, b):
    out = Tensor.__new__(Tensor)
    out.ptr = cnet.tensor_mul(a.ptr, b.ptr)
    out.shape = a.shape
    out._children = [a, b]
    return out

def matmul(a, b):
    out = Tensor.__new__(Tensor)
    out.ptr = cnet.tensor_matmul(a.ptr, b.ptr)
    out.shape = (a.shape[0], b.shape[1])
    out._children = [a, b]
    return out

def log(a):
    out = Tensor.__new__(Tensor)
    out.ptr = cnet.tensor_log(a.ptr)
    out.shape = a.shape
    out._children = [a]
    return out

def sigmoid(a):
    out = Tensor.__new__(Tensor)
    out.ptr = cnet.tensor_sigmoid(a.ptr)
    out.shape = a.shape
    out._children = [a]
    return out

def relu(a):
    out = Tensor.__new__(Tensor)
    out.ptr = cnet.tensor_relu(a.ptr)
    out.shape = a.shape
    out._children = [a]
    return out

def sum(a):
    out = Tensor.__new__(Tensor)
    out.ptr = cnet.tensor_sum(a.ptr)
    out.shape = (1,)
    out._children = [a]
    return out

def randn(shape, requires_grad=False, scale=1.0):
    t = zeros(shape, requires_grad=requires_grad)
    n = math.prod(shape)
    for i in range(n):
        cnet.tensor_set_flat(t.ptr, i, random.gauss(0, scale))
    return t

def zeros(shape, requires_grad = False):
    t = Tensor.__new__(Tensor)
    t.shape = shape
    c_shape = to_c_int_arr(*shape)
    t.ptr = cnet.tensor_create(len(shape), c_shape, requires_grad)
    return t

def softmax(logits):
    n = math.prod(logits.shape)
    buf = (ctypes.c_float * n)()
    cnet.tensor_copy_data_to_buffer(logits.ptr, buf)
    values = list(buf)
    max_val = max(values)
    exps = [math.exp(v - max_val) for v in values]
    total = builtins.sum(exps)
    return [e / total for e in exps]

def cross_entropy_loss(logits, label):
    probs = softmax(logits)
    loss = -math.log(probs[label] + 1e-8)
    probs[label] -= 1.0
    for i, g in enumerate(probs):
        cnet.tensor_set_grad_flat(logits.ptr, i, g)

    return loss

def save_weights(path, tensors):
    with open(path, 'wb') as f:
        f.write(struct.pack('i', len(tensors)))
        for t in tensors:
            f.write(struct.pack('i', len(t.shape)))
            for s in t.shape:
                f.write(struct.pack('i', s))
            n = math.prod(t.shape)
            buf = (ctypes.c_float * n)()
            cnet.tensor_copy_data_to_buffer(t.ptr, buf)
            f.write(bytes(buf))

def load_weights(path, tensors):
    with open(path, 'rb') as f:
        count = struct.unpack('i', f.read(4))[0]
        if count != len(tensors):
            raise ValueError(f"file has {count} tensors, expected {len(tensors)}")
        for t in tensors:
            ndim = struct.unpack('i', f.read(4))[0]
            shape = tuple(struct.unpack('i', f.read(4))[0] for _ in range(ndim))
            if shape != t.shape:
                raise ValueError(f"shape mismatch: file has {shape}, tensor is {t.shape}")
            n = math.prod(shape)
            buf = (ctypes.c_float * n).from_buffer_copy(f.read(n * 4))
            cnet.tensor_set_data_from_buffer(t.ptr, buf)


class MNISTData:
    def __init__(self, image_path, label_path):
        self._ptr = cnet.mnist_load(image_path.encode(), label_path.encode())
        if not self._ptr:
            raise RuntimeError(f"Failed to load MNIST: {image_path}, {label_path}")
        d = self._ptr.contents
        self.count = d.count
        self.rows  = d.rows
        self.cols  = d.cols

    def __del__(self):
        if self._ptr:
            cnet.mnist_free(self._ptr)
            self._ptr = None

    def get_image(self, idx):
        n = self.rows * self.cols
        base = idx * n
        d = self._ptr.contents
        return [d.images[base + i] for i in range(n)]

    def get_label(self, idx):
        return self._ptr.contents.labels[idx]

    def get_image_tensor(self, idx):
        flat = self.get_image(idx)
        t = zeros((1, self.rows * self.cols))
        for i, v in enumerate(flat):
            cnet.tensor_set_flat(t.ptr, i, v)
        return t


if __name__ == "__main__":
    print("=== Test 1: create from data ===")
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    a.print()

    print("\n=== Test 2: zeros ===")
    b = zeros((2, 2))
    b.print()

    print("\n=== Test 3: operator overloading ===")
    c = a + b
    c.print()

    print("\n=== Test 4: matmul ===")
    w = Tensor([[1.0, 0.0], [0.0, 1.0]])
    d = a @ w
    print("expect same as a:")
    d.print()

    print("\n=== Test 5: backward ===")
    x = Tensor([[2.0, 3.0]], requires_grad=True)
    y = Tensor([[4.0, 5.0]], requires_grad=True)

    z = sum(x * y)
    z.backward()

    print("dz/dx expect [4.0, 5.0]:")
    x.print_grad()
    print("dz/dy expect [2.0, 3.0]:")
    y.print_grad()
