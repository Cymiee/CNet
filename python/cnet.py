import ctypes
import os

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

cnet.tensor_print.restype = None
cnet.tensor_print.argtypes = [ctypes.c_void_p]

cnet.tensor_print_grad.restype = None
cnet.tensor_print_grad.argtypes = [ctypes.c_void_p]

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

    def print(self):
        cnet.tensor_print(self.ptr)
    
    def print_grad(self):
        cnet.tensor_print_grad(self.ptr)

    def backward(self):
        cnet.tensor_set_grad_scalar(self.ptr, 1.0)
        cnet.tensor_backward(self.ptr)


def add(a, b):
    out = Tensor.__new__(Tensor)
    out.ptr = cnet.tensor_add(a.ptr, b.ptr)
    out.shape = a.shape
    out._children = [a, b]
    return out

def sub(a, b):
    out = Tensor.__new__(Tensor)
    out.ptr = cnet.tensor_sub(a.ptr, b.ptr)
    out.shape = a.shape
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

def zeros(shape, requires_grad = False):
    t = Tensor.__new__(Tensor)
    t.shape = shape
    c_shape = to_c_int_arr(*shape)
    t.ptr = cnet.tensor_create(len(shape), c_shape, requires_grad)
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
