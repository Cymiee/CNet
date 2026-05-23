import ctypes
import os

dir = os.path.dirname(os.path.abspath(__file__))
cnet = ctypes.CDLL(os.path.join(dir, 'libcnet.so'))

# tensor operations
cnet.tensor_create.restype  = ctypes.c_void_p
cnet.tensor_create.argtypes = [
    ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]

cnet.tensor_free.restype = None
cnet.tensor_free.argtypes = [ctypes.c_void_p]

cnet.tensor_set_grad.restype = None
cnet.tensor_set_grad.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_float]

cnet.tensor_set_grad_scalar.restype = None
cnet.tensor_set_grad_scalar.argtypes = [
    ctypes.c_void_p, ctypes.c_float]

cnet.tensor_get.restype = ctypes.c_float
cnet.tensor_get.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]

cnet.tensor_set.restype = None
cnet.tensor_set.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_float]

cnet.tensor_print.restype = None
cnet.tensor_print.argtypes = [ctypes.c_void_p]

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


class Tensor:
    def __init__(self, shape, requires_grad=False):
        self.shape = shape
        c_shape = to_c_int_arr(*shape)
        self.ptr = cnet.tensor_create(len(shape), c_shape, int(requires_grad))

    def __del__(self):
        if (self.ptr is not None):
            cnet.tensor_free(self.ptr)
    
    def set(self, indices, val):
        c_indices = to_c_int_arr(*indices)
        cnet.tensor_set(self.ptr, c_indices, val)

    def get(self, indices):
        c_indices = to_c_int_arr(*indices)
        return cnet.tensor_get(self.ptr, c_indices)

    def print(self):
        cnet.tensor_print(self.ptr)

    def backward(self):
        cnet.tensor_set_grad_scalar(self.ptr, 1.0)
        cnet.tensor_backward(self.ptr)

def add(a, b):
    out = Tensor.__new__(Tensor)
    out.ptr = cnet.tensor_add(a.ptr, b.ptr)
    out.shape = a.shape
    return out

def sub(a, b):
    out = Tensor.__new__(Tensor)
    out.ptr = cnet.tensor_sub(a.ptr, b.ptr)
    out.shape = a.shape
    return out
    
def mul(a, b):
    out = Tensor.__new__(Tensor)
    out.ptr = cnet.tensor_mul(a.ptr, b.ptr)
    out.shape = a.shape
    return out

def matmul(a, b):
    out = Tensor.__new__(Tensor)
    out.ptr = cnet.tensor_matmul(a.ptr, b.ptr)
    out.shape = (a.shape[0], b.shape[1])
    return out

def log(a):
    out = Tensor.__new__(Tensor)
    out.ptr = cnet.tensor_log(a.ptr)
    out.shape = a.shape
    return out

def sigmoid(a):
    out = Tensor.__new__(Tensor)
    out.ptr = cnet.tensor_sigmoid(a.ptr)
    out.shape = a.shape
    return out

def sum(a):
    out = Tensor.__new__(Tensor)
    out.ptr = cnet.tensor_sum(a.ptr)
    out.shape = a.shape
    return out

a = Tensor((2, 2), requires_grad=True)
b = Tensor((2, 2), requires_grad=True)
a.set([0,0], 1.0); a.set([0,1], 2.0)
a.set([1,0], 3.0); a.set([1,1], 4.0)
b.set([0,0], 1.0); b.set([0,1], 1.0)
b.set([1,0], 1.0); b.set([1,1], 1.0)

c = matmul(a, b)
c.print()
