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

cnet.tensor_get.restype = ctypes.c_float
cnet.tensor_set.argtypes = [
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


def create_shape(*dims):
    arr = (ctypes.c_int * len(dims))(*dims)
    return arr