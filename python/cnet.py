import ctypes
import os

dir = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(os.path.join(dir, 'libcnet.so'))
print('loaded')