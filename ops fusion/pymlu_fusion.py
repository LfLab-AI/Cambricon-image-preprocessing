import os
import ctypes
import numpy as np


# Loading dynamic link library
current_dir = os.path.dirname(os.path.abspath(__file__))
library_path = os.path.join(current_dir, 'mlu.so')
mlu_lib = ctypes.CDLL(library_path)


# Automatic resource initialization & release
class ResourceManager:
    def __init__(self):
        mlu_lib.initialize()
        print("[Initialize]: set device 0 and create queue")
    
    def __del__(self):
        mlu_lib.release()
        print("[Release]: destroy queue")

resource_manager = ResourceManager()


# The python interface is defined as follows

def gray_hflip(img):
    height, width = img.shape[:2]
    input_channels = 1 if len(img.shape) == 2 else img.shape[2]
    
    if input_channels != 3:
        raise ValueError('The number of channels for the input image should be 3')
    
    new_img = np.empty((height, width), dtype=np.uint8)

    src_ptr = np.asarray(img).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    dst_ptr = new_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    mlu_lib.gray_hflip(dst_ptr, src_ptr, height, width)

    return new_img


def resize_hflip(img, height, width):
    src_height, src_width = img.shape[:2]
    input_channels = 1 if len(img.shape) == 2 else img.shape[2]
    
    if input_channels != 3:
        raise ValueError('The number of channels for the input image should be 3')
    
    new_img = np.empty((height, width, 3), dtype=np.uint8)

    src_ptr = np.asarray(img).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    dst_ptr = new_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    mlu_lib.resize_hflip(dst_ptr, src_ptr, height, width, src_height, src_width)

    new_img = new_img[..., ::-1]

    return new_img
