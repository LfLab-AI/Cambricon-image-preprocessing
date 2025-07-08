import os
import ctypes
import numpy as np
import math

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


# The auxiliary functions of the interface are defined as follows

def is_list_of_lists_of_ints(variable):
    return (
        isinstance(variable, list) and
        len(variable) == 4 and
        all(
            isinstance(sublist, list) and
            len(sublist) == 2 and
            all(isinstance(item, int) for item in sublist)
            for sublist in variable
        )
    )


# The python interface is defined as follows

def rotate(img, angle):
    src_height, src_width = img.shape[:2]
    radians = angle * math.pi / 180.0
    cos_angle = math.cos(radians)
    sin_angle = math.sin(radians)
    dst_height = int(math.fabs(src_height * cos_angle) + math.fabs(src_width * sin_angle))
    dst_width = int(math.fabs(src_width * cos_angle) + math.fabs(src_height * sin_angle))
    input_channels = 1 if len(img.shape) == 2 else img.shape[2]

    new_img = np.empty((dst_height, dst_width, input_channels), dtype=np.uint8)
    src_ptr = np.asarray(img).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    dst_ptr = new_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    
    mlu_lib.rotate(src_ptr, dst_ptr, src_height, src_width, dst_height, dst_width, input_channels, ctypes.c_double(angle))
    return new_img 

def affine(img, angle, sx, sy, tx, ty, shear_x, shear_y):
    src_height, src_width = img.shape[:2]
    input_channels = 1 if len(img.shape) == 2 else img.shape[2]

    new_img = np.empty((src_height, src_width, input_channels), dtype=np.uint8)
    src_ptr = np.asarray(img).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    dst_ptr = new_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    
    mlu_lib.affine(src_ptr, dst_ptr, src_height, src_width, input_channels, ctypes.c_float(angle), ctypes.c_float(sx), ctypes.c_float(sy), ctypes.c_float(tx), ctypes.c_float(ty), ctypes.c_float(shear_x), ctypes.c_float(shear_y))
    return new_img

def erase(img, top, left, height, width):
    src_height, src_width = img.shape[:2]
    input_channels = 1 if len(img.shape) == 2 else img.shape[2]

    new_img = np.empty((src_height, src_width, input_channels), dtype=np.uint8)
    src_ptr = np.asarray(img).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    dst_ptr = new_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    
    mlu_lib.erase(src_ptr, dst_ptr, src_height, src_width, input_channels, top, left, height, width)
    return new_img
    

def resize(img, width, height):

    src_height, src_width = img.shape[:2]
 #   input_channels = 1 if len(img.shape) == 2 else img.shape[2]

 #   if input_channels != 3:
 #       raise ValueError('The number of channels for the input image should be 3')

 #   if output_channels == 1:
 #       new_img = np.empty((height, width), dtype=np.uint8)
 #   elif output_channels == 3:
    new_img = np.empty((height, width, 3), dtype=np.uint8)
 #   else:
 #       raise ValueError('The number of channels for the output image should be 1 or 3')
    stepDst = width*3
    stepSrc = src_width*3
    src_ptr = np.asarray(img).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    dst_ptr = new_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    mlu_lib.resizeImage(dst_ptr, src_ptr, width,height,src_height,src_width, stepDst,stepSrc)

    return new_img

def adjustContrast(img,alpha):

    src_height, src_width = img.shape[:2]
 #   input_channels = 1 if len(img.shape) == 2 else img.shape[2]

 #   if input_channels != 3:
 #       raise ValueError('The number of channels for the input image should be 3')

 #   if output_channels == 1:
 #       new_img = np.empty((height, width), dtype=np.uint8)
 #   elif output_channels == 3:
    new_img = np.empty((src_height, src_width, 3), dtype=np.uint8)
 #   else:
 #       raise ValueError('The number of channels for the output image should be 1 or 3')
    
    src_ptr = np.asarray(img).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    dst_ptr = new_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    
    mlu_lib.adjustContrast(dst_ptr, src_ptr, src_height,src_width)

    return new_img


def adjustBrightness(img,alpha):

    src_height, src_width = img.shape[:2]
 #   input_channels = 1 if len(img.shape) == 2 else img.shape[2]

 #   if input_channels != 3:
 #       raise ValueError('The number of channels for the input image should be 3')

 #   if output_channels == 1:
 #       new_img = np.empty((height, width), dtype=np.uint8)
 #   elif output_channels == 3:
    new_img = np.empty((src_height, src_width, 3), dtype=np.uint8)
 #   else:
 #       raise ValueError('The number of channels for the output image should be 1 or 3')

    src_ptr = np.asarray(img).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    dst_ptr = new_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    mlu_lib.adjustBrightness(dst_ptr, src_ptr, src_height,src_width)

    return new_img


def to_grayscale(img, output_channels=1):
    """
    Converts image to grayscale version of image.

    Args:
        img (np.array): Image to be converted to grayscale.
        output_channels (int, optional, default=1): The number of channels for the output image.
    Returns:
        new_img (np.array): Grayscale version of the image.
            if output_channels = 1 : returned image is single channel.
            if output_channels = 3 : returned image is 3 channel with R = G = B.

    Examples:
        .. code-block:: python
            >>> import cv2
            >>> from pymlu import to_grayscale
            >>> src_path = '/Path/to/src.jpg'
            >>> dst_path = '/Path/to/dst.jpg'
            >>> src_img = cv2.imread(src_path)
            >>> dst_img = to_grayscale(src_img)
            >>> cv2.imwrite(dst_path, dst_img)
    """
    height, width = img.shape[:2]
    input_channels = 1 if len(img.shape) == 2 else img.shape[2]
    
    if input_channels != 3:
        raise ValueError('The number of channels for the input image should be 3')
    
    if output_channels == 1:
        new_img = np.empty((height, width), dtype=np.uint8)
    elif output_channels == 3:
        new_img = np.empty((height, width, 3), dtype=np.uint8)
    else:
        raise ValueError('The number of channels for the output image should be 1 or 3')

    src_ptr = np.asarray(img).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    dst_ptr = new_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    mlu_lib.to_grayscale(dst_ptr, src_ptr, height*width, output_channels)

    return new_img
    
def flip(img):
    """
    Flips the image vertically.

    Args:
        img (np.array): Image to be flipped vertically.

    Returns:
        new_img (np.array): Vertically flipped version of the image.

    Examples:
        .. code-block:: python
            >>> import cv2
            >>> from pymlu import vflip
            >>> src_path = '/Path/to/src.jpg'
            >>> dst_path = '/Path/to/dst.jpg'
            >>> src_img = cv2.imread(src_path)
            >>> dst_img = vflip(src_img)
            >>> cv2.imwrite(dst_path, dst_img)
    """
    height, width = img.shape[:2]
    new_img = np.empty((height, width, 3), dtype=np.uint8)
    src_ptr = np.asarray(img).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    dst_ptr = new_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    mlu_lib.flip(dst_ptr, src_ptr, height, width)
    return new_img


def pad(img, padding, fill=0):
    """
    Pads the given image on all sides with specified padding mode and fill value.

    Args:
        img (np.array): Image to be padded.
        padding (int|list|tuple): Padding on each border.
            if a single int is provided, this is used to pad all borders.
            if a list/tuple of length 2 is provided, this is the padding on left/right and top/bottom respectively.
            if a list/tuple of length 4 is provided, this is the padding for the left, top, right and bottom borders respectively.
        fill (int|list|tuple, optional, default=0): Pixel fill value for constant fill.
            if a single int is provided, this is used to fill all channels.
            if a list/tuple of length 3 is provided, this is used to fill R, G, B channels respectively.
        padding_mode: Type of padding.
            now only support constant: pads with a constant value, this value is specified with fill.
    Returns:
        new_img (np.array): Padded image.

    Examples:
        .. code-block:: python
            >>> import cv2
            >>> from pymlu import pad
            >>> src_path = '/Path/to/src.jpg'
            >>> dst_path = '/Path/to/dst.jpg'
            >>> src_img = cv2.imread(src_path)
            >>> dst_img = pad(src_img, padding=[100, 100, 100, 100], fill=[255, 255, 0])
            >>> cv2.imwrite(dst_path, dst_img)
    """
    height, width = img.shape[:2]
    input_channels = 1 if len(img.shape) == 2 else img.shape[2]
    
    if input_channels != 3:
        raise ValueError('The number of channels for the input image should be 3')
    
    if isinstance(padding, int):
        left, top, right, bottom = padding, padding, padding, padding
    elif isinstance(padding, (list, tuple)) and len(padding) == 2:
        left, top, right, bottom = int(padding[0]), int(padding[0]), int(padding[1]), int(padding[1])
    elif isinstance(padding, (list, tuple)) and len(padding) == 4:
        left, top, right, bottom = int(padding[0]), int(padding[1]), int(padding[2]), int(padding[3])
    else:
        raise ValueError('Wrong type of parameter padding')
    
    if top < 0 or bottom < 0 or left < 0 or right < 0:
        raise ValueError('The padding of the border is at least 0')

    new_height = height + top + bottom
    new_width = width + left + right

    if isinstance(fill, int):
        padding_buffer = np.tile(np.array([[np.uint8(fill), np.uint8(fill), np.uint8(fill)]]), (new_width, 1))
    elif isinstance(fill, (list, tuple)) and len(fill) == 3:
        padding_buffer = np.tile(np.array([[np.uint8(fill[0]), np.uint8(fill[1]), np.uint8(fill[2])]]), (new_width, 1))
    else:
        raise ValueError('Wrong type of parameter fill')
    padding_ptr = padding_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    new_img = np.empty((new_height, new_width, 3), dtype=np.uint8)

    src_ptr = np.asarray(img).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    dst_ptr = new_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    mlu_lib.pad(dst_ptr, src_ptr, padding_ptr, height, width, top, bottom, left, right)

    return new_img

def perspective(img, startpoints, endpoints, fill=0):
    """
    Perform perspective transform of the given image.

    Args:
        img (np.array): Image to be transformed.
        startpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the original image.
        endpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the transformed image.
        fill (int|list|tuple, optional, default=0): Pixel fill value for the area outside the transformed image.
            if a single int is provided, this is used to fill all channels.
            if a list/tuple of length 3 is provided, this is used to fill R, G, B channels respectively.
        interpolation: Interpolation method.
            now only support nearest.
    Returns:
        new_img (np.array): Transformed image.

    Examples:
        .. code-block:: python
            >>> import cv2
            >>> from pymlu import perspective
            >>> src_path = '/Path/to/src.jpg'
            >>> dst_path = '/Path/to/dst.jpg'
            >>> startpoints = [[0, 0], [33, 0], [33, 25], [0, 25]]
            >>> endpoints = [[3, 2], [32, 3], [30, 24], [2, 25]]
            >>> src_img = cv2.imread(src_path)
            >>> dst_img = perspective(src_img, startpoints, endpoints)
            >>> cv2.imwrite(dst_path, dst_img)
    """
    height, width = img.shape[:2]
    input_channels = 1 if len(img.shape) == 2 else img.shape[2]
    
    if input_channels != 3:
        raise ValueError('The number of channels for the input image should be 3')
    
    if is_list_of_lists_of_ints(startpoints):
        startpoints = np.array(startpoints, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    else:
        raise ValueError('Wrong type of parameter startpoints')
    if is_list_of_lists_of_ints(endpoints):
        endpoints = np.array(endpoints, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    else:
        raise ValueError('Wrong type of parameter endpoints')
    
    if isinstance(fill, int):
        fill = np.array([np.uint8(fill), np.uint8(fill), np.uint8(fill)]).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    elif isinstance(fill, (list, tuple)) and len(fill) == 3:
        fill = np.array([np.uint8(fill[0]), np.uint8(fill[1]), np.uint8(fill[2])]).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    else:
        raise ValueError('Wrong type of parameter fill')
    
    new_img = np.empty((height, width, input_channels), dtype=np.uint8)

    src_ptr = np.asarray(img).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    dst_ptr = new_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    mlu_lib.perspective(dst_ptr, src_ptr, height, width, startpoints, endpoints, fill)

    return new_img

def normalize(img):
    """
    Normalizes the image using the MLU-based normalizeImage function.

    Args:
        img (np.array): Image to be normalized.
    Returns:
        new_img (np.array): Normalized image (float values).
    """
    # 获取图像尺寸
    height, width,channel = img.shape
    image_size = height * width * channel  # 假设每个像素有3个颜色通道（RGB）

    # 创建一个空的目标图像数组
    new_img = np.empty((height, width, 3), dtype=np.float32)

    # 将源图像转换为ctypes数据指针
    src_ptr = img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    # 将目标图像转换为ctypes数据指针
    dst_ptr = new_img.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # 调用MLU的归一化接口
    mlu_lib.normalizeImage(dst_ptr, src_ptr, image_size)

    return new_img
