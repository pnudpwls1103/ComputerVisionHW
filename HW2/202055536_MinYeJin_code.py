#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import numpy as np
import math

# Part 1: Gaussian Filtering
# 1-1 : boxfilter
def boxfilter(n):
    assert n % 2 != 0, "Dimension must be odd"
    return np.ones((n, n)) / (n*n)

print(boxfilter(3))
print(boxfilter(4))
print(boxfilter(7))


# 1-2
def gauss1d(sigma):
    # gaussian 함수
    def gaussian(sigma, x):
        return 1 / np.sqrt(2*np.pi*sigma**2) * np.exp(-(x**2)/(2*sigma**2))
    # sigma의 6배 후 홀수로 만들기
    length = np.round(6*sigma)
    length = length if length % 2 else length + 1
    
    arr = np.array(range(int(-length/2), int(length/2)+1))
    result = [gaussian(sigma, x) for x in arr]
    # 정규화
    result = result / sum(result)
    return result

print(gauss1d(0.3))
print(gauss1d(0.5))
print(gauss1d(1))
print(gauss1d(2))


# 1-3
def gauss2d(sigma):
    arr1d = gauss1d(sigma)
    # 1d gaussian filter의 transpose
    arr1dT = np.transpose(arr1d)
    # gaussian filter와 transpose의 외적
    result = np.outer(arr1d, arr1dT)
    # 정규화
    result = result / sum(result.flatten())
    return result

print(gauss2d(0.5))
print(gauss2d(1))


# 1-4-a
def convolve2d(array, filter):
    # np.float32로 type 변환
    array_copy = array.astype(np.float32).copy()
    filter = filter.astype(np.float32)

    # image zero padding
    pad_len = int((len(filter[0])-1)/2)
    array_copy = np.pad(array_copy, (pad_len, pad_len), constant_values=0)

    # Convolution
    # 1. kernel 뒤집기
    filter_len = len(filter)
    filter = filter.flatten()[::-1]
    filter = filter.reshape(filter_len, -1)

    # 2. Cross Correlation
    result_width, result_height = len(array[0]), len(array)
    result = np.zeros((result_height, result_width))
    for i in range(result_height):
        for j in range(result_width):
            image_crop = array_copy[i:i+filter_len, j:j+filter_len]
            result[i, j] = np.sum(image_crop * filter)

    return result


# 1-4-b
def gaussconvolve2d(array, sigma):   
    filter = gauss2d(sigma)
    return convolve2d(array, filter)


# 1-4-c
image1 = Image.open('./hw2_image/2b_dog.bmp')

# convert to greyscale
image1 = image1.convert('L')
image_arr = np.asarray(image1)
result_c = gaussconvolve2d(image_arr, 3).astype('uint8')
image_c = Image.fromarray(result_c)

image1.show()
image_c.show()


# 2-1
image2 = Image.open('./hw2_image/3b_tower.bmp')
# 이미지의 r, g, b 채널 분리
r, g, b = image2.split()

arr_r, arr_g, arr_b = np.asarray(r), np.asarray(g), np.asarray(b)
# 가우시안 블러링 
image2_result_r = gaussconvolve2d(arr_r, 1.5)
image2_result_g = gaussconvolve2d(arr_g, 1.5)
image2_result_b = gaussconvolve2d(arr_b, 1.5)

# uint8로 타입 변환 후 Image 객체로 변환
image2_result_r, image2_result_g, image2_result_b = image2_result_r.astype('uint8'), image2_result_g.astype('uint8'), image2_result_b.astype('uint8')
new_r, new_g, new_b = Image.fromarray(image2_result_r), Image.fromarray(image2_result_g), Image.fromarray(image2_result_b)
new_image2 = Image.merge('RGB', (new_r, new_g, new_b))
new_image2.show()


# 2-2
image3 = Image.open('./hw2_image/3a_eiffel.bmp')
# 이미지의 r, g, b 채널 분리
r, g, b = image3.split()

arr_r, arr_g, arr_b = np.asarray(r), np.asarray(g), np.asarray(b)
# 가우시안 블러링 
blur_r = gaussconvolve2d(arr_r, 1.5)
blur_g = gaussconvolve2d(arr_g, 1.5)
blur_b = gaussconvolve2d(arr_b, 1.5)

# 원본 이미지 - 블러링 이미지
image3_result_r = arr_r - blur_r
image3_result_g = arr_g - blur_g
image3_result_b = arr_b - blur_b

# 음수 값 보정
modi_result_r = image3_result_r + 128
modi_result_g = image3_result_g + 128
modi_result_b = image3_result_b + 128

# 255 초과인 값 보정
modi_result_r[np.where(modi_result_r > 255)] = 255
modi_result_g[np.where(modi_result_g > 255)] = 255
modi_result_b[np.where(modi_result_b > 255)] = 255

# uint8로 타입 변환 후 Image 객체로 변환
modi_result_r, modi_result_g, modi_result_b = modi_result_r.astype('uint8'), modi_result_g.astype('uint8'), modi_result_b.astype('uint8')
new_r, new_g, new_b = Image.fromarray(modi_result_r), Image.fromarray(modi_result_g), Image.fromarray(modi_result_b)
new_image2 = Image.merge('RGB', (new_r, new_g, new_b))
new_image2.show()


# 2-3
# image2의 블러링 이미지 + image3의 high frequency 이미지
hybrid_r = image2_result_r + image3_result_r
hybrid_g = image2_result_g + image3_result_g
hybrid_b = image2_result_b + image3_result_b

# 음수 값과 255 초과인 값 보정
hybrid_r[np.where(hybrid_r < 0)] = 0
hybrid_r[np.where(hybrid_r > 255)] = 255
hybrid_g[np.where(hybrid_g < 0)] = 0
hybrid_g[np.where(hybrid_g > 255)] = 255
hybrid_b[np.where(hybrid_b < 0)] = 0
hybrid_b[np.where(hybrid_b > 255)] = 255

# uint로 타입 변환 후 Image 객체로 변환
hybrid_r, hybrid_g, hybrid_b = hybrid_r.astype('uint8'), hybrid_g.astype('uint8'), hybrid_b.astype('uint8')
new_r, new_g, new_b = Image.fromarray(hybrid_r), Image.fromarray(hybrid_g), Image.fromarray(hybrid_b)
new_image3 = Image.merge('RGB', (new_r, new_g, new_b))
new_image3.show()

