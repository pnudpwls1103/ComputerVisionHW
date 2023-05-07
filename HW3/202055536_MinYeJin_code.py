from PIL import Image
import math
import numpy as np

"""
Get and use the functions associated with gaussconvolve2d that you used in the last HW02.
"""
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

def gauss2d(sigma):
    arr1d = gauss1d(sigma)
    # 1d gaussian filter의 transpose
    arr1dT = np.transpose(arr1d)
    # gaussian filter와 transpose의 외적
    result = np.outer(arr1d, arr1dT)
    # 정규화
    result = result / sum(result.flatten())
    return result

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

def gaussconvolve2d(array, sigma):   
    filter = gauss2d(sigma)
    return convolve2d(array, filter)

def sobel_filters(img):
    """ Returns gradient magnitude and direction of input img.
    Args:
        img: Grayscale image. Numpy array of shape (H, W).
    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction of gradient at each pixel in img.
            Numpy array of shape (H, W).
    Hints:
        - Use np.hypot and np.arctan2 to calculate square root and arctan
    """
    x_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    y_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    Ix = convolve2d(img, x_filter)
    Iy = convolve2d(img, y_filter)

    # G : gradient
    G = np.hypot(Ix, Iy)    
    G = G / G.max() * 255

    # theta : direction
    theta = np.arctan2(Iy, Ix)

    return (G, theta)

def non_max_suppression(G, theta):
    """ Performs non-maximum suppression.
    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).
    Returns:
        res: non-maxima suppressed image.
    """
    # radian 변환 후 음수 -> 양수 범위로 변환
    angle = np.degrees(theta)
    angle[np.where(angle < 0)] += 180
    angle[np.where(angle > 180)] -= 180

    H, W = G.shape
    res = np.zeros((H, W))

    # 각도별 neighbor 구하기
    for i in range(1, H-1):
        for j in range(1, W-1):
            # angle : 0
            if (0 <= angle[i, j] < 45):
                neighbors = [ G[i, j-1], G[i, j+1] ]
            # angle : 45
            elif (45 <= angle[i, j] < 90):
                neighbors = [ G[i-1, j+1], G[i+1, j-1] ]
            # angle : 90
            elif (90 <= angle[i, j] < 135):
                neighbors = [ G[i-1, j], G[i+1, j] ]
            # angle : 135
            elif (135 <= angle[i, j] < 180):
                neighbors = [ G[i-1, j-1], G[i+1, j+1] ]
            
            # max 검사
            if G[i, j] > np.max(neighbors):
                res[i, j] = G[i, j]

    return res

def double_thresholding(img):
    """ 
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: double_thresholded image.
    """

    res = np.zeros(img.shape)

    diff = np.max(img) - np.min(img)
    t_high = np.min(img) + diff * 0.15
    t_low = np.min(img) + diff * 0.03

    # weak edge
    res[np.where(t_low < img)] = 80
    # strong edge
    res[np.where(img > t_high)] = 255

    return res

def dfs(img, res, i, j, visited=[]):
    # 호출된 시점의 시작점 (i, j)은 최초 호출이 아닌 이상 
    # strong 과 연결된 weak 포인트이므로 res에 strong 값을 준다
    res[i, j] = 255

    # 이미 방문했음을 표시한다
    visited.append((i, j))

    # (i, j)에 연결된 8가지 방향을 모두 검사하여 weak 포인트가 있다면 재귀적으로 호출
    for ii in range(i-1, i+2) :
        for jj in range(j-1, j+2) :
            if (img[ii, jj] == 80) and ((ii, jj) not in visited) :
                dfs(img, res, ii, jj, visited)

def hysteresis(img):
    """ Find weak edges connected to strong edges and link them.
    Iterate over each pixel in strong_edges and perform depth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: hysteresised image.
    """
    res = np.zeros(img.shape)
    
    # strong edge 추출
    strong_x, strong_y = np.where(img == 255)

    # strong edge 인근의 weak edge를 구한다
    visited = []
    for i, j in zip(strong_x, strong_y):
        dfs(img, res, i, j, visited)
    
    return res



def main():
    image = Image.open('iguana.bmp')
    image1 = image.convert('L')
    image1_arr = np.asarray(image1)
    image1_blur = gaussconvolve2d(image1_arr, 1.6)

    image1 = Image.fromarray(image1_blur.astype('uint8'))
    image1.show()

    G, theta = sobel_filters(image1_blur)
    image2 = Image.fromarray(G.astype('uint8'))
    image2.show()

    image3_arr = non_max_suppression(G, theta)
    image3 = Image.fromarray(image3_arr.astype('uint8'))
    image3.show()

    image4_arr = double_thresholding(image3_arr)
    image4 = Image.fromarray(image4_arr.astype('uint8'))
    image4.show()

    image5_arr = hysteresis(image4_arr)
    image5 = Image.fromarray(image5_arr.astype('uint8'))
    image5.show()

main()