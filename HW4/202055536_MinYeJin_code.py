import numpy as np
import cv2
import math
import random


def RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement):
    """
    This function takes in `matched_pairs`, a list of matches in indices
    and return a subset of the pairs using RANSAC.
    Inputs:
        matched_pairs: a list of tuples [(i, j)],
            indicating keypoints1[i] is matched
            with keypoints2[j]
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        *_agreement: thresholds for defining inliers, floats
    Output:
        largest_set: the largest consensus set in [(i, j)] format

    HINTS: the "*_agreement" definitions are well-explained
           in the assignment instructions.
    """
    assert isinstance(matched_pairs, list)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    ## START
    orient_agreement = math.pi / 180 * orient_agreement
    
    largest_set = []
    for i in range(10): #repeat ten times
        rand = random.randrange(0,len(matched_pairs)) #generate random number 
        choice = matched_pairs[rand]
        orientation = (keypoints1[choice[0]][3] - keypoints2[choice[1]][3])%(2 * math.pi) #calculation first-orientation
        scale = keypoints2[choice[1]][2]/keypoints1[choice[0]][2] #calculation frist-scale ratio
        temp = []
        for j in range(len(matched_pairs)): #calculate the number of all cases
            if j is not rand:
                #calculation second-orientation
                orientation_temp = (keypoints1[matched_pairs[j][0]][3] - keypoints2 [matched_pairs[j][1]][3])%(2*math.pi) #calculation second-scale ratio
                scale_temp = keypoints2[matched_pairs[j][1]][2]/keypoints1[matched_pairs[j][0]][2]
                #check degree error -30degree
                if((orientation-orient_agreement)<orientation_temp) and (orientation_temp<(orientation+orient_agreement)):
                #check scale error +-50%
                    if(scale - scale*scale_agreement < scale_temp and scale_temp < scale + scale*scale_agreement):
                        temp.append([i,j])
        if(len(temp)>len(largest_set)):#choice best match
            largest_set = temp
    for i in range(len(largest_set)):
        largest_set[i] = (matched_pairs[largest_set[i][1]][0], matched_pairs[largest_set[i][1]][1])
    ## END
    assert isinstance(largest_set, list)
    return largest_set



def FindBestMatches(descriptors1, descriptors2, threshold):
    """
    This function takes in descriptors of image 1 and image 2,
    and find matches between them. See assignment instructions for details.
    Inputs:
        descriptors: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
    Outputs:
        matched_pairs: a list in the form [(i, j)] where i and j means
                       descriptors1[i] is matched with descriptors2[j].
    """
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    ## START
    ## the following is just a placeholder to show you the output format
    y1 = descriptors1.shape[0]
    y2 = descriptors2.shape[0]
    temp = np.zeros(y2)
    matched_pairs = []
    for i in range(y1):
        for j in range(y2):
            temp[j] = math.acos(np.dot(descriptors1[i], descriptors2[j]))
        compare = sorted(range(len(temp)), key = lambda k : temp[k])
        if (temp[compare[0]] / temp[compare[1]]) < threshold:
            matched_pairs.append([i, compare[0]])
    ## END
    return matched_pairs

def KeypointProjection(xy_points, h):
    """
    This function projects a list of points in the source image to the
    reference image using a homography matrix `h`.
    Inputs:
        xy_points: numpy array, (num_points, 2)
        h: numpy array, (3, 3), the homography matrix
    Output:
        xy_points_out: numpy array, (num_points, 2), input points in
        the reference frame.
    """
    assert isinstance(xy_points, np.ndarray)
    assert isinstance(h, np.ndarray)
    assert xy_points.shape[1] == 2
    assert h.shape == (3, 3)

    # START
    xy_points_temp = np.c_[xy_points, [1] * len(xy_points)]
    xy_points_out = h @ xy_points_temp.T
    xy_points_out = xy_points_out.T

    for i, point in enumerate(xy_points_out):
        div = point[2]
        if div == 0.0:
            div = 1e10
        
        xy_points_out[i] /= div

    xy_points_out = np.delete(xy_points_out, -1, axis = 1)
    # END
    return xy_points_out

def RANSACHomography(xy_src, xy_ref, num_iter, tol):
    """
    Given matches of keyponit xy coordinates, perform RANSAC to obtain
    the homography matrix. At each iteration, this function randomly
    choose 4 matches from xy_src and xy_ref.  Compute the homography matrix
    using the 4 matches.  Project all source "xy_src" keypoints to the
    reference image.  Check how many projected keyponits are within a `tol`
    radius to the coresponding xy_ref points (a.k.a. inliers).  During the
    iterations, you should keep track of the iteration that yields the largest
    inlier set. After the iterations, you should use the biggest inlier set to
    compute the final homography matrix.
    Inputs:
        xy_src: a numpy array of xy coordinates, (num_matches, 2)
        xy_ref: a numpy array of xy coordinates, (num_matches, 2)
        num_iter: number of RANSAC iterations.
        tol: float
    Outputs:
        h: The final homography matrix.
    """
    assert isinstance(xy_src, np.ndarray)
    assert isinstance(xy_ref, np.ndarray)
    assert xy_src.shape == xy_ref.shape
    assert xy_src.shape[1] == 2
    assert isinstance(num_iter, int)
    assert isinstance(tol, (int, float))
    tol = tol*1.0

    # START
    h = []
    largest_size = 0
    for iter in range(num_iter):
        rand = random.sample(list(range(len(xy_src))), 4)
        homo_src = np.array([xy_src[r] for r in rand])
        homo_ref = np.array([xy_ref[r] for r in rand])

        # homography matrix 구하기
        matrix_a = []
        for r in rand:
            homo_src = xy_src[r]
            homo_ref = xy_ref[r]
            matrix_a.append([homo_src[0], homo_src[1], 1, 0, 0, 0, -homo_ref[0]*homo_src[0], -homo_ref[0]*homo_src[1], -homo_ref[0]])
            matrix_a.append([0, 0, 0, homo_src[0], homo_src[1], 1, -homo_ref[1]*homo_src[0], -homo_ref[1]*homo_src[1], -homo_ref[1]])
        matrix_a = np.asarray(matrix_a)

        # 고유값으로 homography matrix의 값 구하기
        eigvals, eigvecs = np.linalg.eig(matrix_a.T @ matrix_a)
        v = eigvecs[:, np.argmin(eigvals)]
        h_temp = v.reshape((3, 3))
        h_temp /= h_temp[-1, -1]
        
        # projection 후 inlier인지 판별
        proj = KeypointProjection(xy_src, h_temp)
        dist = (proj[:,0] - xy_ref[:,0]) ** 2 + (proj[:,1] - xy_ref[:,1]) ** 2
        consensus = dist[np.where(tol * tol >= dist)]
        
        if (largest_size < consensus.shape[0]):
            largest_size = consensus.shape[0]
            h = h_temp

    # END
    assert isinstance(h, np.ndarray)
    assert h.shape == (3, 3)
    return h


def FindBestMatchesRANSAC(
        keypoints1, keypoints2,
        descriptors1, descriptors2, threshold,
        orient_agreement, scale_agreement):
    """
    Note: you do not need to change this function.
    However, we recommend you to study this function carefully
    to understand how each component interacts with each other.

    This function find the best matches between two images using RANSAC.
    Inputs:
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        descriptors1, 2: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
        orient_agreement: in degrees, say 30 degrees.
        scale_agreement: in floating points, say 0.5
    Outputs:
        matched_pairs_ransac: a list in the form [(i, j)] where i and j means
        descriptors1[i] is matched with descriptors2[j].
    Detailed instructions are on the assignment website
    """
    orient_agreement = float(orient_agreement)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    matched_pairs = FindBestMatches(
        descriptors1, descriptors2, threshold)
    matched_pairs_ransac = RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement)
    return matched_pairs_ransac
