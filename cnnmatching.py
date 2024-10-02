import argparse
import cv2
import numpy as np
import imageio
import plotmatch
from lib.cnn_feature import cnn_feature_extract
import matplotlib.pyplot as plt
from skimage import measure
from skimage import transform


def cnn_matching(path1,path2):

    _RESIDUAL_THRESHOLD = 30
    
    image1 = imageio.imread(path1)
    image2 = imageio.imread(path2)

    
    kps_left, sco_left, des_left = cnn_feature_extract(image1,  nfeatures = -1)
    kps_right, sco_right, des_right = cnn_feature_extract(image2,  nfeatures = -1)
    


    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=40)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_left, des_right, k=2)
    
    goodMatch = []
    locations_1_to_use = []
    locations_2_to_use = []

    min_dist = 1000
    max_dist = 0
    disdif_avg = 0

    for m, n in matches:
        disdif_avg += n.distance - m.distance
    disdif_avg = disdif_avg / len(matches)
    
    for m, n in matches:
        #自适应阈值
        if n.distance > m.distance + disdif_avg:
            goodMatch.append(m)
            p2 = cv2.KeyPoint(kps_right[m.trainIdx][0],  kps_right[m.trainIdx][1],  1)
            p1 = cv2.KeyPoint(kps_left[m.queryIdx][0], kps_left[m.queryIdx][1], 1)
            locations_1_to_use.append([p1.pt[0], p1.pt[1]])
            locations_2_to_use.append([p2.pt[0], p2.pt[1]])
    #goodMatch = sorted(goodMatch, key=lambda x: x.distance)
    print('match num is %d' % len(goodMatch))
    locations_1_to_use = np.array(locations_1_to_use)
    locations_2_to_use = np.array(locations_2_to_use)
    
    # Perform geometric verification using RANSAC.
    _, inliers = measure.ransac((locations_1_to_use, locations_2_to_use),
                              transform.AffineTransform,
                              min_samples=3,
                              residual_threshold=_RESIDUAL_THRESHOLD,
                              max_trials=1000)
    
    print('Found %d inliers' % sum(inliers))
    
    inlier_idxs = np.nonzero(inliers)[0]

    return locations_1_to_use[inlier_idxs],locations_2_to_use[inlier_idxs]

