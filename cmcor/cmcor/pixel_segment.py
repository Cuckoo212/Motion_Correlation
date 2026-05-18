"""
Return a segment for grasping given a binary mask image and a pixel
within the mask.
"""
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt



def get_covariance_eigenvalues(points):
    x = points - np.mean(points, axis=0, keepdims=True)
    cov = x.T.dot(x)
    assert(cov.shape == (2,2))
    eigval, eigvec = np.linalg.eig(cov)
    cov_a = np.amax(eigval)
    cov_b = np.amin(eigval)
    return cov_a, cov_b


def pixel2segment(
        pixel, mask_image, region_radius=20, min_area=100, min_ecc=2.0,
        verbose=False):
    """Return a segment for grasping given a binary mask image and a pixel.

    Arguments:
    pixel -- 2D pixel coordinates in the mask image ([int, int])
    mask_image -- a binary mask image (numpy.ndarray, shape (height, width))
    region_radius -- the half of the side size of the sqaure region to
        construct around the pixel for cropping the grasping segment
        (int, default 20)
    min_area -- the minimum acceptable area of the returned grasping_segment
        in pixels (int, default 100)
    min_ecc -- the minimum eccentricity of the grasping segment
        (float, default 2.0), eccentricity = segment_length / segment_width,
        where segment_length >= segment_width and the length line is
        perpendicular to the width line.

    Returns:
    grasping_segment -- a binary image (numpy.ndarray, shape (height, width))
    """
    assert(pixel[0] >= 0)
    assert(pixel[1] >= 0)
    assert(pixel[0] < mask_image.shape[0])
    assert(pixel[1] < mask_image.shape[1])
    i0 = max(0, pixel[0] - region_radius)
    i1 = min(mask_image.shape[0], pixel[0] + region_radius)
    j0 = max(0, pixel[1] - region_radius)
    j1 = min(mask_image.shape[1], pixel[1] + region_radius)
    crop = mask_image[i0:i1,j0:j1]
    out = cv2.connectedComponentsWithStats(crop.astype(np.uint8))
    numLabels, labels, stats, centroids = out
    segment_mask = np.zeros(
        (mask_image.shape[0], mask_image.shape[1]), dtype=bool)
    for i in range(1,numLabels):
        if stats[i,cv2.CC_STAT_AREA] < min_area:
            if verbose:
                print("segment too small")
            continue
        segment_mask_crop = (labels == i)
        segment_mask[i0:i1,j0:j1] = segment_mask_crop
        if segment_mask[pixel[0], pixel[1]] == False:
            if verbose:
                print("pixel not in segment")
            continue
        points = np.argwhere(segment_mask_crop)
        cov_a, cov_b = get_covariance_eigenvalues(points)
        ecc = np.sqrt(cov_a/cov_b)
        if verbose:
            print(f"ecc {ecc}")
        if ecc < min_ecc:
            continue
        return segment_mask
    return None


def test_plot(mask_path, pixel, expect_success=True):
    print("=== Test starts. ===")
    mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    segment_mask = pixel2segment(pixel, mask_image, verbose=True)
    if expect_success:
        assert(segment_mask is not None)
        plt.imshow(mask_image)
        plt.imshow(segment_mask, alpha=0.7)
        plt.show()
    else:
        assert(segment_mask is None)


def main_test():
    data_root = "test_data/mcor_outputs/2024-08-06-d-shaped-combo/"
    mask_path = os.path.join(data_root, "corr_0_1.png")
    pixel = [407, 768]
    test_plot(mask_path, pixel)
    pixel = [348, 765]
    test_plot(mask_path, pixel)
    data_root = "test_data/mcor_outputs/2024-08-06-z-shaped-combo/"
    mask_path = os.path.join(data_root, "corr_1_0.png")
    pixel = [462, 645]
    test_plot(mask_path, pixel)
    pixel = [281, 761]
    test_plot(mask_path, pixel)
    pixel = [271, 750]
    test_plot(mask_path, pixel, expect_success=False)
    pixel = [255, 774]
    test_plot(mask_path, pixel, expect_success=False)
    pixel = [279, 756]
    test_plot(mask_path, pixel)
    data_root = "test_data/pixel_segment_test"
    mask_path = os.path.join(data_root, "corr_1_1.png")
    pixel = [404, 658]
    test_plot(mask_path, pixel, expect_success=False)
    pixel = [387, 773]
    test_plot(mask_path, pixel, expect_success=False)



if __name__=="__main__":
    main_test()
