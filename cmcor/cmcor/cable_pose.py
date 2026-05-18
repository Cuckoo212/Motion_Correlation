"""
Estimate the cable segment pose in a depth image.

Requires:
depth image
cable segmentation mask
camera matrix

The main functions are:

center, axis, points = find_cable_pose_depth(
    depth, K, mask, inlier_dist=0.04)

center, axis, points = find_cable_pose(points, inlier_dist=0.04)
"""

import os
import json
import functools

import cv2
import numpy as np
import matplotlib.pyplot as plt



@functools.lru_cache(maxsize=16)
def precompute_grid(depth_shape):
    ui = np.arange(0, depth_shape[1])
    vi = np.arange(0, depth_shape[0])
    v,u = np.meshgrid(vi, ui, indexing='ij')
    return u,v


def depth2points(depth, K, mask, u, v, max_points=500):
    """Extract a point cloud given a depth image and a mask.

    Arguments:
    depth - depth image, depth in millimeters
    mask - a binary image to select pixels from the depth image
    K - camera matrix
    u, v - precomputed pixel grid matching the depth image
    max_points - extract at most this amount of points (randomly subsample)

    Returns:
    xyz - 3D point cloud with coordinates in meters
    """
    #print(depth.shape)
    # return numpy array (N, 3)
    u_sel = u[mask].flatten()
    v_sel = v[mask].flatten()
    z_sel = depth[mask].flatten()
    n_sel = np.ones(z_sel.size)
    valid = z_sel > 0
    u_sel = u_sel[valid]
    v_sel = v_sel[valid]
    z_sel = z_sel[valid]
    n_sel = n_sel[valid]
    if u_sel.size > max_points:
        rng = np.random.default_rng()
        idx = rng.choice(u_sel.size, size=max_points, replace=False)
        u_sel = u_sel[idx]
        v_sel = v_sel[idx]
        z_sel = z_sel[idx]
        n_sel = n_sel[idx]
    z_sel = z_sel.astype(np.float32) / 1000.0
    #print(u_sel.shape, v_sel.shape, n_sel.shape)
    uvn = np.concatenate(
        (u_sel[None,:], v_sel[None,:], n_sel[None,:]), axis=0)
    K_inv = np.linalg.inv(K)
    xyn = K_inv.dot(uvn)
    xyn[0:2,:] = xyn[0:2,:]/xyn[None,2,:]
    xyz = xyn * z_sel[None,:]
    return xyz.T


def geometric_median(points):
    """
    https://en.wikipedia.org/wiki/Geometric_median
    """
    gm = np.mean(points, axis=0)
    min_change = 0.0005
    eps = 1e-4
    for i in range(20):
        gm_prev = gm
        dists = np.linalg.norm(points - gm_prev[None,:], axis=1)
        valid = dists > eps
        weights_valid = 1.0/dists[valid]
        w_sum = np.sum(weights_valid)
        assert(w_sum > eps)
        weights_valid = weights_valid/w_sum
        gm = np.sum(points[valid,:]*weights_valid[:,None],axis=0)
        change = np.linalg.norm(gm-gm_prev)
        if change < min_change:
            break
    #print(f"geometric_median ended after {i} iterations")
    return gm


def get_principal_axis(points):
    x = points - np.mean(points, axis=0, keepdims=True)
    cov = x.T.dot(x)
    assert(cov.shape == (3,3))
    eigval, eigvec = np.linalg.eig(cov)
    i_max = np.argmax(eigval)
    return eigvec[:, i_max]


def find_cable_pose(points, inlier_dist=0.04):
    center = geometric_median(points)
    dists = np.linalg.norm(points - center[None,:], axis=1)
    points = points[dists<inlier_dist,:]
    axis = get_principal_axis(points)
    return center, axis, points


def find_cable_pose_depth(depth, K, mask, inlier_dist=0.04):
    """Find cable pose in a depth image.

    Arguments:
    depth - depth image
    K - camera matrix
    mask - cable segment segmentation mask
        (a binary image of the same shape as the depth image)

    Returns:
    center - the center point of the cable segment
    axis - a vector pointing along the cable segment axis
    points - the pointcloud of the cable segment
    """
    assert(mask.shape == depth.shape)
    u,v = precompute_grid(depth.shape)
    if mask.dtype != bool:
        mask = mask > 250
    points = depth2points(depth, K, mask, u, v)
    if points.size < 30:
        return None
    center, axis, points = find_cable_pose(points, inlier_dist)
    return center, axis, points


def plot_points(points, center=None, axis=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    if center is not None:
        ax.scatter(center[0], center[1], center[2])
        if axis is not None:
            ax.plot(
                [center[0], center[0]+axis[0]],
                [center[1], center[1]+axis[1]],
                [center[2], center[2]+axis[2]],
                c="tab:orange")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def sample_test_data(rng):
    """
    X - points to the right
    Y - points down
    Z - points horizontally forward (away from the camera)
    """
    radius = 0.00752
    length = 0.03
    th_min = np.pi
    th_max = 2*np.pi
    N_points = rng.integers(90, 111)
    y_min = -0.5*length
    y_max = 0.5*length
    z_noise = 0.005
    points = np.zeros((N_points, 3))
    points[:,1] = length*rng.random(N_points) + y_min
    th = (th_max-th_min)*rng.random(N_points) + th_min
    points[:,0] = radius*np.cos(th)
    points[:,2] = radius*np.sin(th)
    points[:,2] += rng.uniform(-z_noise, z_noise, size=N_points)
    return points


def synthetic_test():
    rng = np.random.default_rng(seed=82546835)
    points = sample_test_data(rng)
    center = geometric_median(points)
    axis = get_principal_axis(points)
    plot_points(points, center, 0.015*axis)


def load_camera_matrix(fn_parameters):
    with open(fn_parameters) as f:
        parameters = json.load(f)
    camera_matrix = np.array(parameters["camera_matrix"])
    return camera_matrix


def real_test():
    inlier_dist = 0.04
    test_root = "test_data/2024-03-12-125440"
    fn_params = os.path.join(test_root, "parameters.json")
    fn_depth = os.path.join(test_root, "depth.png")
    folder_masks = test_root
    K = load_camera_matrix(fn_params)
    depth = cv2.imread(fn_depth, cv2.IMREAD_UNCHANGED)
    u,v = precompute_grid(depth.shape)
    filenames = os.listdir(folder_masks)
    for name in sorted(filenames):
        if not name.startswith("mask_"):
            continue
        if not name.endswith(".png"):
            continue
        fn_mask = os.path.join(folder_masks, name)
        if not os.path.isfile(fn_mask):
            continue
        mask = cv2.imread(fn_mask, cv2.IMREAD_UNCHANGED)
        assert(mask.shape == depth.shape)
        mask = mask > 250
        points = depth2points(depth, K, mask, u, v)
        center, axis, points = find_cable_pose(points, inlier_dist)
        plot_points(points, center, 0.03*axis)


def main_test():
    real_test()
    #synthetic_test()
    

if __name__ == "__main__":
    main_test()
