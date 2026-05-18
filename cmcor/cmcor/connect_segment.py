import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from scipy.ndimage import convolve
from scipy.interpolate import make_interp_spline, make_splprep, splev
import numpy as np


MIN_AREA = 50
MIN_LENGTH = 3
MIN_STRETCH = 2


def find_endpoints(skeleton):
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0  # Ignore the center pixel
    neighbor_count = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
    return np.argwhere((skeleton) & (neighbor_count == 1))  # Endpoints have 1 neighbor


def trace_skeleton_line(path):
    max_x, max_y = np.max(path, axis=0)
    mask = np.zeros((max_x + 10, max_y + 10), dtype=bool)
    mask[path[:, 0], path[:, 1]] = True
    endpoints = find_endpoints(mask)
    ordered_points = trace_skeleton(mask, endpoints)
    return ordered_points


def longest_path_in_skeleton(skeleton):
    endpoints = find_endpoints(skeleton)
    if len(endpoints) == 0:
        return np.array([])  # No path found (e.g., circular skeleton)

    # Build adjacency list
    skeleton_points = set(map(tuple, np.argwhere(skeleton)))
    adjacency = {}
    for (y, x) in skeleton_points:
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                if (y + dy, x + dx) in skeleton_points:
                    neighbors.append((y + dy, x + dx))
        adjacency[(y, x)] = neighbors

    # BFS to find the farthest node from an arbitrary endpoint
    def bfs(start):
        visited = {start: None}
        queue = [start]
        farthest_node = start
        while queue:
            current = queue.pop(0)
            for neighbor in adjacency.get(current, []):
                if neighbor not in visited:
                    visited[neighbor] = current
                    queue.append(neighbor)
                    farthest_node = neighbor
        return farthest_node, visited

    # First BFS to find one end of the longest path
    start_node = tuple(endpoints[0])
    end_node, _ = bfs(start_node)
    # Second BFS to find the other end and the path
    true_end, parents = bfs(end_node)

    # Reconstruct the longest path
    path = []
    current = true_end
    while current is not None:
        path.append(current)
        current = parents.get(current)
    return np.array(path[::-1])  # Order from start to end


def smooth_path(longest_path, segment_mask, smoothing_factor=3.0):
    n_points = len(longest_path)

    if len(longest_path) == 0:
        return np.array([])

    sorted_y, sorted_x = trace_skeleton_line(longest_path).T

    # Fit parametric spline
    bsp, _ = make_splprep(np.array([sorted_x, sorted_y]), s=smoothing_factor)
    new_points = bsp(np.linspace(0, 1, n_points))
    return new_points


def trace_segment(prop, img_shape):
    segment_mask = np.zeros(img_shape, dtype=bool)
    segment_mask[prop.coords[:, 0], prop.coords[:, 1]] = True

    skeleton = morphology.skeletonize(segment_mask)
    longest_path_coords = longest_path_in_skeleton(skeleton)
    smoothed_path = smooth_path(longest_path_coords, segment_mask, smoothing_factor=2.0)
    smoothed_path = np.round(smoothed_path).astype(int).T

    if len(smoothed_path) < 2:
        return None, None

    return smoothed_path[:, [1, 0]]


def find_endpoints(skeleton):
    # k_size = 3
    # kernel = np.ones((k_size, k_size), np.uint8)
    # kernel[k_size // 2 + 1, k_size // 2 + 1] = 0
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    conv = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    return np.argwhere((conv == 1) & (skeleton))


def trace_skeleton(skeleton, endpoints):
    if len(endpoints) < 2:
        return []

    # (y, x) coords
    endpoints = [tuple(e) for e in endpoints]

    # first endpoint
    current = endpoints[0]
    ordered = [current]
    visited = set([current])

    while True:
        # 8-connected neighbors
        y, x = current
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and skeleton[ny, nx] and (ny, nx) not in visited:
                    neighbors.append((ny, nx))

        if not neighbors:
            break

        current = neighbors[0]
        ordered.append(current)
        visited.add(current)

    return np.array(ordered)


def validate_segment(prop):
    major = prop.major_axis_length
    minor = prop.minor_axis_length
    stretch = major / minor if minor > 0 else 0
    return (prop.area >= MIN_AREA and major >= MIN_LENGTH and stretch >= MIN_STRETCH)


def fit_bspline(points):
    points = np.array(points)
    t = np.linspace(0, 1, len(points))
    spl_x = make_interp_spline(t, points[:, 0], k=3)
    spl_y = make_interp_spline(t, points[:, 1], k=3)

    t_new = np.linspace(0, 1, 1000)
    return np.column_stack([spl_x(t_new), spl_y(t_new)])


def intergroup_distance(query, ug, allow_query_rotation=False):
    compare_idx = np.array([    # order: query, ug
        [-1,0],
        [-1,-1],
        [0,-1],
        [0,0],
        ])
    query_rotations = [False, False, True, True]
    ug_rotations = [False, True, True, False]
    if not allow_query_rotation:
        compare_idx = compare_idx[0:2,:]
        query_rotations = query_rotations[0:2]
        ug_rotations = ug_rotations[0:2]
    diffs = query[compare_idx[:,0],:]-ug[compare_idx[:,1],:]
    distances = np.sqrt(diffs[:,0]**2+diffs[:,1]**2)
    min_idx = np.argmin(distances)
    return distances[min_idx], query_rotations[min_idx], ug_rotations[min_idx]


def assign_next_group(
        ordered_groups, unassigned_groups, allow_query_rotation):
    query = ordered_groups[-1]
    best_group = None
    best_distance = None
    best_rotations = None
    for idx, ug in enumerate(unassigned_groups):
        dist, rotate_query, rotate_ug = intergroup_distance(
            query, ug, allow_query_rotation)
        if best_distance is None or dist < best_distance:
            best_distance = dist
            best_group_idx = idx
            best_rotations = [rotate_query, rotate_ug]
    rotate_query, rotate_best = best_rotations
    if rotate_query:
        ordered_groups[-1] = ordered_groups[-1][::-1,:]
    ug = unassigned_groups[best_group_idx]
    if rotate_best:
        ug = ug[::-1,:]
    ordered_groups.append(ug)
    del unassigned_groups[best_group_idx]


def reorder_point_groups(point_groups):
    if not point_groups:
        return point_groups
    ordered_groups = [point_groups[0]]
    unassigned_groups = point_groups[1:]
    first = True
    while len(ordered_groups) != len(point_groups):
        assign_next_group(
            ordered_groups, unassigned_groups, allow_query_rotation=first)
        first = False
    all_points = []
    for group in ordered_groups:
        all_points.extend(group)
    return all_points


def process_mask_with_reordering(mask):
    labels = measure.label(mask)
    props = measure.regionprops(labels)

    point_groups = []
    for prop in props:
        # skip small segments
        if not validate_segment(prop):
            continue
        points = trace_segment(prop, mask.shape)
        if points is not None and len(points) >= 2:
            # Convert to (x, y) format
            point_groups.append(points[:, [1, 0]])
    all_points = reorder_point_groups(point_groups)
    if all_points:
        return fit_bspline(all_points)
    else:
        return []

