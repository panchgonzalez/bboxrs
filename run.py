import time
from functools import wraps

import numpy as np
import matplotlib.pyplot as plt

from bboxrs import bbox_overlaps
from cython_bbox import bbox_overlaps as bbox_overlaps_cython_


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start} seconds")
        return result

    return wrapper


@timeit
def bbox_overlaps_cython(dt, gt):
    return bbox_overlaps_cython_(dt, gt)


@timeit
def bbox_overlaps_rust(dt, gt):
    return bbox_overlaps(dt, gt)


@timeit
def bbox_overlaps_numpy(dt, gt):
    """Loop over dt and gt and compute overlaps"""
    num_dt = dt.shape[0]
    num_gt = gt.shape[0]
    overlaps = np.zeros((num_dt, num_gt), dtype=np.float32)
    for i in range(num_dt):
        bbox_area = (dt[i, 2] - dt[i, 0] + 1) * (dt[i, 3] - dt[i, 1] + 1)
        for j in range(num_gt):
            iw = min(dt[i, 2], gt[j, 2]) - max(dt[i, 0], gt[j, 0]) + 1
            if iw > 0:
                ih = min(dt[i, 3], gt[j, 3]) - max(dt[i, 1], gt[j, 1]) + 1
                if ih > 0:
                    gt_area = (gt[j, 2] - gt[j, 0] + 1) * (gt[j, 3] - gt[j, 1] + 1)
                    ua = float(bbox_area + gt_area - iw * ih)
                    overlaps[i, j] = iw * ih / ua
    return overlaps

gt = np.random.random((100, 4)).astype(float)

dt_sizes = np.unique(np.logspace(3, 7, num=10, dtype=int)).tolist()
cython_times = []
rust_times = []
# numpy_times = []

for dt_size in dt_sizes:
    dt = np.random.random((dt_size, 4)).astype(float)

    start = time.time()
    cython_overlaps = bbox_overlaps_cython(dt, gt)
    cython_times.append(time.time() - start)

    start = time.time()
    rust_overlaps = bbox_overlaps_rust(dt, gt)
    rust_times.append(time.time() - start)

    # start = time.time()
    # numpy_overlaps = bbox_overlaps_numpy(dt, gt)
    # numpy_times.append(time.time() - start)

    # diff = np.abs(cython_overlaps - rust_overlaps)
    # print(f"dt_size={dt_size}")
    # print("  Max difference between cython and rust overlaps:", np.max(diff))
    # print("  Mean difference between cython and rust overlaps:", np.mean(diff))
    # print("  Number of elements with difference > 1e-6:", np.sum(diff > 1e-6))
    # assert np.allclose(cython_overlaps, rust_overlaps)

plt.xkcd()
plt.figure(figsize=(10, 6))
plt.plot(dt_sizes, cython_times, label="cython_bbox")
plt.plot(dt_sizes, rust_times, label="bboxrs (Rust)")
# plt.plot(dt_sizes, numpy_times, label="numpy (python loop)")
plt.xlabel("dt boxes")
plt.ylabel("seconds")
plt.title("bbox_overlaps runtime")
plt.legend()
plt.grid(True)
plt.tight_layout()  # Adjust layout to prevent cutoff
plt.savefig("doc/speedup.png")  # Save the plot to a file
