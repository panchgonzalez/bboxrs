# `bboxrs`

A reimplementation of [`cython_bbox`](https://github.com/samson-wang/cython_bbox) in Rust. It uses the [`pyo3`](https://pyo3.rs/v0.20.3/) crate to interface with Python.



```python
from bboxrs import bbox_overlaps

import numpy as np

gt = np.random.random((5, 4)).astype(float)
dt = np.random.random((10, 4)).astype(float)

overlaps = bbox_overlaps(dt, gt)
```

**Disclaimer**
- This is about an order of magnitude slower than the Cython implementation. Probably because I had to do a couple of unnecessary conversions. This is really just an exercise in learning Rust and interfacing with Python.