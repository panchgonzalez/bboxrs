use pyo3::prelude::*;
extern crate ndarray;
extern crate numpy;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArrayDyn};

#[pyfunction]
fn bbox_overlaps<'a>(
    boxes: PyReadonlyArrayDyn<'a, f64>,
    query_boxes: PyReadonlyArrayDyn<'a, f64>,
    py: Python<'a>,
) -> &'a PyArray2<f64> {
    // Convert to array
    let boxes = boxes.as_array();
    let query_boxes = query_boxes.as_array();

    // Get the number of boxes and query_boxes
    let n = boxes.shape()[0];
    let k = query_boxes.shape()[0];

    // Initialize overlaps array of size [n, k]
    let mut overlaps = Array2::<f64>::zeros((n, k));

    // Loop through the boxes and query boxes
    for k_idx in 0..k {
        let bbox_area = (query_boxes[[k_idx, 2]] - query_boxes[[k_idx, 0]] + 1.0)
            * (query_boxes[[k_idx, 3]] - query_boxes[[k_idx, 1]] + 1.0);

        for n_idx in 0..n {
            let iw = f64::min(boxes[[n_idx, 2]], query_boxes[[k_idx, 2]])
                - f64::max(boxes[[n_idx, 0]], query_boxes[[k_idx, 0]])
                + 1.0;
            if iw > 0.0 {
                let ih = f64::min(boxes[[n_idx, 3]], query_boxes[[k_idx, 3]])
                    - f64::max(boxes[[n_idx, 1]], query_boxes[[k_idx, 1]])
                    + 1.0;
                if ih > 0.0 {
                    let ua = (boxes[[n_idx, 2]] - boxes[[n_idx, 0]] + 1.0)
                        * (boxes[[n_idx, 3]] - boxes[[n_idx, 1]] + 1.0)
                        + bbox_area
                        - iw * ih;
                    overlaps[[n_idx, k_idx]] = iw * ih / ua;
                }
            }
        }
    }

    overlaps.into_pyarray(py)
}

/// A Python module implemented in Rust.
#[pymodule]
fn bboxrs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bbox_overlaps, m)?)?;
    Ok(())
}
