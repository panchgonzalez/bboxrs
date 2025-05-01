use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
fn bbox_overlaps<'py>(
    boxes: PyReadonlyArray2<'py, f64>,
    query_boxes: PyReadonlyArray2<'py, f64>,
    py: Python<'py>,
) -> &'py PyArray2<f64> {
    let boxes = boxes.as_array();
    let query_boxes = query_boxes.as_array();

    let n = boxes.shape()[0];
    let k = query_boxes.shape()[0];

    let boxes = boxes.as_slice().unwrap();
    let query_boxes = query_boxes.as_slice().unwrap();

    let box_areas: Vec<f64> = (0..n)
        .map(|i| {
            let x1 = boxes[i * 4 + 0];
            let y1 = boxes[i * 4 + 1];
            let x2 = boxes[i * 4 + 2];
            let y2 = boxes[i * 4 + 3];
            (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
        })
        .collect();

    let query_areas: Vec<f64> = (0..k)
        .map(|i| {
            let x1 = query_boxes[i * 4 + 0];
            let y1 = query_boxes[i * 4 + 1];
            let x2 = query_boxes[i * 4 + 2];
            let y2 = query_boxes[i * 4 + 3];
            (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
        })
        .collect();

    // Allocate output in row-major order
    let mut overlaps = vec![0.0f64; n * k];

    // Parallelize over rows (boxes)
    overlaps
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(n_idx, overlaps_row)| {
            let bx1 = boxes[n_idx * 4 + 0];
            let by1 = boxes[n_idx * 4 + 1];
            let bx2 = boxes[n_idx * 4 + 2];
            let by2 = boxes[n_idx * 4 + 3];
            let barea = box_areas[n_idx];

            for k_idx in 0..k {
                let qx1 = query_boxes[k_idx * 4 + 0];
                let qy1 = query_boxes[k_idx * 4 + 1];
                let qx2 = query_boxes[k_idx * 4 + 2];
                let qy2 = query_boxes[k_idx * 4 + 3];
                let qarea = query_areas[k_idx];

                let iw = f64::min(bx2, qx2) - f64::max(bx1, qx1) + 1.0;
                if iw > 0.0 {
                    let ih = f64::min(by2, qy2) - f64::max(by1, qy1) + 1.0;
                    if ih > 0.0 {
                        let ua = barea + qarea - iw * ih;
                        overlaps_row[k_idx] = iw * ih / ua;
                    }
                }
            }
        });

    let overlaps = ndarray::Array2::from_shape_vec((n, k), overlaps).unwrap();
    overlaps.into_pyarray(py)
}

#[pymodule]
fn bboxrs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bbox_overlaps, m)?)?;
    Ok(())
}
