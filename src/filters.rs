//! Collection of 1d and 2d filters, both parametrized and fixed.
//!

use ndarray::prelude::*;

const SOBEL_X: [[f32; 3]; 3] = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]];
const SOBEL_Y: [[f32; 3]; 3] = [[ 1., 2., 1.], [ 0., 0., 0.], [-1., -2., -1.]];
const SHARPEN: [[f32; 3]; 3] = [[0., -1., 0.], [ -1., 5., -1.], [0., -1., 0.]];

pub fn blur_box(dim: usize) -> Array2<f64> {
    let filt: Array2<f64> = Array2::ones((dim, dim)) * (1.0/(dim as f64*dim as f64));
    filt
}

pub fn sharpen() -> Array2<f64> {
    let filt: Array2<f64> = array![ [ 0., -1.,  0.],
                                    [-1.,  5., -1.],
                                    [ 0., -1.,  0.] ];
    filt.clone()
}

pub fn sobel_x() -> Array2<f64> {
    let filt: Array2<f64> = array![ [-1., 0., 1.],
                                    [-2., 0., 2.],
                                    [-1., 0., 1.] ];
    filt.clone()
}

pub fn sobel_y() -> Array2<f64> {
    let filt: Array2<f64> = array![ [ 1.,  2.,  1.],
                                    [ 0.,  0.,  0.],
                                    [-1., -2., -1.] ];
    filt.clone()
}
