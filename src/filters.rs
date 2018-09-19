//! Collection of 1d and 2d filters, both parametrized and fixed.
//!

use ndarray::prelude::*;
use std::f64::consts::PI;

/// Normalized box Blur filter
/// every element (1.0) is normalized by the sum of total elements (dim^2*1.0)
/// argument dim is used for both dimensions due to the filters quadratic shape
pub fn blur2d(dim: usize) -> Array2<f64> {
    let filt: Array2<f64> = Array2::ones((dim, dim)) * (1.0/(dim as f64*dim as f64));
    filt
}


/// 2D Gaussian Filter for image blurring/smoothing
/// currently uses square support pattern
pub fn gaussian2d(dim: usize, std: f64) -> Array2<f64> {

    let mut filt: Array2<f64> = Array2::zeros((dim, dim));
    let mut sum: f64 = 0.;
    // exponent denominator (doesnt change)
    let s: f64 = 2. * std * std;
    // exponent nominator (updated every pixel)
    let mut r: f64 = 0.;
    // offset for centered kernel
    let off: i64 = (dim as f64/2.).floor() as i64;
    // temporary vars for safe array bounds
    let mut xp: usize = 0;
    let mut yp: usize = 0;

    for x in -off..off-1 {
        for y in -off..off-1 {
            // this looks ugly, better way to deal with overflows safely?
            xp = (x as usize).checked_add(off as usize).unwrap_or(dim-1);
            yp = (y as usize).checked_add(off as usize).unwrap_or(dim-1);
            // compute the values of the gaussian function
            r = ((x*x + y*y) as f64).sqrt();
            filt[[xp, yp]] = ( ((-(r * r) / s)) / (PI * s) ).exp();
            sum += filt[[xp, yp]];
        }
    }
    // normalize the kernel
    for i in 0..dim {
        for j in 0..dim {
            filt[[i, j]] /= sum;
        }
    }
    filt
}


/// Basic sharpening filter to sharpen edge content of an image
pub fn sharpen2d() -> Array2<f64> {
    let filt: Array2<f64> = array![ [ 0., -1.,  0.],
                                    [-1.,  5., -1.],
                                    [ 0., -1.,  0.] ];
    filt.clone()
}


/// Sobel filter for edge detection in X direction
pub fn sobel_x() -> Array2<f64> {
    let filt: Array2<f64> = array![ [-1., 0., 1.],
                                    [-2., 0., 2.],
                                    [-1., 0., 1.] ];
    filt.clone()
}


/// Sobel filter for edge detection in Y direction
pub fn sobel_y() -> Array2<f64> {
    let filt: Array2<f64> = array![ [ 1.,  2.,  1.],
                                    [ 0.,  0.,  0.],
                                    [-1., -2., -1.] ];
    filt.clone()
}
