//! Integration tests for the 2-dimensional convolution on all defined 2-dimensional filters.
//!
extern crate sigpro;
extern crate ndarray;

use ndarray::prelude::*;
use sigpro::*;

#[test]
fn conv2d_sharpen() {
    let n = 16;
    let mut a = Array::zeros((n, n));
    let b = filters::sharpen();
    // make a circle
    let c = (8., 8.);
    for ((i, j), elt) in a.indexed_iter_mut() {
        {
            let s = ((i as f32) - c.0).powi(2) + (j as f32 - c.1).powi(2);
            if s.sqrt() > 3. && s.sqrt() < 6. {
                *elt = 1.;
            }
        }
    }
    println!("{:2}", a);
    let mut res = Array::zeros(a.dim());
    for _ in 0..10 {
        convolution::conv2d(&a.view(), &b.view(), &mut res.view_mut());
    }
    println!("{:2}", res);
    //assert_eq!(a.max(), res.max())
}


#[test]
fn conv2d_sobel() {
    let n = 16;
    let mut a = Array::zeros((n, n));
    let b_x = filters::sobel_x();
    let b_y = filters::sobel_y();
    // make a circle
    let c = (8., 8.);
    for ((i, j), elt) in a.indexed_iter_mut() {
        {
            let s = ((i as f32) - c.0).powi(2) + (j as f32 - c.1).powi(2);
            if s.sqrt() > 3. && s.sqrt() < 6. {
                *elt = 1.;
            }
        }
    }
    println!("{:2}", a);
    let mut res_x = Array::zeros(a.dim());
    let mut res_y = Array::zeros(a.dim());
    convolution::conv2d(&a.view(), &b_x.view(), &mut res_x.view_mut());
    convolution::conv2d(&a.view(), &b_y.view(), &mut res_y.view_mut());
    let res = res_x + res_y;
    println!("{:2}", res);
    //assert_eq!(a.max(), res.max())
}
