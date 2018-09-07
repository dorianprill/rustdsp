//! rustdsp - digital signal processing in rust
//! module: convolution.rs
//! info:   implementations of convolution operations for general 1d and 2d cases
//!         1d is based on Vec<> and 2d is based on ndarray

use num_traits::Float;
use ndarray::prelude::*;
use std::f64;

#[derive(PartialEq)]
pub enum ConvMode {
    Full,       // output length (N+M-1)
    Same,       // output length max(M, N)
    Valid       // output length max(M, N) - min(M, N) + 1
}

/// calculate the concolution y = convolve(x,h)
/// y(i) = sum_k{ x[i-k] * h[k]}
/// offer three modes 'full', 'same', and 'valid'
pub fn conv(input_a: &Vec<f64>, input_b: &Vec<f64>, mode: ConvMode) -> Vec<f64> {
    if mode == ConvMode::Full {
        return conv_full(input_a, input_b);
    } else { // dummy to shut up the compiler until other modes get implemented
        return conv_full(input_a, input_b);
    }
}


pub fn xcorr(a: &Vec<f64>, b: &Vec<f64>, mode: ConvMode) -> Vec<f64> {
    let brev = b.clone().into_iter().rev().collect();
    if mode == ConvMode::Full {
        return conv(a, &brev, ConvMode::Full);
    } else { // dummy to shut up the compiler until other modes get implemented
        return conv(a, &brev, ConvMode::Full);
    }
}


pub fn autocorr(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    let brev = b.clone().into_iter().rev().collect();
    let res = conv(a, &brev, ConvMode::Full);
    let mid: usize = ((res.len() as f64)/2.).floor() as usize;
    return res[mid..].to_vec()
}


/// calculate the convolution y = x * h
/// in line with its mathematical definition (reversed impulse response)
/// results in better caching because each element in y is only written once
/// implementation for convolution mode 'full' where a single point overlap suffices
/// mode is the same as in numpy.convolve
fn conv_full(input_x: &Vec<f64>, input_h: &Vec<f64>) -> Vec<f64> {
    let mut out_y: Vec<f64> = vec![0f64; (input_x.len() + input_h.len() - 1) as usize];
    for i in 0..out_y.len() {
        out_y[i] = 0.0; // implicit padding
        let start: usize = if i >= input_x.len() {i - input_x.len() + 1 } else {0};
		let end:   usize = if i < input_h.len() {i+1} else {input_h.len()};
        for k in start..end {
            out_y[i] += input_x[i-k] * input_h[k];
        }
    }
    out_y
}


/// calculate the convolution y = x * h
/// alternative implementation to conv_full
/// has worse caching behaviour because each element in y is written multiple times
/// maybe better for parallelization (?)
fn conv_full_scatter(input_a: &Vec<f64>, input_b: &Vec<f64>) -> Vec<f64> {
    let mut out: Vec<f64> = vec![0f64; (input_a.len() + input_b.len() - 1) as usize];
    for (a_i, &a) in input_a.iter().enumerate() {
        for (b_i, &b) in input_b.iter().enumerate() {
            out[a_i+b_i] += a*b;
        }
    }
    out
}



//#[inline(never)]
pub fn conv2d<F>(a: &ArrayView2<F>, b: &ArrayView2<F>, out: &mut ArrayViewMut2<F>)
    where F: Float,
{
    let (na, ma) = a.dim();
    let (nb, mb) = b.dim();
    let (np, mp) = out.dim();
    let noff: usize = b.dim().0; // border regions are excluded for now, need padding!
    let moff: usize = b.dim().1;
    // check if image is smaller than the kernel in any dim
    if na < nb || ma < mb {
        return;
    }
    assert!(np >= na && mp >= ma && np >= nb && mp >= mb);
        for i in 0..na - noff {
            for j in 0..ma - moff {
                let mut conv = F::zero();
                for k in 0..nb {
                    for l in 0..mb {
                        conv = conv + a[[i + k, j + l]] * b[[k, l]];
                    }
                }
                out[[i + 1, j + 1]] = conv;
            }
        }
}


#[cfg(test)]
mod tests {
    use ndarray::prelude::*;
    use convolution::conv;
    use convolution::xcorr;
    use convolution::autocorr;
    use convolution::ConvMode;
    use convolution::conv2d;

    #[test]
    /// calculate a known impulse response in all different convolution modes and check the output
    fn conv_xcorr_autocorr_1d() {
        let x:     Vec<f64> = vec![3.0, 4.0, 5.0];         // excitation signal
        let h:     Vec<f64> = vec![2.0, 1.0];              // impulse response of dummy system
        let y:     Vec<f64> = vec![6.0, 11.0, 14.0, 5.0];  // expected output y=conv(x,h)
        let yy:    Vec<f64> = vec![30., 139., 290., 378., 290., 139.,  30.];
        let res:   Vec<f64> = conv(&x,&h, ConvMode::Full);
        let resyy: Vec<f64> = xcorr(&y, &y, ConvMode::Full);
        assert_eq!(res, y);
        assert_eq!(resyy, yy);
        assert_eq!(resyy[3..].to_vec(), autocorr(&y, &y))
    }


    #[test]
    fn conv2d_uniform_rect_smoothing() {
        let n = 16;
        let mut a = Array::zeros((n, n));
        let b = Array::ones((5, 5)) * (1.0/9.0); // create a normalized uniform rectangular filter (smoothing)
        //make a circle
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
        conv2d(&a.view(), &b.view(), &mut res.view_mut());
        println!("{:2}", res);
        //assert_eq!(a.max(), res.max())
    }
}
