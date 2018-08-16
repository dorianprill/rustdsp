// dspr - digital signal processing in rust
// module: convolution.rs
// info:   implementations of convolution operations for general 1d and 2d cases
//         1d is based on Vec<> and 2d is based on ndarray


use num_traits::Float;
use ndarray::prelude::*;


pub fn convolve(input_a: &Vec<f64>, input_b: &Vec<f64>) -> Vec<f64> {
    let mut vec: Vec<f64> = vec![0f64; (input_a.len() + input_b.len() - 1) as usize];
    for (a_i, &a) in input_a.iter().enumerate() {
        for (b_i, &b) in input_b.iter().enumerate() {
            vec[a_i+b_i] += a*b;
        }
    }
    vec
}



#[inline(never)]
pub fn convolve2d<F>(a: &ArrayView2<F>, b: &ArrayView2<F>, out: &mut ArrayViewMut2<F>)
    where F: Float,
{
    let (na, ma) = a.dim();
    let (nb, mb) = b.dim();
    let (np, mp) = out.dim();
    // check if image is smaller than the kernel in any dim
    if na < nb || ma < mb {
        return;
    }
    assert!(np >= na && mp >= ma && np >= nb && mp >= mb);
    // i, j offset by -1 so that we can use unsigned indices
    unsafe {
        for i in 0..na - 2 {
            for j in 0..ma - 2 {
                let mut conv = F::zero();
                for k in 0..nb {
                    for l in 0..mb {
                        conv = conv + *a.uget((i + k, j + l)) * *b.uget((k, l));
                        //conv += a[[i + k, j + l]] * x_kernel[k][l];
                    }
                }
                *out.uget_mut((i + 1, j + 1)) = conv;
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use ndarray::prelude::*;
    use convolution::convolve;
    use convolution::convolve2d;

    #[test]
    fn conv1d_impulse_response() {
        let x: Vec<f64>   = vec![3.0, 4.0, 5.0]; // excitation signal
        let h: Vec<f64>   = vec![2.0, 1.0, 0.0];    // impulse response of dummy system
        let y: Vec<f64>   = vec![6.0, 11.0, 14.0, 5.0, 0.0]; // system output
        let res: Vec<f64> = convolve(&x,&h);
        // This loop prints: 0 1 2
        for r in &res {
            print!("{} ", r);
        }
        assert_eq!(res, y);
    }


    #[test]
    fn conv2d_uniform_rect_smoothing_3x3() {
        let n = 16;
        let mut a = Array::zeros((n, n));
        let b = Array::ones((3, 3)) * (1.0/9.0); // create a normalized uniform rectangular filter (smoothing)
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
        for _ in 0..100 {
            convolve2d(&a.view(), &b.view(), &mut res.view_mut());
        }
        println!("{:2}", res);
        //assert_eq!(a.max(), res.max())
    }
}
