
#[macro_use]
    extern crate ndarray;
//    extern crate ndarray_linalg;
//    extern crate rustfft as fft;
pub extern crate num_traits;
pub extern crate num_complex;

// submodules

pub mod convolution;
pub mod padding;
pub mod signal;
pub mod filters;

#[cfg(test)]
mod tests {
    #[test]
    fn dummy() {
        println!("dummy test", );
        assert_eq!(2 + 2, 4);
    }
}
