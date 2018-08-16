extern crate rustfft as fft;
extern crate ndarray;
extern crate ndarray_linalg;
extern crate num_traits;

// submodules
mod convolution;
mod padding;

#[cfg(test)]
mod tests {
    #[test]
    fn dummy() {
        println!("dummy test", );
        assert_eq!(2 + 2, 4);
    }
}
