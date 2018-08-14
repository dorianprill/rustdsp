extern crate rustfft as fft;
extern crate ndarray;
extern crate num_traits;

// submodules
mod convolution;

#[cfg(test)]
mod tests {
    #[test]
    fn dummy() {
        println!("dummy test", );
        assert_eq!(2 + 2, 4);
    }
}
