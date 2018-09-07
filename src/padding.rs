// dspr - digital signal processing in rust
// module: padding.rs
// info: provides various padding functions for 1d and 2d signals
//       used for convolution operations


pub trait Pad {
    fn pad_to(&self, len: usize) -> Vec<f64>;
}

impl Pad for Vec<f64> {

    fn pad_to(&self, len: usize) -> Vec<f64> {
        let mut padded = self.clone();
        if len <= self.len() {
            return padded; // silently return original signal for now
        }
        padded.extend(vec![0f64; len - self.len()]);
        padded
    }

}


pub fn pad_equal(a: &mut Vec<f64>, b: &mut Vec<f64>) {
    let la = a.len();
    let lb = b.len();
    if a.len() < b.len() {
        a.extend(vec![0f64; lb - la]);
    } else if b.len() < a.len() {
        b.extend(vec![0f64; la - lb]);
    }
}


#[cfg(test)]
mod tests {
    use padding::pad_equal;
    use padding::Pad;

    #[test]
    fn pad_to_len() {
        let x: Vec<f64> = vec![3.0, 4.0, 5.0]; // excitation signal
        let h: Vec<f64> = vec![2.0, 1.0];      // impulse response of some system
        let pad = h.pad_to(x.len());
        assert_eq!(pad.len(), x.len());
        assert!(&x as *const _ != &pad as *const _)
    }

    #[test]
    fn pad_equalize_impure() {
        let mut x: Vec<f64> = vec![3.0, 4.0, 5.0]; // excitation signal
        let mut h: Vec<f64> = vec![2.0, 1.0];
        pad_equal(&mut x, &mut h);
        assert_eq!(x.len(), h.len());
    }
}
