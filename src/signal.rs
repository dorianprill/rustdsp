//use std::ops::{Add, Sub};
use std::vec::Vec;
use std::slice::IterMut;
use std::vec::IntoIter;

// Signal type based on newtype wrapper around Vec<T>
#[derive(Default, Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
struct Signal<T>(Vec<T>);

/// Iterator object implementation for new type
// You can create a new struct which will contain a reference to your set of data.
struct IterSignal<'a, T: 'a> {
    inner: &'a Signal<T>,
    // And there is a position used to know where you are in your iteration.
    pos: usize,
}

// Now you can just implement the `Iterator` trait on your `IterNewType` struct.
impl<'a, T> Iterator for IterSignal<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.inner.0.len() {
            // Obviously, there isn't any more data to read so let's stop here.
            None
        } else {
            // We increment the position of our iterator.
            self.pos += 1;
            // We return the current value pointed by our iterator.
            self.inner.0.get(self.pos - 1)
        }
    }
}

/// BOILERPLATE
/// Since we can't(?) access or derive the underlying vector implementation for
/// len(), iter(), iter_mut(), we have to implement simple wrappers on our own
impl<T> Signal<T> {

    fn len(self) -> usize {
        self.len()
    }

    fn iter_mut(&mut self) -> IterMut<T> {
        self.iter_mut()
    }

    fn iter(self) -> IntoIter<T> {
        self.iter()
    }

    // fn zip(self, other: U) -> Zip<Self, <U as IntoIterator::IntoIter>
    // where U: IntoIterator, {
    //     self.zip(other)
    // }
}


// obvious_impl! { impl IntoIterator for Signal { fn iter_mut } }
// obvious_impl! { impl IntoIterator for Signal { fn into_iter } }


// Since we use plain Vec<T> as implementation, we would expect the add operation to be pointwise.
// impl Add<Signal<f64>> for Signal<f64> {
//     type Output = Signal<f64>;
//     fn add(self, other: Signal<f64>) -> Signal<f64> {
//         let mut res: Signal<f64> = Signal(vec![0f64; self.len() as usize]);
//         for ((zref, a), b) in res.iter_mut().zip(&self).zip(&other) {
//             *zref = a + b;
//         }
//         res
//     }
// }


// #[cfg(test)]
// mod tests {
//     use signal::Signal;
//
//     fn add_real() {
//         let a: Signal<f64> = Signal(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
//         let b: Signal<f64> = Signal(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
//         let y: Signal<f64> = Signal(vec![2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0]);
//         let mut c: Signal<f64> = a + b;
//         assert_eq!(c, y);
//     }
//
// }
