use std::{cmp::Ordering, io::Cursor, num::NonZeroUsize, ops::Deref};

use crate::quantization::{QuantizationAlgorithm, Quantizer};

/// Dense store of vector data, analogous to a `Vec``.
pub trait VectorStore {
    /// Type of the underlying vector data.
    type Vector: ?Sized;

    /// Obtain a reference to the contents of the vector by VectorId.
    /// *Panics* if `i`` is out of bounds.
    fn get(&self, i: usize) -> &Self::Vector;

    /// Return the total number of vectors in the store.
    fn len(&self) -> usize;

    /// Return an iterator over the vectors in this store.
    fn iter(&self) -> VectorStoreIterator<'_, Self> {
        return VectorStoreIterator::new(self);
    }

    /// Compute the mean vector of all the points in the store.
    fn mean_vector(&self) -> <Self::Vector as ToOwned>::Owned
    where
        Self::Vector: ToOwned;
}

pub struct VectorStoreIterator<'a, S>
where
    S: ?Sized,
{
    store: &'a S,
    next: usize,
}

impl<'a, S> VectorStoreIterator<'a, S>
where
    S: VectorStore + ?Sized,
{
    fn new(store: &'a S) -> Self {
        Self { store, next: 0 }
    }
}

impl<'a, S> Iterator for VectorStoreIterator<'a, S>
where
    S: VectorStore,
{
    type Item = &'a S::Vector;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next < self.store.len() {
            let v = self.store.get(self.next);
            self.next += 1;
            Some(v)
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct SliceQuantizedVectorStore<S> {
    data: S,
    quantizer: Quantizer,
    stride: usize,
    len: usize,
}

impl<S> SliceQuantizedVectorStore<S>
where
    S: Deref<Target = [u8]>,
{
    pub fn new(data: S, dimensions: NonZeroUsize) -> Self {
        let mut cursor = Cursor::new(&*data);
        let (quantizer, footer_len) =
            Quantizer::read_footer(dimensions.get(), &mut cursor).unwrap();
        assert!(footer_len < data.len());
        let data_len = data.len() - footer_len;
        let stride = quantizer.quantized_bytes(dimensions.get());
        assert_eq!(data_len % stride, 0);
        let len = data_len / stride;
        Self {
            data,
            quantizer,
            stride,
            len,
        }
    }

    pub fn quantizer(&self) -> &Quantizer {
        &self.quantizer
    }

    fn mean_binary_vector(&self) -> Vec<u8> {
        // NB: this _could_ be paralellized, but it also only takes 1.6s to compute the centroid
        // over 1M 1536d vectors so it's not a huge problem.
        let dim = self.get(0).len() * 8;
        let mut counts = vec![0usize; dim];
        for i in 0..self.len() {
            let v = self.get(i);
            for (j, sv) in v.chunks(8).enumerate() {
                let mut sv64: u64 = if sv.len() == 8 {
                    u64::from_le_bytes(sv.try_into().unwrap())
                } else {
                    let mut svb = [0u8; 8];
                    svb[..sv.len()].copy_from_slice(sv);
                    u64::from_le_bytes(svb)
                };
                while sv64 != 0 {
                    let k = sv64.trailing_zeros() as usize;
                    counts[j * 64 + k] += 1;
                    sv64 ^= 1u64 << k;
                }
            }
        }

        let mut mean = vec![0u8; dim / 8];
        for (i, c) in counts.into_iter().enumerate() {
            let d = match c.cmp(&(self.len() / 2)) {
                Ordering::Less => false,
                Ordering::Greater => true,
                Ordering::Equal => {
                    if self.len() & 1 == 0 {
                        i % 2 == 0 // on tie, choose an element at random
                    } else {
                        false // middle is X.5 and integers round down, so like Less
                    }
                }
            };
            if d {
                mean[i / 8] |= 1u8 << (i % 8);
            }
        }

        mean
    }
}

impl<S> VectorStore for SliceQuantizedVectorStore<S>
where
    S: Deref<Target = [u8]>,
{
    type Vector = [u8];

    fn get(&self, i: usize) -> &Self::Vector {
        let start = i * self.stride;
        let end = start + self.stride;
        return &self.data[start..end];
    }

    fn len(&self) -> usize {
        self.len
    }

    fn mean_vector(&self) -> <Self::Vector as ToOwned>::Owned
    where
        Self::Vector: ToOwned,
    {
        match &self.quantizer.algorithm() {
            QuantizationAlgorithm::Binary | QuantizationAlgorithm::BinaryMean => {
                self.mean_binary_vector()
            }
            QuantizationAlgorithm::Scalar(_) => {
                todo!()
            }
        }
    }
}

#[derive(Debug)]
pub struct SliceFloatVectorStore<S> {
    data: S,
    stride: usize,
    len: usize,
}

impl<S> SliceFloatVectorStore<S>
where
    S: AsRef<[u8]>,
{
    pub fn new(data: S, dimensions: NonZeroUsize) -> Self {
        assert_eq!(data.as_ref().len() % std::mem::size_of::<f32>(), 0);
        let stride = dimensions.get() * std::mem::size_of::<f32>();
        assert_eq!(data.as_ref().len() % stride, 0); // XXX improve error handling
        assert!(unsafe { data.as_ref().align_to::<f32>().0.is_empty() });
        let len = data.as_ref().len() / stride;
        Self { data, stride, len }
    }
}

impl<S> VectorStore for SliceFloatVectorStore<S>
where
    S: AsRef<[u8]>,
{
    type Vector = [f32];

    fn get(&self, i: usize) -> &Self::Vector {
        let start = i * self.stride;
        let end = start + self.stride;
        // safety: we ensured that the slice aligned correctly at construction.
        return unsafe { self.data.as_ref()[start..end].align_to::<f32>().1 };
    }

    fn len(&self) -> usize {
        self.len
    }

    fn mean_vector(&self) -> Vec<f32> {
        todo!()
    }
}

#[derive(Debug)]
pub struct SliceU32VectorStore<S> {
    data: S,
    stride: usize,
    len: usize,
}

impl<S> SliceU32VectorStore<S>
where
    S: AsRef<[u8]>,
{
    pub fn new(data: S, dimensions: NonZeroUsize) -> Self {
        assert_eq!(data.as_ref().len() % std::mem::size_of::<u32>(), 0);
        let stride = dimensions.get() * std::mem::size_of::<u32>();
        assert_eq!(data.as_ref().len() % stride, 0); // XXX improve error handling
        assert!(unsafe { data.as_ref().align_to::<u32>().0.is_empty() });
        let len = data.as_ref().len() / stride;
        Self { data, stride, len }
    }
}

impl<S> VectorStore for SliceU32VectorStore<S>
where
    S: AsRef<[u8]>,
{
    type Vector = [u32];

    fn get(&self, i: usize) -> &Self::Vector {
        let start = i * self.stride;
        let end = start + self.stride;
        // safety: we ensured that the slice aligned correctly at construction.
        return unsafe { self.data.as_ref()[start..end].align_to::<u32>().1 };
    }

    fn len(&self) -> usize {
        self.len
    }

    fn mean_vector(&self) -> Vec<u32> {
        todo!()
    }
}
