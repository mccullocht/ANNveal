use std::{cmp::Ordering, num::NonZeroUsize};

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
pub struct SliceBitVectorStore<S> {
    data: S,
    stride: usize,
    len: usize,
}

impl<S> SliceBitVectorStore<S>
where
    S: AsRef<[u8]>,
{
    pub fn new(data: S, dimensions: NonZeroUsize) -> Self {
        let stride = (dimensions.get() + 7) / 8;
        assert_eq!(data.as_ref().len() % stride, 0); // XXX improve error handling
        let len = data.as_ref().len() / stride;
        Self { data, stride, len }
    }
}

impl<S> VectorStore for SliceBitVectorStore<S>
where
    S: AsRef<[u8]>,
{
    type Vector = [u8];

    fn get(&self, i: usize) -> &Self::Vector {
        let start = i * self.stride;
        let end = start + self.stride;
        return &self.data.as_ref()[start..end];
    }

    fn len(&self) -> usize {
        self.len
    }

    fn mean_vector(&self) -> Vec<u8> {
        if self.len() == 0 {
            return vec![];
        }

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
