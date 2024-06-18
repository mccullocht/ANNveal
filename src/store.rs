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

// XXX more generally I could compute a mean float vector when *quantizing* and write it into the
// store the mean vector could then I could delete much of this implementation.
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

    fn mean_statistical_binary_vector(&self, bits: usize) -> Vec<u8> {
        // XXX FIXME _sample_ to compute the mean.
        let num_dims = self.stride * 8 / bits;
        let mut sums = vec![0u64; num_dims];
        for v in self.iter() {
            for (c, u) in v.chunks(8).enumerate() {
                let mut word = if u.len() == 8 {
                    u64::from_le_bytes(u.try_into().unwrap())
                } else {
                    let mut buf = [0u8; 8];
                    buf[..u.len()].copy_from_slice(u);
                    u64::from_le_bytes(buf)
                };

                while word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    sums[(c * 64 + bit) / bits] += 1;
                    word ^= 1 << bit;
                }
            }
        }
        let mean_dims = sums
            .into_iter()
            .map(|s| (s as f64 / self.len() as f64).floor() as usize)
            .collect::<Vec<_>>();
        let mut mean_vector = self.quantizer.quantization_buffer(num_dims);
        for (i, c) in mean_dims.into_iter().enumerate() {
            let start_bit = i * bits;
            for j in 0..c {
                mean_vector[(start_bit + j) / 8] |= 1 << ((start_bit + j) % 8);
            }
        }
        mean_vector
    }

    fn mean_scalar_vector(&self, mut bits: usize) -> Vec<u8> {
        bits = bits.next_power_of_two();
        let dims = self.stride * 8 / bits;
        let mut sums = vec![0u64; dims];
        for v in self.iter() {
            match bits {
                2 => {
                    for (i, o) in v
                        .iter()
                        .flat_map(|d| {
                            [*d & 0x3, (*d >> 2) & 0x3, (*d >> 4) & 0x3, *d >> 6].into_iter()
                        })
                        .zip(sums.iter_mut())
                    {
                        *o += i as u64;
                    }
                }
                4 => {
                    for (i, o) in v
                        .iter()
                        .flat_map(|d| [*d & 0xf, *d >> 4].into_iter())
                        .zip(sums.iter_mut())
                    {
                        *o += i as u64;
                    }
                }
                8 => {
                    for (i, o) in v.iter().copied().zip(sums.iter_mut()) {
                        *o += i as u64;
                    }
                }
                _ => unreachable!(),
            }
        }
        match bits {
            2 => sums
                .chunks(4)
                .map(|c| {
                    c.iter()
                        .enumerate()
                        .map(|(i, s)| ((*s as f64 / self.len() as f64).round() as u8) << (i * 2))
                        .reduce(|a, b| a | b)
                        .unwrap_or(0u8)
                })
                .collect::<Vec<_>>(),
            4 => sums
                .chunks(2)
                .map(|c| {
                    c.iter()
                        .enumerate()
                        .map(|(i, s)| ((*s as f64 / self.len() as f64).round() as u8) << (i * 4))
                        .reduce(|a, b| a | b)
                        .unwrap_or(0u8)
                })
                .collect::<Vec<_>>(),
            8 => sums
                .into_iter()
                .map(|s| (s as f64 / self.len() as f64).round() as u8)
                .collect::<Vec<_>>(),
            _ => unreachable!(),
        }
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
        &self.data[start..end]
    }

    fn len(&self) -> usize {
        self.len
    }

    fn mean_vector(&self) -> <Self::Vector as ToOwned>::Owned
    where
        Self::Vector: ToOwned,
    {
        match self.quantizer.algorithm() {
            QuantizationAlgorithm::Binary | QuantizationAlgorithm::BinaryMean => {
                self.mean_binary_vector()
            }
            QuantizationAlgorithm::StatisticalBinary(bits) => {
                self.mean_statistical_binary_vector(bits)
            }
            QuantizationAlgorithm::Scalar(bits) => self.mean_scalar_vector(bits),
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
        unimplemented!()
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
        unimplemented!()
    }
}
