use std::{
    cmp::Ordering,
    fs::File,
    io::Cursor,
    num::NonZero,
    ops::{Deref, Index},
    path::Path,
};

use memmap2::Mmap;
use stable_deref_trait::StableDeref;

use crate::quantization::{QuantizationAlgorithm, Quantizer};

/// Dense store of vector data, analogous to a `Vec``.
pub trait VectorStore: Index<usize, Output = Self::Vector> {
    /// Type of the underlying vector data.
    type Vector: ?Sized;

    /// Return the total number of vectors in the store.
    fn len(&self) -> usize;

    /// Return an iterator over the vectors in this store.
    fn iter(&self) -> impl ExactSizeIterator<Item = &Self::Vector>;
}

/// Trait for vector stores that can compute a mean vector.
pub trait MeanVectorStore: VectorStore {
    /// Compute the mean vector of all the points in the store.
    fn mean_vector(&self) -> <Self::Vector as ToOwned>::Owned
    where
        Self::Vector: ToOwned;
}

pub struct StableDerefVectorStore<E: 'static, D> {
    // NB: the contents of data is referenced by raw_vectors.
    #[allow(dead_code)]
    data: D,
    raw_vectors: &'static [E],

    stride: usize,
    len: usize,
}

impl<E: 'static, D: StableDeref<Target = [u8]>> StableDerefVectorStore<E, D> {
    /// Return a new store where each vector contains stride elements.
    pub fn new(data: D, stride: usize) -> Self {
        let elem_width = std::mem::size_of::<E>();
        let vectorp = data.as_ptr() as *const E;
        assert!(vectorp.is_aligned());
        assert_eq!(data.len() % (elem_width * stride), 0);
        let len = data.len() / (stride * elem_width);

        // Safety: StableDeref guarantees the pointer is stable even after a move.
        let raw_vectors: &'static [E] =
            unsafe { std::slice::from_raw_parts(vectorp, data.len() / elem_width) };
        Self {
            data,
            raw_vectors,
            stride,
            len,
        }
    }
}

impl<E: 'static, D: StableDeref<Target = [u8]>> VectorStore for StableDerefVectorStore<E, D> {
    type Vector = [E];

    fn len(&self) -> usize {
        self.len
    }

    fn iter(&self) -> impl ExactSizeIterator<Item = &Self::Vector> {
        self.raw_vectors.chunks(self.stride)
    }
}

impl<E: 'static, D: StableDeref<Target = [u8]>> Index<usize> for StableDerefVectorStore<E, D> {
    type Output = [E];

    fn index(&self, index: usize) -> &Self::Output {
        let start = index * self.stride;
        let end = start + self.stride;
        &self.raw_vectors[start..end]
    }
}

/// Create a new vector store over file at path backed by mmap().
pub fn new_mmap_vector_store<E: 'static>(
    path: &Path,
    stride: usize,
) -> std::io::Result<StableDerefVectorStore<E, Mmap>> {
    let map = unsafe { Mmap::map(&File::open(path)?)? };
    Ok(StableDerefVectorStore::new(map, stride))
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
    pub fn new(data: S, dimensions: NonZero<usize>) -> Self {
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
        let dim = self[0].len() * 8;
        let mut counts = vec![0usize; dim];
        for i in 0..self.len() {
            let v = &self[i];
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

    fn len(&self) -> usize {
        self.len
    }

    fn iter(&self) -> impl ExactSizeIterator<Item = &Self::Vector> {
        self.data.chunks(self.stride).take(self.len)
    }
}

impl<S: Deref<Target = [u8]>> Index<usize> for SliceQuantizedVectorStore<S> {
    type Output = [u8];

    fn index(&self, index: usize) -> &Self::Output {
        let start = index * self.stride;
        let end = start + self.stride;
        &self.data[start..end]
    }
}

impl<S: Deref<Target = [u8]>> MeanVectorStore for SliceQuantizedVectorStore<S> {
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
