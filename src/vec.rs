use serde::Serialize;
use simsimd::{BinarySimilarity, SpatialSimilarity};
use std::{collections::hash_map::DefaultHasher, hash::Hasher, io::Write, marker::PhantomData};

// XXX should dimensions be non-zero usize?

pub trait VectorView<'a> {
    fn new(data: &'a [u8], dimensions: usize) -> Self;
    fn len(dimensions: usize) -> usize;

    fn dimensions(&self) -> usize;
    fn score(&self, other: &Self) -> f32;
}

pub struct FloatVectorView<'a> {
    data: &'a [f32],
}

impl<'a> FloatVectorView<'a> {
    fn binary_quantize_chunk(chunk: &[f32]) -> u8 {
        chunk
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > 0.0)
            .map(|(i, _)| 1u8 << i)
            .reduce(|a, b| a | b)
            .unwrap_or(0u8)
    }

    pub fn binary_quantize(&self) -> impl Iterator<Item = u8> + 'a {
        self.data.chunks(8).map(Self::binary_quantize_chunk)
    }

    pub fn write_binary_quantized<W: Write>(&self, out: &mut W) -> std::io::Result<()> {
        for b in self.binary_quantize() {
            out.write_all(std::slice::from_ref(&b))?;
        }
        Ok(())
    }
}

impl<'a> VectorView<'a> for FloatVectorView<'a> {
    fn new(data: &'a [u8], dimensions: usize) -> Self {
        assert!(data.len() >= Self::len(dimensions));
        let (prefix, v, suffix) = unsafe { data.align_to::<f32>() };
        assert_eq!(prefix.len(), 0);
        assert_eq!(suffix.len(), 0);
        Self {
            data: &v[0..dimensions],
        }
    }

    fn len(dimensions: usize) -> usize {
        dimensions * std::mem::size_of::<f32>()
    }

    fn dimensions(&self) -> usize {
        self.data.len()
    }

    fn score(&self, other: &Self) -> f32 {
        SpatialSimilarity::cosine(self.data, other.data).unwrap() as f32
    }
}

#[derive(Serialize)]
pub struct BinaryVectorView<'a> {
    data: &'a [u8],
    dimensions: usize,
}

impl<'a> BinaryVectorView<'a> {
    pub fn count_ones(&self) -> u32 {
        let chunks = self.data.chunks_exact(8);
        let mut count = chunks.remainder().iter().copied().map(u8::count_ones).sum();
        for chunk in chunks {
            count += u64::from_le_bytes(chunk.try_into().unwrap()).count_ones();
        }
        count
    }

    pub fn word_iter(&self) -> WordIter<'a> {
        WordIter::new(self.data)
    }

    pub fn ones_iter(&self) -> OnesIter<'_> {
        OnesIter::new(self.data)
    }

    pub fn hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        hasher.write(self.data);
        hasher.finish()
    }

    pub fn distance(&self, other: &Self) -> u32 {
        BinarySimilarity::hamming(self.data, other.data).unwrap() as u32
    }
}

impl<'a> VectorView<'a> for BinaryVectorView<'a> {
    fn new(data: &'a [u8], dimensions: usize) -> Self {
        let bytes = Self::len(dimensions);
        assert!(data.len() >= bytes, "{} >= {}", data.len(), bytes,);
        Self {
            data: &data[..bytes],
            dimensions,
        }
    }

    fn len(dimensions: usize) -> usize {
        (dimensions + 7) / 8
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn score(&self, other: &Self) -> f32 {
        1.0f32 / (1.0f32 + BinarySimilarity::hamming(self.data, other.data).unwrap() as f32)
    }
}

pub struct WordIter<'a> {
    data: &'a [u8],
}

impl<'a> WordIter<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data }
    }
}

impl<'a> Iterator for WordIter<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.data.len() >= 8 {
            let (bb, rem) = self.data.split_at(std::mem::size_of::<u64>());
            self.data = rem;
            Some(u64::from_le_bytes(bb.try_into().unwrap()))
        } else if self.data.len() > 0 {
            let mut bb = [0u8; std::mem::size_of::<u64>()];
            bb[..self.data.len()].copy_from_slice(self.data);
            self.data = &[];
            Some(u64::from_le_bytes(bb))
        } else {
            None
        }
    }
}

pub struct OnesIter<'a> {
    data: &'a [u8],
    base: usize,
    buf: u64,
}

impl<'a> OnesIter<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            buf: 0,
            base: usize::MAX,
        }
    }
}

impl<'a> Iterator for OnesIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while self.buf == 0 {
            if self.data.is_empty() {
                return None;
            }

            if self.data.len() >= 8 {
                let (bb, rem) = self.data.split_at(std::mem::size_of::<u64>());
                self.buf = u64::from_le_bytes(bb.try_into().unwrap());
                self.data = rem;
            } else {
                let mut bb = [0u8; std::mem::size_of::<u64>()];
                bb[..self.data.len()].copy_from_slice(self.data);
                self.buf = u64::from_le_bytes(bb);
                self.data = &[];
            }

            self.base = if self.base == usize::MAX {
                0
            } else {
                self.base + 64
            };
        }

        let next = self.buf.trailing_zeros();
        self.buf ^= 1 << next;
        Some(self.base + next as usize)
    }
}

pub struct VectorViewStore<'a, V>
where
    V: VectorView<'a>,
{
    data: &'a [u8],
    dimensions: usize,
    stride: usize,
    len: usize,
    m: PhantomData<V>,
}

impl<'a, V> VectorViewStore<'a, V>
where
    V: VectorView<'a>,
{
    pub fn new(data: &'a [u8], dimensions: usize) -> Self {
        let stride = V::len(dimensions);
        assert_eq!(data.len() % stride, 0);
        let len = data.len() / stride;
        Self {
            data,
            dimensions,
            stride,
            len,
            m: PhantomData,
        }
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn get(&self, i: usize) -> V {
        let start = self.stride * i;
        V::new(&self.data[start..(start + self.stride)], self.dimensions)
    }

    pub fn iter(&self) -> Iter<'a, '_, V> {
        Iter {
            store: self,
            next: 0,
        }
    }
}

pub struct Iter<'a, 'b, V>
where
    V: VectorView<'a>,
{
    store: &'b VectorViewStore<'a, V>,
    next: usize,
}

impl<'a, 'b, V> Iterator for Iter<'a, 'b, V>
where
    V: VectorView<'a>,
{
    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.store.len() {
            return None;
        }

        let v = self.store.get(self.next);
        self.next += 1;
        Some(v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.store.len() - self.next;
        (len, Some(len))
    }
}

impl<'a, 'b, V> ExactSizeIterator for Iter<'a, 'b, V> where V: VectorView<'a> {}

// store?
// random vector access + iteration
