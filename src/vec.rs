use simsimd::{BinarySimilarity, SpatialSimilarity};
use std::{io::Write, marker::PhantomData};

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

pub struct BinaryVectorView<'a> {
    data: &'a [u8],
    dimensions: usize,
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
