use std::io::{Read, Seek, SeekFrom, Write};

use rand::Rng;
use rand_pcg::Pcg64Mcg;

use crate::store::VectorStore;

/// A quantization algorithm to use on input f32 element vectors.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum QuantizationAlgorithm {
    /// Trivial binary quantization. Map each dimension to 1 if > 0.0, else 0.
    Binary,
    /// Binary mean quantization. Map each dimension to 1 if > mean[dimension], else 0.
    BinaryMean,
    /// Statistical binary quantization. Map each dimension to one of bits + 1 dimensions and unary
    /// encode the bucket number. Scoring may still be done with hamming distance.
    StatisticalBinary(usize),
    /// Lucene-style scalar quantizer.
    ///
    /// Compute a min and max threshold within some confidence based on dimension, then map values
    /// linearly into buckets.
    Scalar(usize),
}

impl From<QuantizationAlgorithm> for u64 {
    fn from(value: QuantizationAlgorithm) -> Self {
        match value {
            QuantizationAlgorithm::Binary => 0u64,
            QuantizationAlgorithm::BinaryMean => 1u64,
            QuantizationAlgorithm::StatisticalBinary(bits) => 2u64 | ((bits as u64) << 32),
            QuantizationAlgorithm::Scalar(bits) => 16u64 | ((bits as u64) << 32),
        }
    }
}

impl TryFrom<u64> for QuantizationAlgorithm {
    type Error = u64;

    /// Convert u64 to a `QuantizationAlgorithm`, return the value if invalid.`
    fn try_from(value: u64) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(QuantizationAlgorithm::Binary),
            1 => Ok(QuantizationAlgorithm::BinaryMean),
            b if b & 0xff == 2 => {
                let bits = (b >> 32) as usize;
                if bits > 1 {
                    Ok(QuantizationAlgorithm::StatisticalBinary(bits))
                } else {
                    Err(value)
                }
            }
            s if s & 0xff == 16 => {
                let bits = (s >> 32) as usize;
                if (2..=8).contains(&bits) {
                    Ok(QuantizationAlgorithm::Scalar(bits))
                } else {
                    Err(value)
                }
            }
            _ => Err(value),
        }
    }
}

#[derive(Debug)]
enum QuantizerState {
    Binary,
    BinaryMean(Vec<f32>),
    StatisticalBinary(StatisticalBinaryQuantizerState),
    Scalar(ScalarQuantizerState),
}

impl QuantizerState {
    fn algorithm(&self) -> QuantizationAlgorithm {
        match self {
            QuantizerState::Binary => QuantizationAlgorithm::Binary,
            QuantizerState::BinaryMean(_) => QuantizationAlgorithm::BinaryMean,
            QuantizerState::StatisticalBinary(state) => {
                QuantizationAlgorithm::StatisticalBinary(state.bits)
            }
            QuantizerState::Scalar(state) => QuantizationAlgorithm::Scalar(state.bits),
        }
    }
}

#[derive(Debug)]
struct StatisticalBinaryQuantizerState {
    dimensions: Vec<SBDimension>,
    bits: usize,
}

impl StatisticalBinaryQuantizerState {
    fn new(bits: usize, dimensions: Vec<SBDimension>) -> Self {
        Self { dimensions, bits }
    }

    fn from_store(
        bits: usize,
        store: &impl VectorStore<Vector = [f32]>,
        dimensions: usize,
    ) -> Self {
        let mut means = vec![0f32; dimensions];
        let mut m2s = vec![0f32; dimensions];
        let iter = SampleIterator::new(store);
        let count = iter.len();
        for (i, v) in iter.enumerate() {
            for (s, (m, m2)) in v.iter().zip(means.iter_mut().zip(m2s.iter_mut())) {
                let delta = s - *m;
                *m += delta / (i + 1) as f32;
                *m2 += delta * (*s - *m);
            }
        }
        Self::new(
            bits,
            means
                .iter()
                .zip(m2s.iter())
                .map(|(mean, mean_squared)| SBDimension {
                    mean: *mean,
                    std_dev: (*mean_squared / count as f32).sqrt(),
                })
                .collect::<Vec<_>>(),
        )
    }

    fn read_footer(
        dimensions: usize,
        reader: &mut (impl Read + Seek),
        bits: usize,
    ) -> std::io::Result<(Self, usize)> {
        let dim_size = 8 * dimensions;
        reader.seek(SeekFrom::Current(-(dim_size as i64)))?;
        let mut dim = Vec::with_capacity(dimensions);
        let mut buf = [0u8; 4];
        for _ in 0..dimensions {
            reader.read_exact(&mut buf)?;
            let mean = f32::from_le_bytes(buf);
            reader.read_exact(&mut buf)?;
            let std_dev = f32::from_le_bytes(buf);
            dim.push(SBDimension { mean, std_dev });
        }
        Ok((Self::new(bits, dim), dim_size))
    }

    fn write_footer(&self, writer: &mut impl Write) -> std::io::Result<()> {
        for dim in self.dimensions.iter() {
            writer.write_all(&dim.mean.to_le_bytes())?;
            writer.write_all(&dim.std_dev.to_le_bytes())?;
        }
        Ok(())
    }

    fn quantize_to(&self, vector: &[f32], out: &mut [u8]) {
        // TODO: when applied with 2 bits you get a value distribution split of 25-50-25.
        // We might prefer an even split
        out.fill(0u8);
        let buckets = self.bits + 1;
        for (i, (v, d)) in vector.iter().zip(self.dimensions.iter()).enumerate() {
            let start_bit = i * self.bits;
            // Map z scores between -2 and 2 to `buckets` values, which are then unary coded
            // into the output buffer.
            let zscore = (*v - d.mean) / d.std_dev;
            let bucket_float = (zscore + 2.0) / (4.0 / buckets as f32);
            let bucket = if bucket_float < 0.0 {
                0
            } else {
                std::cmp::min(bucket_float.floor() as usize, buckets - 1)
            };
            // This is pretty inefficient to handle cases where state.bits does not evenly
            // divide the length of a byte.
            for u in 0..bucket {
                out[(start_bit + u) / 8] |= 1u8 << ((start_bit + u) % 8);
            }
        }
    }

    fn dequantize_to(&self, vector: &[u8], out: &mut [f32]) {
        // TODO: store this in self rather than recomputing it.
        let stddev_bucket_span = 4.0 / (self.bits + 1) as f32;
        let stddev_buckets: Vec<f32> = (0..(self.bits + 1))
            .map(|i| -2.0 + (stddev_bucket_span / 2.0) + (i as f32 * stddev_bucket_span))
            .collect();
        for ((v, d), o) in (0..out.len())
            .map(|i| self.get_dimension_value(vector, i))
            .zip(self.dimensions.iter())
            .zip(out.iter_mut())
        {
            *o = d.mean + (stddev_buckets[v.trailing_ones() as usize] * d.std_dev);
        }
    }

    fn get_dimension_value(&self, vector: &[u8], dimension: usize) -> u16 {
        assert!(self.bits <= 8);
        let start_bit = dimension * self.bits;
        let src = &vector[(start_bit / 8)..];
        let short = if src.len() > 1 {
            let mut buf = [0u8; 2];
            buf.copy_from_slice(&src[..2]);
            u16::from_le_bytes(buf)
        } else {
            u16::from(src[0])
        };
        (short >> (start_bit % 8)) & ((1u16 << self.bits) - 1)
    }
}

#[derive(Debug)]
struct SBDimension {
    mean: f32,
    std_dev: f32,
}

#[derive(Debug)]
struct ScalarQuantizerState {
    min_quantile: f32,
    max_quantile: f32,
    scale: f32,
    bits: usize,
}

impl ScalarQuantizerState {
    fn new(bits: usize, min_quantile: f32, max_quantile: f32) -> Self {
        let range = max_quantile - min_quantile;
        let divisor = (1usize << bits) - 1;
        let scale = divisor as f32 / range;
        Self {
            min_quantile,
            max_quantile,
            scale,
            bits,
        }
    }

    fn from_store(bits: usize, store: &impl VectorStore<Vector = [f32]>) -> Self {
        // XXX missing any notion of confidence
        let mut min_quantile = f32::MAX;
        let mut max_quantile = f32::MIN;
        for v in SampleIterator::new(store) {
            for d in v.iter() {
                min_quantile = min_quantile.min(*d);
                max_quantile = max_quantile.max(*d);
            }
        }
        Self::new(bits, min_quantile, max_quantile)
    }

    fn read_footer(reader: &mut (impl Read + Seek), bits: usize) -> std::io::Result<(Self, usize)> {
        let mut fbuf = [0u8; 4];
        reader.seek(SeekFrom::Current(-8))?;
        reader.read_exact(&mut fbuf)?;
        let min_quantile = f32::from_le_bytes(fbuf);
        reader.read_exact(&mut fbuf)?;
        let max_quantile = f32::from_le_bytes(fbuf);
        Ok((
            ScalarQuantizerState::new(bits, min_quantile, max_quantile),
            8,
        ))
    }

    pub fn write_footer(&self, writer: &mut impl Write) -> std::io::Result<()> {
        writer.write_all(&self.min_quantile.to_le_bytes())?;
        writer.write_all(&self.max_quantile.to_le_bytes())
    }

    fn quantize_to(&self, vector: &[f32], out: &mut [u8]) {
        let quantize_value = |d: f32| -> u8 {
            (self.scale * (self.min_quantile.max(self.max_quantile.min(d)) - self.min_quantile))
                .round() as u8
        };
        match self.bits.next_power_of_two() {
            2 => {
                for (c, o) in vector.chunks(4).zip(out.iter_mut()) {
                    *o = c
                        .iter()
                        .enumerate()
                        .map(|(i, d)| quantize_value(*d) << (i * 2))
                        .fold(0, |a, b| a | b);
                }
            }
            4 => {
                for (c, o) in vector.chunks(2).zip(out.iter_mut()) {
                    *o = c
                        .iter()
                        .enumerate()
                        .map(|(i, d)| quantize_value(*d) << (i * 4))
                        .fold(0, |a, b| a | b)
                }
            }
            8 => {
                for (i, o) in vector.iter().zip(out.iter_mut()) {
                    *o = quantize_value(*i);
                }
            }
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
pub struct Quantizer {
    state: QuantizerState,
}

impl Quantizer {
    /// Create a new quantizer for the given algorithm using the contents of store.
    ///
    /// This may sample from the store to compute quantization parameters.
    // TODO: dimensions should be a property of the store.
    pub fn from_store(
        algo: QuantizationAlgorithm,
        store: &impl VectorStore<Vector = [f32]>,
        dimensions: usize,
    ) -> Self {
        let state = match algo {
            QuantizationAlgorithm::Binary => QuantizerState::Binary,
            QuantizationAlgorithm::BinaryMean => {
                let mut means = vec![0f32; dimensions];
                for (i, v) in SampleIterator::new(store).enumerate() {
                    for (m, s) in means.iter_mut().zip(v.iter()) {
                        *m = (s - *m) / (i + 1) as f32;
                    }
                }
                QuantizerState::BinaryMean(means)
            }
            QuantizationAlgorithm::StatisticalBinary(bits) => QuantizerState::StatisticalBinary(
                StatisticalBinaryQuantizerState::from_store(bits, store, dimensions),
            ),
            QuantizationAlgorithm::Scalar(bits) => {
                QuantizerState::Scalar(ScalarQuantizerState::from_store(bits, store))
            }
        };
        Self { state }
    }

    /// Return a new stateless binary quantizer.
    pub fn new_binary_quantizer() -> Self {
        Self {
            state: QuantizerState::Binary,
        }
    }

    /// Create a quantizer from state held in the footer of a file.
    ///
    /// Returns the quantizer and number of bytes consumed from the end of the store, or an error
    /// if the data is invalid.
    pub fn read_footer(
        dimensions: usize,
        reader: &mut (impl Read + Seek),
    ) -> std::io::Result<(Quantizer, usize)> {
        reader.seek(SeekFrom::End(-8))?;
        let mut algo_buf = [0u8; 8];
        reader.read_exact(&mut algo_buf)?;
        let algo = QuantizationAlgorithm::try_from(u64::from_le_bytes(algo_buf)).unwrap();
        let result = match algo {
            QuantizationAlgorithm::Binary => Ok((QuantizerState::Binary, 0usize)),
            QuantizationAlgorithm::BinaryMean => {
                let means_size = 4 * dimensions;
                reader.seek(SeekFrom::Current(-(means_size as i64)))?;
                let mut means = Vec::with_capacity(dimensions);
                let mut buf = [0u8; 4];
                for _ in 0..dimensions {
                    reader.read_exact(&mut buf)?;
                    means.push(f32::from_le_bytes(buf));
                }
                Ok((QuantizerState::BinaryMean(means), means_size))
            }
            QuantizationAlgorithm::StatisticalBinary(bits) => {
                StatisticalBinaryQuantizerState::read_footer(dimensions, reader, bits)
                    .map(|(s, c)| (QuantizerState::StatisticalBinary(s), c))
            }
            QuantizationAlgorithm::Scalar(bits) => ScalarQuantizerState::read_footer(reader, bits)
                .map(|(s, c)| (QuantizerState::Scalar(s), c)),
        };
        result.map(|(state, count)| (Self { state }, count + 8))
    }

    pub fn write_footer(&self, writer: &mut impl Write) -> std::io::Result<()> {
        match self.state {
            QuantizerState::Binary => {}
            QuantizerState::BinaryMean(ref means) => {
                for m in means.iter() {
                    writer.write_all(&m.to_le_bytes())?;
                }
            }
            QuantizerState::StatisticalBinary(ref state) => state.write_footer(writer)?,
            QuantizerState::Scalar(ref state) => state.write_footer(writer)?,
        }
        writer.write_all(&u64::from(self.state.algorithm()).to_le_bytes())
    }

    /// Return a quantized version of input vector.
    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        let mut buf = self.quantization_buffer(vector.len());
        self.quantize_to(vector, &mut buf);
        buf
    }

    /// Quantize the input vector into the output buffer.
    /// *Panics* if out.len() < self.quantized_bytes(vector.len())
    pub fn quantize_to(&self, vector: &[f32], out: &mut [u8]) {
        assert!(out.len() >= self.quantized_bytes(vector.len()));
        match &self.state {
            QuantizerState::Binary => {
                let means = [0.0f32; 8];
                for (c, o) in vector.chunks(8).zip(out.iter_mut()) {
                    *o = Self::quantize_binary_chunk(&means, c);
                }
            }
            QuantizerState::BinaryMean(means) => {
                for ((c, m), o) in vector.chunks(8).zip(means.chunks(8)).zip(out.iter_mut()) {
                    *o = Self::quantize_binary_chunk(m, c);
                }
            }
            QuantizerState::StatisticalBinary(state) => state.quantize_to(vector, out),
            QuantizerState::Scalar(state) => state.quantize_to(vector, out),
        }
    }

    fn quantize_binary_chunk(means_chunk: &[f32], vector_chunk: &[f32]) -> u8 {
        vector_chunk
            .iter()
            .zip(means_chunk.iter())
            .enumerate()
            .filter_map(|(i, (v, m))| if *v > *m { Some(1u8 << i) } else { None })
            .reduce(|a, b| a | b)
            .unwrap_or(0u8)
    }

    /// Lossy transform of a quantized vector back into a float vector.
    pub fn dequantize_to(&self, vector: &[u8], out: &mut [f32]) {
        match &self.state {
            QuantizerState::Binary => {
                for (b, o) in vector.iter().zip(out.chunks_mut(8)) {
                    o[0..4].copy_from_slice(&BINARY_DEQUANTIZE_4_BITS[*b as usize & 0xf]);
                    o[5..8].copy_from_slice(&BINARY_DEQUANTIZE_4_BITS[*b as usize >> 4]);
                }
            }
            QuantizerState::BinaryMean(means) => {
                for (b, (mc, oc)) in vector.iter().zip(means.chunks(8).zip(out.chunks_mut(8))) {
                    oc[0..4].copy_from_slice(&BINARY_DEQUANTIZE_4_BITS[*b as usize & 0xf]);
                    oc[5..8].copy_from_slice(&BINARY_DEQUANTIZE_4_BITS[*b as usize >> 4]);
                    for (m, o) in mc.iter().zip(oc.iter_mut()) {
                        *o += *m;
                    }
                }
            }
            QuantizerState::StatisticalBinary(state) => state.dequantize_to(vector, out),
            QuantizerState::Scalar(_state) => todo!(),
        }
    }

    /// Return the number of output bytes required to quantize a float vector `dimensions` long.
    pub fn quantized_bytes(&self, dimensions: usize) -> usize {
        ((dimensions * self.element_bits()) + 7) / 8
    }

    /// Return a buffer that is an appropriate length to quantize `dimensions` length vectors.`
    pub fn quantization_buffer(&self, dimensions: usize) -> Vec<u8> {
        vec![0u8; self.quantized_bytes(dimensions)]
    }

    /// Return the quantizer algorithm used by this quantizer.
    pub fn algorithm(&self) -> QuantizationAlgorithm {
        self.state.algorithm()
    }

    /// Return the number of bits used to store each element in the vector.
    pub fn element_bits(&self) -> usize {
        match &self.state {
            QuantizerState::Binary | QuantizerState::BinaryMean(_) => 1,
            QuantizerState::StatisticalBinary(state) => state.bits,
            QuantizerState::Scalar(state) => state.bits.next_power_of_two(),
        }
    }
}

const BINARY_DEQUANTIZE_4_BITS: [[f32; 4]; 16] = [
    [-1.0, -1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0, -1.0],
    [-1.0, -1.0, 1.0, -1.0],
    [1.0, -1.0, 1.0, -1.0],
    [-1.0, 1.0, 1.0, -1.0],
    [1.0, 1.0, 1.0, -1.0],
    [-1.0, -1.0, -1.0, 1.0],
    [1.0, -1.0, -1.0, 1.0],
    [-1.0, 1.0, -1.0, 1.0],
    [1.0, 1.0, -1.0, 1.0],
    [-1.0, -1.0, 1.0, 1.0],
    [1.0, -1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
];

/// XXX this should be adjustable by the algorithm. In practice ~1M is very cheap for sbq but with
/// scalar with confidence intervals would likely be very expensive at this scale.
const MAX_SAMPLE_SIZE: usize = 1 << 20;
const RANDOM_SEED: u128 = 0xbeab3d60061ed00d;

// TODO: move this into store module and make it public.
struct SampleIterator<'a, S> {
    store: &'a S,
    samples: std::vec::IntoIter<usize>,
}

impl<'a, S> SampleIterator<'a, S> {
    fn new(store: &'a S) -> Self
    where
        S: VectorStore<Vector = [f32]>,
    {
        if store.len() < MAX_SAMPLE_SIZE {
            let samples = (0..store.len()).collect::<Vec<_>>().into_iter();
            Self { store, samples }
        } else {
            let mut reservoirs = (0..MAX_SAMPLE_SIZE).collect::<Vec<_>>();
            let mut rng = Pcg64Mcg::new(RANDOM_SEED);
            for i in reservoirs.len()..store.len() {
                let j = rng.gen_range(0..(i + 1));
                if j < reservoirs.len() {
                    reservoirs[j] = i;
                }
            }
            reservoirs.sort();
            Self {
                store,
                samples: reservoirs.into_iter(),
            }
        }
    }
}

impl<'a, S> Iterator for SampleIterator<'a, S>
where
    S: VectorStore<Vector = [f32]>,
{
    type Item = &'a [f32];

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.samples.next()?;
        Some(self.store.get(index))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.samples.size_hint()
    }
}

impl<'a, S> ExactSizeIterator for SampleIterator<'a, S> where S: VectorStore<Vector = [f32]> {}
