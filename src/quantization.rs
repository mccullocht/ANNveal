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
    // TODO: BinaryVariance(usize),
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
    #[allow(dead_code)]
    BinaryMean(Vec<f32>),
    Scalar(ScalarQuantizerState),
}

impl QuantizerState {
    fn algorithm(&self) -> QuantizationAlgorithm {
        match self {
            QuantizerState::Binary => QuantizationAlgorithm::Binary,
            QuantizerState::BinaryMean(_) => QuantizationAlgorithm::BinaryMean,
            QuantizerState::Scalar(state) => QuantizationAlgorithm::Scalar(state.bits),
        }
    }
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
}

#[derive(Debug)]
pub struct Quantizer {
    state: QuantizerState,
}

#[allow(dead_code)]
impl Quantizer {
    /// Create a new quantizer for the given algorithm using the contents of store.
    ///
    /// This may sample from the store to compute quantization parameters.
    // XXX TODO: dimensions should be a property of the store.
    pub fn from_store(
        algo: QuantizationAlgorithm,
        store: &impl VectorStore<Vector = [f32]>,
        _dimensions: usize,
    ) -> Self {
        match algo {
            QuantizationAlgorithm::Binary => Self::new_binary_quantizer(),
            QuantizationAlgorithm::BinaryMean => {
                todo!()
            }
            QuantizationAlgorithm::Scalar(bits) => {
                // XXX missing any notion of confidence
                let mut min_quantile = f32::MAX;
                let mut max_quantile = f32::MIN;
                for v in SampleIterator::new(store) {
                    for d in v.iter() {
                        min_quantile = min_quantile.min(*d);
                        max_quantile = max_quantile.max(*d);
                    }
                }
                Self {
                    state: QuantizerState::Scalar(ScalarQuantizerState::new(
                        bits,
                        min_quantile,
                        max_quantile,
                    )),
                }
            }
        }
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
        _dimensions: usize,
        reader: &mut (impl Read + Seek),
    ) -> std::io::Result<(Quantizer, usize)> {
        reader.seek(SeekFrom::End(-8))?;
        let mut algo_buf = [0u8; 8];
        reader.read_exact(&mut algo_buf)?;
        let algo = QuantizationAlgorithm::try_from(u64::from_le_bytes(algo_buf)).unwrap();
        match algo {
            QuantizationAlgorithm::Binary => Ok((Self::new_binary_quantizer(), 8)),
            QuantizationAlgorithm::BinaryMean => todo!(),
            QuantizationAlgorithm::Scalar(bits) => {
                let mut fbuf = [0u8; 4];
                reader.seek(SeekFrom::Current(-8))?;
                reader.read_exact(&mut fbuf)?;
                let min_quantile = f32::from_le_bytes(fbuf);
                reader.read_exact(&mut fbuf)?;
                let max_quantile = f32::from_le_bytes(fbuf);
                Ok((
                    Self {
                        state: QuantizerState::Scalar(ScalarQuantizerState::new(
                            bits,
                            min_quantile,
                            max_quantile,
                        )),
                    },
                    16,
                ))
            }
        }
    }

    pub fn write_footer(&self, writer: &mut impl Write) -> std::io::Result<()> {
        match self.state {
            QuantizerState::Binary => {}
            QuantizerState::BinaryMean(_) => todo!(),
            QuantizerState::Scalar(ref state) => {
                writer.write_all(&state.min_quantile.to_le_bytes())?;
                writer.write_all(&state.max_quantile.to_le_bytes())?;
            }
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
                for (c, o) in vector.chunks(8).zip(out.iter_mut()) {
                    *o = c
                        .iter()
                        .enumerate()
                        .filter_map(|(i, e)| if *e > 0.0 { Some(1u8 << i) } else { None })
                        .reduce(|a, b| a | b)
                        .unwrap_or(0u8);
                }
            }
            QuantizerState::BinaryMean(means) => {
                for ((c, m), o) in vector.chunks(8).zip(means.chunks(8)).zip(out.iter_mut()) {
                    *o = c
                        .iter()
                        .zip(m.iter())
                        .enumerate()
                        .filter_map(|(i, (d, m))| if *d > *m { Some(1u8 << i) } else { None })
                        .reduce(|a, b| a | b)
                        .unwrap_or(0u8)
                }
            }
            QuantizerState::Scalar(state) => {
                let quantize_value = |d: f32| -> u8 {
                    (state.scale
                        * (state.min_quantile.max(state.max_quantile.min(d)) - state.min_quantile))
                        .round() as u8
                };
                match state.bits.next_power_of_two() {
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
            QuantizerState::Scalar(state) => state.bits.next_power_of_two(),
        }
    }
}

const MAX_SAMPLE_SIZE: usize = 32 << 10;
const RANDOM_SEED: u128 = 0xbeab3d60061ed00d;

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
            let samples = (0..store.len()).into_iter().collect::<Vec<_>>().into_iter();
            Self { store, samples }
        } else {
            let mut reservoirs = (0..MAX_SAMPLE_SIZE).into_iter().collect::<Vec<_>>();
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
}
