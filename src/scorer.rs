use ordered_float::NotNan;
use simsimd::{BinarySimilarity, SpatialSimilarity};

use crate::quantization::{QuantizationAlgorithm, Quantizer};

/// Trait for scoring vectors against one another.
pub trait VectorScorer {
    /// Type for the underlying vector data.
    type Vector: ?Sized;

    /// Return the non-nan score of the two vectors. Larger values are better.
    fn score(&self, a: &Self::Vector, b: &Self::Vector) -> NotNan<f32>;
}

/// A scorer based on the l2 distance between two vectors.
#[derive(Copy, Clone)]
pub struct EuclideanScorer;

impl VectorScorer for EuclideanScorer {
    type Vector = [f32];

    fn score(&self, a: &Self::Vector, b: &Self::Vector) -> NotNan<f32> {
        NotNan::new((1.0 / (1.0 + SpatialSimilarity::l2sq(a, b).unwrap())) as f32).unwrap()
    }
}

/// A scorer that computes the hamming distance between two vectors.
#[derive(Copy, Clone)]
pub struct HammingScorer;

impl VectorScorer for HammingScorer {
    type Vector = [u8];

    fn score(&self, a: &Self::Vector, b: &Self::Vector) -> NotNan<f32> {
        let dim = (a.len() * 8) as f32;
        let distance = BinarySimilarity::hamming(a, b).unwrap() as f32;
        NotNan::new((dim - distance) / dim).unwrap()
    }
}

#[derive(Copy, Clone, Debug)]
enum QuantizedEuclideanScorerState {
    Binary,
    Scalar(usize, f32),
}

impl QuantizedEuclideanScorerState {
    fn new(quantizer: &Quantizer) -> Self {
        match quantizer.algorithm() {
            QuantizationAlgorithm::Binary
            | QuantizationAlgorithm::BinaryMean
            | QuantizationAlgorithm::StatisticalBinary(_) => Self::Binary,
            QuantizationAlgorithm::Scalar(bits) => Self::Scalar(
                bits.next_power_of_two(),
                quantizer.score_adjustment_multiplier(),
            ),
        }
    }

    fn score(&self, a: &[u8], b: &[u8]) -> NotNan<f32> {
        match self {
            Self::Binary => HammingScorer.score(a, b),
            Self::Scalar(bits, scoring_adjustment_multiplier) => {
                let distance = match bits {
                    2 => Self::distance2_doubleword(a, b),
                    4 => Self::distance4_doubleword(a, b),
                    8 => unsafe {
                        SpatialSimilarity::l2sq(
                            std::slice::from_raw_parts(a.as_ptr() as *const i8, a.len()),
                            std::slice::from_raw_parts(b.as_ptr() as *const i8, b.len()),
                        )
                        .unwrap()
                    },
                    _ => unreachable!(),
                };
                NotNan::new(1.0 / (1.0 + (distance as f32 * *scoring_adjustment_multiplier)))
                    .unwrap()
            }
        }
    }

    fn distance2_doubleword(a: &[u8], b: &[u8]) -> f64 {
        let chunk_to_buf = |c: &[u8]| -> [i8; 16] {
            let word = if c.len() == 4 {
                u32::from_le_bytes(c.try_into().unwrap())
            } else {
                let mut b = [0u8; 4];
                b[..c.len()].copy_from_slice(c);
                u32::from_le_bytes(b)
            };
            let wbuf = [
                word & 0x03030303,
                (word >> 2) & 0x03030303,
                (word >> 4) & 0x03030303,
                (word >> 6) & 0x03030303,
            ];
            unsafe { std::mem::transmute(wbuf) }
        };
        a.chunks(8)
            .zip(b.chunks(8))
            .map(|(ac, bc)| {
                let abuf = chunk_to_buf(ac);
                let bbuf = chunk_to_buf(bc);
                SpatialSimilarity::l2sq(&abuf, &bbuf).unwrap()
            })
            .sum()
    }

    fn distance4_doubleword(a: &[u8], b: &[u8]) -> f64 {
        let chunk_to_buf = |c: &[u8]| -> [i8; 16] {
            let word = if c.len() == 8 {
                u64::from_le_bytes(c.try_into().unwrap())
            } else {
                let mut b = [0u8; 8];
                b[..c.len()].copy_from_slice(c);
                u64::from_le_bytes(b)
            };
            let wbuf = [word & 0x0f0f0f0f0f0f0f0f, (word >> 4) & 0x0f0f0f0f0f0f0f0f];
            unsafe { std::mem::transmute(wbuf) }
        };
        a.chunks(8)
            .zip(b.chunks(8))
            .map(|(ac, bc)| {
                let abuf = chunk_to_buf(ac);
                let bbuf = chunk_to_buf(bc);
                SpatialSimilarity::l2sq(&abuf, &bbuf).unwrap()
            })
            .sum()
    }
}

/// A scorer that computes the l2 distance between two quantized vectors.
#[derive(Copy, Clone, Debug)]
pub struct QuantizedEuclideanScorer(QuantizedEuclideanScorerState);

impl QuantizedEuclideanScorer {
    pub fn new(quantizer: &Quantizer) -> Self {
        Self(QuantizedEuclideanScorerState::new(quantizer))
    }
}

impl VectorScorer for QuantizedEuclideanScorer {
    type Vector = [u8];

    fn score(&self, a: &Self::Vector, b: &Self::Vector) -> NotNan<f32> {
        self.0.score(a, b)
    }
}

/// Scores a fixed query value against a passed query.
pub trait QueryScorer {
    /// Type for the underlying vector data.
    type Vector: ?Sized;

    /// Return the non-nan score of the vector against the query vector.
    fn score(&self, a: &Self::Vector) -> NotNan<f32>;
}

/// Scores a fixed query against other vector values using a `VectorScorer`.
#[derive(Clone)]
pub struct DefaultQueryScorer<'a, 'b, Q: ?Sized, S> {
    query: &'a Q,
    scorer: &'b S,
}

impl<'a, 'b, Q, S> DefaultQueryScorer<'a, 'b, Q, S>
where
    Q: ?Sized,
    S: VectorScorer<Vector = Q>,
{
    pub fn new(query: &'a Q, scorer: &'b S) -> Self {
        Self { query, scorer }
    }
}

impl<'a, 'b, Q, S> QueryScorer for DefaultQueryScorer<'a, 'b, Q, S>
where
    Q: ?Sized,
    S: VectorScorer<Vector = Q>,
{
    type Vector = Q;

    fn score(&self, a: &Self::Vector) -> NotNan<f32> {
        self.scorer.score(self.query, a)
    }
}

/// Quantize an input query and score it against other quantized queries.
pub struct QuantizedEuclideanQueryScorer {
    scorer: QuantizedEuclideanScorer,
    query: Vec<u8>,
}

impl QuantizedEuclideanQueryScorer {
    /// Create a new query scorer from a quantizer and a raw float query.
    pub fn new(quantizer: &Quantizer, query: &[f32]) -> Self {
        let scorer = QuantizedEuclideanScorer::new(quantizer);
        let query = quantizer.quantize(query);
        Self { scorer, query }
    }
}

impl QueryScorer for QuantizedEuclideanQueryScorer {
    type Vector = [u8];

    fn score(&self, a: &Self::Vector) -> NotNan<f32> {
        self.scorer.score(self.query.as_ref(), a)
    }
}

pub struct EuclideanDequantizeScorer<'a, 'b> {
    quantizer: &'a Quantizer,
    query: &'b [f32],
}

impl<'a, 'b> EuclideanDequantizeScorer<'a, 'b> {
    pub fn new(quantizer: &'a Quantizer, query: &'b [f32]) -> Self {
        Self { quantizer, query }
    }
}

impl<'a, 'b> QueryScorer for EuclideanDequantizeScorer<'a, 'b> {
    type Vector = [u8];

    fn score(&self, a: &Self::Vector) -> NotNan<f32> {
        // XXX this is deeply inefficient. we could probably dequantize chunks to avoid allocation?
        let mut doc = vec![0.0f32; self.query.len()];
        self.quantizer.dequantize_to(a, &mut doc);
        EuclideanScorer.score(self.query, &doc)
    }
}
