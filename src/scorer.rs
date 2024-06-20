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
                    2 => Self::distance2(a, b),
                    4 => Self::distance4(a, b),
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

    fn score_unpacked(&self, unpacked: &[i8], packed: &[u8]) -> NotNan<f32> {
        let distance = match self {
            Self::Binary => unreachable!(),
            Self::Scalar(bits, score_adjustment_multiplier) => {
                let raw_distance: f64 = match bits {
                    2 => unpacked
                        .chunks(64)
                        .zip(packed.chunks(16))
                        .map(|(u, p)| {
                            let buf = Self::unpack2_quad(p);
                            SpatialSimilarity::l2sq(u, &buf).unwrap()
                        })
                        .sum(),
                    4 => unpacked
                        .chunks(32)
                        .zip(packed.chunks(16))
                        .map(|(u, p)| {
                            let buf = Self::unpack4_quad(p);
                            SpatialSimilarity::l2sq(u, &buf).unwrap()
                        })
                        .sum(),
                    _ => unreachable!(),
                };
                raw_distance as f32 * *score_adjustment_multiplier
            }
        };
        NotNan::new(1.0 / (1.0 + distance)).unwrap()
    }

    fn unpack_vector(&self, packed: &[u8]) -> Option<Vec<i8>> {
        match self {
            Self::Binary => None,
            Self::Scalar(bits, _) => match bits {
                2 => Some(
                    packed
                        .chunks(16)
                        .flat_map(|p| Self::unpack2_quad(p))
                        .collect(),
                ),
                4 => Some(
                    packed
                        .chunks(16)
                        .flat_map(|p| Self::unpack4_quad(p))
                        .collect(),
                ),
                _ => None,
            },
        }
    }

    fn unpack2_quad(packed: &[u8]) -> [i8; 64] {
        debug_assert!(packed.len() <= 16);
        let quad = if packed.len() == 16 {
            u128::from_le_bytes(packed.try_into().unwrap())
        } else {
            let mut b = [0u8; 16];
            b[..packed.len()].copy_from_slice(packed);
            u128::from_le_bytes(b)
        };
        let mask = u128::from_le_bytes([0x3u8; 16]);
        let wbuf = [
            quad & mask,
            (quad >> 2) & mask,
            (quad >> 4) & mask,
            (quad >> 6) & mask,
        ];
        // Safety: array-to-array conversion of a primitive type.
        unsafe { std::mem::transmute(wbuf) }
    }

    fn unpack4_quad(packed: &[u8]) -> [i8; 32] {
        let quad = if packed.len() == 16 {
            u128::from_le_bytes(packed.try_into().unwrap())
        } else {
            let mut b = [0u8; 16];
            b[..packed.len()].copy_from_slice(packed);
            u128::from_le_bytes(b)
        };
        let mask = u128::from_le_bytes([0xfu8; 16]);
        let wbuf = [quad & mask, (quad >> 4) & mask];
        // Safety: array-to-array conversion of a primitive type.
        unsafe { std::mem::transmute(wbuf) }
    }

    fn distance2(a: &[u8], b: &[u8]) -> f64 {
        a.chunks(16)
            .zip(b.chunks(16))
            .map(|(ac, bc)| {
                let abuf = Self::unpack2_quad(ac);
                let bbuf = Self::unpack2_quad(bc);
                SpatialSimilarity::l2sq(&abuf, &bbuf).unwrap()
            })
            .sum()
    }

    fn distance4(a: &[u8], b: &[u8]) -> f64 {
        a.chunks(16)
            .zip(b.chunks(16))
            .map(|(ac, bc)| {
                let abuf = Self::unpack4_quad(ac);
                let bbuf = Self::unpack4_quad(bc);
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

    /// If the vector packs multiple elements per byte, unpack them to produce a vector with one
    /// element per byte.
    ///
    /// Returns `None` if thie quantizer does not produce packed vectors.
    fn unpack_vector(&self, packed: &[u8]) -> Option<Vec<i8>> {
        self.0.unpack_vector(packed)
    }

    /// Score an unpacked vector against a packed vector.
    /// If the quantizer packs multiple elements into a single byte this may elide a bunch of work
    /// by only unpacking the query once.
    fn score_unpacked(&self, unpacked: &[i8], packed: &[u8]) -> NotNan<f32> {
        self.0.score_unpacked(unpacked, packed)
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
    unpacked: Option<Vec<i8>>,
}

impl QuantizedEuclideanQueryScorer {
    /// Create a new query scorer from a quantizer and a raw float query.
    pub fn new(quantizer: &Quantizer, query: &[f32]) -> Self {
        let scorer = QuantizedEuclideanScorer::new(quantizer);
        let query = quantizer.quantize(query);
        let unpacked = scorer.unpack_vector(&query);
        Self {
            scorer,
            query,
            unpacked,
        }
    }
}

impl QueryScorer for QuantizedEuclideanQueryScorer {
    type Vector = [u8];

    fn score(&self, a: &Self::Vector) -> NotNan<f32> {
        if let Some(unpacked) = self.unpacked.as_ref() {
            self.scorer.score_unpacked(unpacked, a)
        } else {
            self.scorer.score(self.query.as_ref(), a)
        }
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
