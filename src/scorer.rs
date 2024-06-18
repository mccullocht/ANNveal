use ordered_float::NotNan;
use simsimd::{BinarySimilarity, SpatialSimilarity};

use crate::quantization::Quantizer;

/// Trait for scoring vectors against one another.
pub trait VectorScorer {
    /// Type for the underlying vector data.
    type Vector: ?Sized;

    /// Return the non-nan score of the two vectors. Larger values are better.
    fn score(&self, a: &Self::Vector, b: &Self::Vector) -> NotNan<f32>;
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

/// A scorer that computes the euclidean distance between two vectors of packed 2-bit elements.
#[derive(Copy, Clone)]
pub struct Int2EuclideanScorer;

impl VectorScorer for Int2EuclideanScorer {
    type Vector = [u8];

    fn score(&self, a: &Self::Vector, b: &Self::Vector) -> NotNan<f32> {
        let distance: usize = a
            .iter()
            .zip(b.iter())
            .map(|(a, b)| {
                let d = [
                    (a & 3).abs_diff(b & 3),
                    ((a >> 2) & 3).abs_diff((b >> 2) & 3),
                    ((a >> 4) & 3).abs_diff((b >> 4) & 3),
                    (a >> 6).abs_diff(b >> 6),
                ];
                d.into_iter().map(|d| d * d).sum::<u8>() as usize
            })
            .sum();
        NotNan::new(1.0 / (1.0 + distance as f32)).unwrap()
    }
}

/// A scorer that computes the euclidean distance between two vectors of packed 2-bit elements.
#[derive(Copy, Clone)]
pub struct Int4EuclideanScorer;

impl VectorScorer for Int4EuclideanScorer {
    type Vector = [u8];

    fn score(&self, a: &Self::Vector, b: &Self::Vector) -> NotNan<f32> {
        let distance: usize = a
            .iter()
            .zip(b.iter())
            .map(|(a, b)| {
                let d = [
                    (a & 0xf).abs_diff(b & 0xf),
                    ((a >> 4) & 0xf).abs_diff((b >> 4) & 0xf),
                ];
                d.into_iter().map(|d| d * d).sum::<u8>() as usize
            })
            .sum();
        NotNan::new(1.0 / (1.0 + distance as f32)).unwrap()
    }
}

#[derive(Copy, Clone)]
pub struct EuclideanScorer;

impl VectorScorer for EuclideanScorer {
    type Vector = [f32];

    fn score(&self, a: &Self::Vector, b: &Self::Vector) -> NotNan<f32> {
        NotNan::new((1.0 / (1.0 + SpatialSimilarity::l2sq(a, b).unwrap())) as f32).unwrap()
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
        // XXX this is deeply inefficient. we could probably dequantize chunks to avoid allcoation?
        let mut doc = vec![0.0f32; self.query.len()];
        self.quantizer.dequantize_to(a, &mut doc);
        EuclideanScorer.score(self.query, &doc)
    }
}
