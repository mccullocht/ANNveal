use ordered_float::NotNan;
use simsimd::{BinarySimilarity, SpatialSimilarity};

/// Trait for scoring vectors against one another.
pub trait VectorScorer {
    /// Type for the underlying vector data.
    type Vector: ?Sized;

    /// Return the non-nan score of the two vectors. Larger values are better.
    fn score(&self, a: &Self::Vector, b: &Self::Vector) -> NotNan<f32>;
}

pub struct HammingScorer;

impl VectorScorer for HammingScorer {
    type Vector = [u8];

    fn score(&self, a: &Self::Vector, b: &Self::Vector) -> NotNan<f32> {
        let dim = (a.len() * 8) as f32;
        let distance = BinarySimilarity::hamming(a, b).unwrap() as f32;
        NotNan::new((dim - distance) / dim).unwrap()
    }
}

pub struct EuclideanScorer;

impl VectorScorer for EuclideanScorer {
    type Vector = [f32];

    fn score(&self, a: &Self::Vector, b: &Self::Vector) -> NotNan<f32> {
        NotNan::new((1.0 / (1.0 + SpatialSimilarity::l2sq(a, b).unwrap())) as f32).unwrap()
    }
}
