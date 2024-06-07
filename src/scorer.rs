use ordered_float::NotNan;
use simsimd::{BinarySimilarity, SpatialSimilarity};

/// Trait for scoring vectors against one another.
pub trait VectorScorer {
    /// Type for the underlying vector data.
    // XXX should this be Borrow? or AsRef?
    type Vector: ?Sized;

    /// Return the non-nan score of the two vectors. Larger values are better.
    fn score(&self, a: &Self::Vector, b: &Self::Vector) -> NotNan<f32>;
}

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

/// Scores an f32 query vector against bit vectors.
/// Remaps bit vectors into f32 space and scores for higher fidelity output.
// XXX generalize this
pub struct F32xBitEuclideanQueryScorer<'a> {
    query: &'a [f32],
}

impl<'a> F32xBitEuclideanQueryScorer<'a> {
    pub fn new(query: &'a [f32]) -> Self {
        Self { query }
    }
}

const DECODE_4_BITS: [[f32; 4]; 16] = [
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

impl<'a> QueryScorer for F32xBitEuclideanQueryScorer<'a> {
    type Vector = [u8];

    fn score(&self, a: &Self::Vector) -> NotNan<f32> {
        // TODO: consider using slice.array_chunks()
        // TODO: simsimd may not be buying me all that much.
        let dist: f32 = self
            .query
            .chunks(8)
            .zip(a)
            .map(|(f, b)| {
                let mut db = [0f32; 8];
                db[0..4].copy_from_slice(&DECODE_4_BITS[*b as usize & 0xf]);
                db[4..8].copy_from_slice(&DECODE_4_BITS[*b as usize >> 4]);
                SpatialSimilarity::l2sq(f, &db).unwrap() as f32
            })
            .sum();
        NotNan::new(1.0 / (1.0 + dist)).unwrap()
    }
}
