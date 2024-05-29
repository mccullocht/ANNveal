use rand::Rng;
use rand_pcg::Pcg64Mcg;

use crate::store::VectorStore;

const ZERO_CHUNK: [f32; 8] = [0.0f32; 8];

/// Produce an iterator that maps each dimension in `vector` to a 0 or 1 value.
pub fn binary_quantize_f32(vector: &[f32]) -> impl Iterator<Item = u8> + '_ {
    vector
        .chunks(8)
        .map(|v| binary_quantize_f32_median_chunk(v, &ZERO_CHUNK))
}

/// Computes a vector containing the median point in every dimension for an adaptive binary
/// quantization scheme.
pub fn sampled_binary_quantization_median<S>(store: &S, samples: usize) -> Vec<f32>
where
    S: VectorStore<Vector = [f32]>,
{
    let mut rng = Pcg64Mcg::new(0xcafef00dd15ea5e5);
    let mut sampled: Vec<usize> = (0..std::cmp::min(samples, store.len())).collect();
    for i in sampled.len()..store.len() {
        let j = rng.gen_range(0..i);
        if j < sampled.len() {
            sampled[j] = i;
        }
    }
    sampled.sort();

    let mut data = vec![0.0f32; sampled.len() * store.dimensions()];
    for (i, s) in sampled.iter().enumerate() {
        for (j, v) in store.get(*s).iter().enumerate() {
            data[j * sampled.len() + i] = *v;
        }
    }
    data.chunks_mut(sampled.len())
        .map(|d| {
            d.sort_by(|a, b| a.partial_cmp(b).unwrap());
            d[d.len() / 2]
        })
        .collect()
}

pub fn binary_quantize_f32_median<'a, 'b: 'a>(
    vector: &'a [f32],
    median: &'b [f32],
) -> impl Iterator<Item = u8> + 'a
where
    'a: 'b,
{
    vector
        .chunks(8)
        .zip(median.chunks(8))
        .map(move |(v, m)| binary_quantize_f32_median_chunk(v, m))
}

fn binary_quantize_f32_median_chunk(vector_chunk: &[f32], median_chunk: &[f32]) -> u8 {
    vector_chunk
        .iter()
        .zip(median_chunk.iter())
        .enumerate()
        .filter_map(|(i, (v, e))| if *v > *e { Some(1u8 << i) } else { None })
        .fold(0u8, |a, b| a | b)
}
