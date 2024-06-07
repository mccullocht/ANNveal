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

/// Computes vectors containing the 25th, median, and 75th percentile values in each dimension from
/// a sample. The median is used to improve the quality of quantization, the other percentiles are
/// used to improve the quality of dequantization when scoring against a float vector.
pub fn sampled_binary_quantization_quartiles<S>(store: &S, samples: usize) -> [Vec<f32>; 3]
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
    let mut quartiles: [Vec<f32>; 3] = vec![Vec::with_capacity(store.dimensions()); 3]
        .try_into()
        .unwrap();
    let interval = sampled.len() / 4;
    for dim in data.chunks_mut(sampled.len()) {
        dim.sort_by(|a, b| a.partial_cmp(b).unwrap());
        quartiles[0].push(dim[interval]);
        quartiles[1].push(dim[interval * 2]);
        quartiles[2].push(dim[interval * 3]);
    }
    quartiles
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
