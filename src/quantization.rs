/// Produce an iterator that maps each dimension in `vector` to a 0 or 1 value.
pub fn binary_quantize_f32(vector: &[f32]) -> impl Iterator<Item = u8> + '_ {
    vector.chunks(8).map(binary_quantize_f32_chunk)
}

fn binary_quantize_f32_chunk(chunk: &[f32]) -> u8 {
    chunk
        .iter()
        .enumerate()
        .filter_map(|(i, e)| if *e > 0.0 { Some(1u8 << i) } else { None })
        .reduce(|a, b| a | b)
        .unwrap_or(0u8)
}
