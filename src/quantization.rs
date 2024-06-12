use crate::store::VectorStore;

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

/// Quantize f32 inputs into 2-8 bit outputs.
// XXX doesn't compute a correction, although lucene does not use this for euclidean distance.
#[derive(Copy, Clone, Debug)]
pub struct ScalarQuantizer {
    min_quantile: f32,
    max_quantile: f32,
    scale: f32,
    bits: usize,
}

impl ScalarQuantizer {
    /// Create a new quantizer with the given parameters.
    pub fn new(min_quantile: f32, max_quantile: f32, bits: usize) -> Self {
        assert!(min_quantile < max_quantile);
        assert!((1usize..=8).contains(&bits));

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

    /// Initialize a new quantizer from the contents of a store.
    pub fn from_store<S>(store: &S, bits: usize) -> Self
    where
        S: VectorStore<Vector = [f32]>,
    {
        // XXX missing any notion of confidence
        let mut min_quantile = f32::MAX;
        let mut max_quantile = f32::MIN;
        for v in store.iter() {
            for d in v.iter() {
                min_quantile = min_quantile.min(*d);
                max_quantile = max_quantile.max(*d);
            }
        }

        Self::new(min_quantile, max_quantile, bits)
    }

    /// Serialize the state of this quantizer to a 16 byte array.
    pub fn serialize(&self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[0..4].copy_from_slice(&self.min_quantile.to_le_bytes());
        out[4..8].copy_from_slice(&self.max_quantile.to_le_bytes());
        out[8..12].copy_from_slice(&(self.bits as u32).to_le_bytes());
        out[12..16].copy_from_slice(&0u32.to_le_bytes());
        out
    }

    /// Create a quantizer from a 16 byte array.
    pub fn deserialize(state: &[u8; 16]) -> Self {
        Self::new(
            f32::from_le_bytes(state[0..4].try_into().unwrap()),
            f32::from_le_bytes(state[4..8].try_into().unwrap()),
            u32::from_le_bytes(state[8..12].try_into().unwrap()) as usize,
        )
    }

    /// Quantize a single value into a u8 value.
    pub fn quantize(&self, f: f32) -> u8 {
        let raw =
            self.scale * (self.min_quantile.max(self.max_quantile.min(f)) - self.min_quantile);
        raw.round() as u8
    }

    /// Return the quantized version of the query vector.
    #[allow(dead_code)]
    pub fn quantize_query(&self, vector: &[f32]) -> Vec<u8> {
        let mut out = vec![0u8; self.output_vector_len(vector.len())];
        self.quantize_vector(vector, &mut out);
        out
    }

    /// Quantize a vector and emit it to the output vector.
    /// *Panics* if `out.len() < self.output_vector_len(vector.len())`
    pub fn quantize_vector(&self, vector: &[f32], out: &mut [u8]) {
        match self.bits.next_power_of_two() {
            2 => {
                for (c, o) in vector.chunks(4).zip(out.iter_mut()) {
                    *o = c
                        .iter()
                        .enumerate()
                        .map(|(i, d)| self.quantize(*d) << (i * 2))
                        .fold(0, |a, b| a | b);
                }
            }
            4 => {
                for (c, o) in vector.chunks(2).zip(out.iter_mut()) {
                    *o = c
                        .iter()
                        .enumerate()
                        .map(|(i, d)| self.quantize(*d) << (i * 4))
                        .fold(0, |a, b| a | b)
                }
            }
            8 => {
                for (i, o) in vector.iter().zip(out.iter_mut()) {
                    *o = self.quantize(*i);
                }
            }
            _ => unreachable!(),
        }
    }

    /// Returns the required vector length for an output vector with `dimensions`.
    pub fn output_vector_len(&self, dimensions: usize) -> usize {
        match self.bits().next_power_of_two() {
            2 => (dimensions + 3) / 4,
            4 => (dimensions + 1) / 2,
            8 => dimensions,
            _ => unreachable!(),
        }
    }

    /// Returns the number of bits a value quantizes into.
    pub fn bits(&self) -> usize {
        self.bits
    }
}
