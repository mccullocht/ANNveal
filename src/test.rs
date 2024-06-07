use std::cmp::Ordering;

use ordered_float::NotNan;

use crate::{scorer::VectorScorer, store::VectorStore};

impl VectorStore for Vec<u64> {
    type Vector = u64;

    fn get(&self, i: usize) -> &Self::Vector {
        &self[i]
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn mean_vector(&self) -> u64 {
        let mut counts = [0usize; 64];
        for i in 0..self.len() {
            let mut v = *self.get(i);
            while v != 0 {
                let j = v.trailing_zeros() as usize;
                counts[j] += 1;
                v ^= 1 << j;
            }
        }

        let mut mean = 0u64;
        for (i, c) in counts.into_iter().enumerate() {
            let d = match c.cmp(&(self.len() / 2)) {
                Ordering::Less => false,
                Ordering::Greater => true,
                Ordering::Equal => {
                    if self.len() & 1 == 0 {
                        i % 2 == 0 // on tie, choose an element at random
                    } else {
                        false // middle is X.5 and integers round down, so like Less
                    }
                }
            };
            if d {
                mean |= 1u64 << i;
            }
        }
        mean
    }
}

pub(crate) struct Hamming64;

impl VectorScorer for Hamming64 {
    type Vector = u64;

    fn score(&self, a: &Self::Vector, b: &Self::Vector) -> ordered_float::NotNan<f32> {
        let distance = (a ^ b).count_ones();
        let score = ((u64::BITS - distance) as f32) / u64::BITS as f32;
        NotNan::new(score).expect("constant")
    }
}
