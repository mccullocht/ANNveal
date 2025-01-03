use smallvec::{smallvec, SmallVec};

/// Tracks a set of `usize` up to a fixed value.
///
/// This is useful for small sets, particularly where there is a random access requirement.
pub(crate) struct FixedBitSet {
    rep: SmallVec<[u64; 7]>,
    len: usize,
}

impl FixedBitSet {
    /// Create a new set enough capacit to set up to the `i`th entry.
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            rep: smallvec![0u64; (capacity + 63) / 64],
            len: 0usize,
        }
    }

    /// Get the `i`th etnry.
    pub(crate) fn get(&self, i: usize) -> bool {
        self.rep[i / 64] & (1u64 << (i % 64)) > 0
    }

    /// Set the `i`th entry and return true if the value was inserted.
    pub(crate) fn set(&mut self, i: usize) -> bool {
        let w = &mut self.rep[i / 64];
        let m = 1u64 << (i % 64);
        if *w & m == 0 {
            *w |= m;
            self.len += 1;
            true
        } else {
            false
        }
    }

    /// Return an iterator over the set bits.
    pub(crate) fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.rep
            .as_slice()
            .iter()
            .enumerate()
            .flat_map(|(b, w)| WordSetIter(*w).map(move |x| b + x))
    }

    /// Get the number of set elements in the set.
    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

/// Yields the set of bits set in the word as an `Iterator<Item=usize>``
struct WordSetIter(u64);

impl Iterator for WordSetIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 > 0 {
            let i = self.0.trailing_zeros();
            self.0 ^= 1 << i;
            Some(i as usize)
        } else {
            None
        }
    }
}
