pub mod vamana;

use std::ops::Index;

use ordered_float::NotNan;

use crate::utils::FixedBitSet;

/// `Graph` provides an interface for searchers to use when traversing the graph.
pub trait Graph {
    type NeighborEdgeIterator<'c>: Iterator<Item = usize>
    where
        Self: 'c;

    // Returns the entry point to the graph if any nodes have been populated.
    fn entry_point(&self) -> Option<usize>;
    /// Return an iterate over the list of nodes neighboring `ord`.
    fn neighbors_iter(&self, i: usize) -> Self::NeighborEdgeIterator<'_>;
    /// Return the number of nodes in the graph.
    #[allow(dead_code)]
    fn len(&self) -> usize;
}

/// `EggePruner` provides an interface for reducing a node's edges during graph construction.
pub trait EdgePruner {
    /// Prune `edges`` based on policy defined by the pruner.
    /// `first_unpruned` is the index of the first edges that has not been seen by the pruner.
    fn prune(&self, first_unpruned: usize, edges: &mut NeighborSet);
}

/// Identify a node by id and (non-`NaN`) score against another node..
///
/// This is used as part of enighbor sets and result lists. By default comparisons include both
/// members; when sorting they are ordering in descending value by score followed by ascending
/// value by id.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Neighbor {
    pub id: u32,
    pub score: NotNan<f32>,
}

impl Neighbor {
    pub fn new(id: u32, score: NotNan<f32>) -> Self {
        Self { id, score }
    }
}

// FIXME remove this it's ridiculous.
impl From<(u32, NotNan<f32>)> for Neighbor {
    fn from(value: (u32, NotNan<f32>)) -> Self {
        Self {
            id: value.0,
            score: value.1,
        }
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .cmp(&other.score)
            .reverse()
            .then(self.id.cmp(&other.id))
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Pack a `Neighbor` into `u64` for use as part of an atomic.
///
/// This encoding is _not_ designed for ordering, just equality comparison.
impl From<Neighbor> for u64 {
    fn from(value: Neighbor) -> Self {
        (u64::from(value.score.to_bits()) << 32) | u64::from(value.id)
    }
}

/// Unpack a `Neighbor` from a `u64` value.
impl From<u64> for Neighbor {
    fn from(value: u64) -> Self {
        Neighbor {
            id: value as u32,
            score: NotNan::new(f32::from_bits((value >> 32) as u32)).unwrap(),
        }
    }
}

/// A sorted set of `Neighbor`s.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NeighborSet(pub(crate) Vec<Neighbor>);

impl NeighborSet {
    /// Return a new empty set.
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Return a new empty set with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    pub(crate) fn from_sorted(n: Vec<Neighbor>) -> Self {
        Self(n)
    }

    /// Return an iterator over the set.
    pub fn iter(&self) -> std::slice::Iter<'_, Neighbor> {
        self.0.iter()
    }

    /// Return the contents of the set as a slice.
    pub fn as_slice(&self) -> &[Neighbor] {
        self.0.as_slice()
    }

    /// Return the number of elements this set can fit without reallocation.
    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    /// Return the number of entries in this set.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Return true if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Insert a neighbor into the set. If the neighbor was inserted return its rank in the set,
    /// otherwise return None.
    /// NB: neighbors are only equivalent if they have the same id _and_ score.
    pub fn insert(&mut self, n: Neighbor) -> Option<usize> {
        if let Err(i) = self.0.binary_search(&n) {
            self.0.insert(i, n);
            Some(i)
        } else {
            None
        }
    }

    /// Keep the selected neighbors and discard the rest.
    pub(crate) fn prune(&mut self, selected: &FixedBitSet) {
        for (i, o) in selected.iter().zip(0..self.len()) {
            self.0.swap(i, o);
        }
        self.0.truncate(selected.len())
    }

    /// Remove all entries from the set.
    pub fn clear(&mut self) {
        self.0.clear();
    }
}

impl Default for NeighborSet {
    fn default() -> Self {
        Self::new()
    }
}

impl From<NeighborSet> for Vec<Neighbor> {
    fn from(value: NeighborSet) -> Self {
        value.0
    }
}

/// Convert a vector of neighbors to a `NeighborSet` by sorting it.
impl From<Vec<Neighbor>> for NeighborSet {
    fn from(mut value: Vec<Neighbor>) -> Self {
        value.sort();
        Self(value)
    }
}

impl IntoIterator for NeighborSet {
    type IntoIter = std::vec::IntoIter<Neighbor>;
    type Item = Neighbor;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Index<usize> for NeighborSet {
    type Output = Neighbor;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
