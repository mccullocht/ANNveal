use std::{
    borrow::Borrow,
    cell::RefCell,
    fmt::Debug,
    io::Write,
    iter::FusedIterator,
    num::NonZero,
    ops::{AddAssign, Deref, Range},
    sync::{
        atomic::{AtomicU64, Ordering},
        RwLock, RwLockReadGuard,
    },
};

use ahash::AHashSet;
use crossbeam_skiplist::SkipSet;
use ordered_float::NotNan;
use rayon::{
    current_num_threads,
    iter::{IntoParallelIterator, ParallelIterator},
};
use thread_local::ThreadLocal;

use crate::{
    graph::{EdgePruner, Graph, Neighbor, NeighborSet},
    scorer::{DefaultQueryScorer, QueryScorer, VectorScorer},
    store::{MeanVectorStore, VectorStore},
    utils::FixedBitSet,
};

/// A `Neighbor` annotated with an additional bit to indicate if it has been visited yet.
#[derive(Debug)]
struct NeighborResult {
    neighbor: Neighbor,
    visited: bool,
}

impl From<Neighbor> for NeighborResult {
    fn from(value: Neighbor) -> Self {
        Self {
            neighbor: value,
            visited: false,
        }
    }
}

/// An ordered set of `NeighborResult`s as a hybrid priority queue.
///
/// Results are ordered by `Neighbor` value and the set is capped to a fixed capacity. This queue
/// also has a mechansim to read the best unvisited neighbor, a core part of the Vamana search
/// algorithm.
#[derive(Debug)]
struct NeighborResultSet {
    results: Vec<NeighborResult>,
    next_unvisited: usize,
}

impl NeighborResultSet {
    fn new(capacity: usize) -> Self {
        Self {
            results: Vec::with_capacity(capacity),
            next_unvisited: 0,
        }
    }

    fn add(&mut self, neighbor: Neighbor) {
        if self.results.len() >= self.results.capacity()
            && neighbor >= self.results.last().unwrap().neighbor
        {
            return;
        }

        if self.results.len() >= self.results.capacity() {
            self.results.pop();
        }
        let insert_idx = match self.results.binary_search_by_key(&neighbor, |r| r.neighbor) {
            Ok(_) => return,
            Err(idx) => idx,
        };
        self.results.insert(insert_idx, neighbor.into());
        if insert_idx < self.next_unvisited {
            self.next_unvisited = insert_idx;
        }
    }

    fn best_unvisited(&mut self) -> Option<Neighbor> {
        if self.next_unvisited >= self.results.len() {
            return None;
        }

        let best_index = self.next_unvisited;
        self.next_unvisited = self
            .results
            .iter()
            .enumerate()
            .skip(self.next_unvisited + 1)
            .find_map(|(i, r)| if r.visited { None } else { Some(i) })
            .unwrap_or(self.results.len());
        let best = &mut self.results[best_index];
        best.visited = true;
        Some(best.neighbor)
    }

    fn clear(&mut self) {
        self.results.clear();
        self.next_unvisited = 0;
    }
}

impl From<NeighborResultSet> for Vec<Neighbor> {
    fn from(value: NeighborResultSet) -> Self {
        value.results.into_iter().map(|r| r.neighbor).collect()
    }
}

#[derive(Default, Debug)]
pub struct GraphSearchStats {
    /// Number of nodes seen and scored for navigation.
    pub seen: usize,
    /// Number of nodes visited as part of result set calculation.
    pub visited: usize,
}

impl AddAssign for GraphSearchStats {
    fn add_assign(&mut self, rhs: Self) {
        self.seen += rhs.seen;
        self.visited += rhs.visited;
    }
}

impl std::fmt::Display for GraphSearchStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "seen {} visited {}", self.seen, self.visited)
    }
}

#[derive(Debug)]
pub struct GraphSearcher {
    results: NeighborResultSet,
    seen: AHashSet<usize>,
    visited: usize,
}

impl GraphSearcher {
    /// Create a new GraphSearcher that will return k results searching.
    pub fn new(k: NonZero<usize>) -> Self {
        Self {
            results: NeighborResultSet::new(k.into()),
            seen: AHashSet::new(),
            visited: 0,
        }
    }

    /// Search `graph` over vector `store` for `query` and return the `k` nearest neighbors in
    /// descending order by score.
    pub fn search<G, V, Q>(&mut self, graph: &G, store: &V, query_scorer: &Q) -> NeighborSet
    where
        G: Graph,
        Q: QueryScorer,
        V: VectorStore<Vector = Q::Vector>,
    {
        self.search_internal(graph, store, query_scorer);
        NeighborSet(self.results.results.iter().map(|r| r.neighbor).collect())
    }

    fn search_internal<G, V, Q>(&mut self, graph: &G, store: &V, query_scorer: &Q)
    where
        G: Graph,
        Q: QueryScorer,
        V: VectorStore<Vector = Q::Vector>,
    {
        if graph.entry_point().is_none() {
            return;
        }
        let entry_point = graph.entry_point().unwrap();
        let score = query_scorer.score(&store[entry_point]);
        self.seen.insert(entry_point);
        self.results.add((entry_point as u32, score).into());

        while let Some(candidate) = self.results.best_unvisited() {
            self.visited += 1;
            for neighbor_id in graph.neighbors_iter(candidate.id as usize) {
                // skip candidates we've already visited.
                if !self.seen.insert(neighbor_id) {
                    continue;
                }

                let score = query_scorer.score(&store[neighbor_id]);
                self.results.add((neighbor_id as u32, score).into());
            }
        }
    }

    pub fn stats(&self) -> GraphSearchStats {
        GraphSearchStats {
            seen: self.seen.len(),
            visited: self.visited,
        }
    }

    pub fn clear(&mut self) {
        self.results.clear();
        self.seen.clear();
        self.visited = 0;
    }
}

/// `GraphBuilder` is used to build a vamana graph.
pub struct GraphBuilder<'a, V, S, C> {
    max_degree: usize,
    beam_width: NonZero<usize>,
    alpha: f32,
    store: &'a V,
    scorer: S,
    mean_vector: C,

    graph: MutableGraph,
    searcher: ThreadLocal<RefCell<GraphSearcher>>,
    in_flight: SkipSet<usize>,
}

impl<'a, V, S, C> GraphBuilder<'a, V, S, C>
where
    V: MeanVectorStore + Sync + Send,
    S: VectorScorer<Vector = V::Vector> + Sync + Send,
    C: Send + Sync + Borrow<V::Vector>,
    V::Vector: ToOwned<Owned = C>,
{
    /// Create a new graph builder.
    ///
    /// The graph is constructed by reading input vectors from `store` and
    /// scoring them using `scorer`.
    ///
    /// * `max_degree` controls the number of outbound edges from each node.
    /// * `beam_width` control the number of candidates in the initial search
    ///   when selecting edges.
    /// * `alpha` is a hyper parameter that loosens filtering on longer edges
    ///   when pruning the graph.
    pub fn new(
        max_degree: NonZero<usize>,
        beam_width: NonZero<usize>,
        alpha: f32,
        store: &'a V,
        scorer: S,
    ) -> Self {
        u32::try_from(store.len()).expect("ordinal count limited to 32 bits");
        let mean_vector = store.mean_vector();
        Self {
            max_degree: max_degree.into(),
            beam_width,
            alpha,
            store,
            scorer,
            mean_vector,
            graph: MutableGraph::new(store.len(), max_degree),
            searcher: ThreadLocal::with_capacity(current_num_threads()),
            in_flight: SkipSet::new(),
        }
    }

    /// Add all nodes in `node_id_range` to the graph.
    /// *Panics* if `node_id_range` is out of range.
    #[allow(dead_code)]
    pub fn add_nodes(&self, node_id_range: Range<usize>) {
        self.add_nodes_with_progress(node_id_range, || {})
    }

    /// Add all nodes in `node_id_range` to the graph and call `progress` when each completes.
    /// *Panics* if `node_id_range` is out of range.
    pub fn add_nodes_with_progress<U>(&self, node_id_range: Range<usize>, progress: U)
    where
        U: Fn() + Send + Sync,
    {
        node_id_range.into_par_iter().for_each(|n| {
            self.add_node(n);
            progress();
        })
    }

    /// Add a single vector from the backing store.
    /// *Panics* if `node` is out of range.
    fn add_node(&self, node: usize) {
        assert!(node < self.store.len());
        let query = &self.store[node];
        let centroid_neighbor = Neighbor::from((
            node as u32,
            self.scorer.score(query, self.mean_vector.borrow()),
        ));
        if self.graph.try_set_entry_point(centroid_neighbor) {
            return;
        }

        let mut searcher = self
            .searcher
            .get_or(|| RefCell::new(GraphSearcher::new(self.beam_width)))
            .borrow_mut();
        searcher.clear();
        // Update the set of in-flight nodes to include this one. All in-flight nodes will be added
        // to the visited set to ensure we generate links.
        self.in_flight.insert(node);
        // Initialize seen set with this node so don't score ourselves if a concurrent neighbor
        // yields a pointer to use during a search. Similarly, add all other concurrent nodes to the
        // result set so we don't accidentally omit them.
        searcher.seen.insert(node);
        for n in self
            .in_flight
            .iter()
            .filter(|i| !i.is_removed() && *i.value() != node)
        {
            searcher.results.add(Neighbor::from((
                *n as u32,
                self.scorer.score(query, &self.store[*n]),
            )));
        }
        let query_scorer = DefaultQueryScorer::new(&self.store[node], &self.scorer);
        // The paper uses a "visisted" set instead of the result set here. The visited set is a
        // super set of the result set and only has value if we think pruning would remove so many
        // edges that we need several hundred additional nodes to properly saturate the graph. In
        // practice this is not an issue.
        let mut neighbors = searcher.search(&self.graph, self.store, &query_scorer);
        self.prune(0, &mut neighbors);
        for n in self.graph.insert_neighbors(node, neighbors, self) {
            self.graph
                .insert_neighbor(n.id as usize, Neighbor::from((node as u32, n.score)), self);
        }

        self.in_flight.remove(&node);
        self.graph.maybe_update_entry_point(centroid_neighbor);
    }

    /// Finish graph construction and return the graph.
    #[allow(dead_code)]
    pub fn finish(self) -> MutableGraph {
        self.finish_with_progress(|| {})
    }

    /// Finish graph construction and return the graph.
    pub fn finish_with_progress<U>(self, progress: U) -> MutableGraph
    where
        U: Fn() + Send + Sync,
    {
        (0..self.store.len()).into_par_iter().for_each(|i| {
            self.graph.maybe_prune_neighbors(i, &self);
            progress();
        });

        self.graph
    }
}

impl<V, S, C> EdgePruner for GraphBuilder<'_, V, S, C>
where
    V: VectorStore + Sync + Send,
    S: VectorScorer<Vector = V::Vector> + Sync + Send,
    C: Send + Sync + Borrow<V::Vector>,
    V::Vector: ToOwned<Owned = C>,
{
    /// Prune edges based on the RobustPrune algorithm in the paper.
    ///
    /// This should yield the same results as the algorithm in the paper but is optimized:
    /// * Edges may be added to the list without immediately pruning; `first_unpruned` is the first
    ///   such node added that way and we automatically include any nodes before this index.
    /// * When considering the RNG rule for a candidate we consider the set of selected nodes
    ///   instead the set of other potential candidates. This greatly reduces the amount of scoring
    ///   we do since we only have to consider up to `max_degree` neighbors rather than `beam_width`
    fn prune(&self, first_unpruned: usize, edges: &mut NeighborSet) {
        // Create a selected set and initialize it with all the edges that have already been pruned.
        let mut selected = FixedBitSet::new(edges.len());
        for i in 0..first_unpruned {
            selected.set(i);
        }

        let mut cur_alpha = NotNan::new(1.0f32).expect("not NaN constant");
        let max_alpha = NotNan::new(self.alpha).expect("not NaN constant");
        while cur_alpha <= max_alpha && selected.len() < self.max_degree {
            for (i, n) in edges.iter().enumerate().skip(first_unpruned) {
                if selected.get(i) {
                    continue;
                }

                let n_vector = &self.store[n.id as usize];
                // Check if this vector is closer to any already selected neighbor than it is to
                // the node itself. If it is not, then select this node.
                if !selected
                    .iter()
                    .take_while(|a| *a < i)
                    .any(|a| self.scorer.score(n_vector, &self.store[a]) > n.score * cur_alpha)
                {
                    selected.set(i);
                    if selected.len() >= self.max_degree {
                        break;
                    }
                }
            }

            cur_alpha *= 1.2;
        }

        edges.prune(&selected);
    }
}

#[derive(Debug)]
struct MutableGraphNode {
    edges: NeighborSet,
    first_unpruned: usize,
}

impl MutableGraphNode {
    /// Create a new graph with the given edge capacity.
    fn with_capacity(capacity: usize) -> Self {
        Self {
            edges: NeighborSet::with_capacity(capacity),
            first_unpruned: 0,
        }
    }

    /// Add a neighbor to the edge list. Returns the number of edges on the node.
    fn add_neighbor(&mut self, neighbor: Neighbor) -> usize {
        if let Some(i) = self.edges.insert(neighbor) {
            self.first_unpruned = std::cmp::min(self.first_unpruned, i);
        }
        self.len()
    }

    /// Add a set of neighbors to the edge list. Returns the number of edges on the node.
    /// REQUIRES: `neighbors` is a pruned edge set.
    fn add_neighbors(&mut self, mut neighbors: NeighborSet) -> usize {
        if self.edges.len() < neighbors.len() {
            // Try to merge the smaller set into the larger set.
            std::mem::swap(&mut self.edges, &mut neighbors);
            self.edges.0.shrink_to(neighbors.capacity());
            self.first_unpruned = self.edges.len();
        }

        for n in neighbors {
            if let Some(i) = self.edges.insert(n) {
                self.first_unpruned = std::cmp::min(self.first_unpruned, i);
            }
        }

        self.edges.len()
    }

    /// Run the pruner on this node.
    fn prune(&mut self, pruner: &impl EdgePruner) {
        pruner.prune(self.first_unpruned, &mut self.edges);
        self.first_unpruned = self.edges.len();
    }

    fn iter(&self) -> impl Iterator<Item = &Neighbor> {
        self.edges.iter()
    }

    fn len(&self) -> usize {
        self.edges.len()
    }

    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}

const INITIAL_ENTRY_POINT: Neighbor = Neighbor {
    id: u32::MAX,
    score: unsafe { NotNan::new_unchecked(f32::MIN) },
};

#[derive(Debug)]
pub struct MutableGraph {
    max_degree: usize,
    entry_point: AtomicU64,
    nodes: Vec<RwLock<MutableGraphNode>>,
}

impl MutableGraph {
    fn new(len: usize, max_degree: NonZero<usize>) -> Self {
        let mut nodes = Vec::with_capacity(len);
        nodes.resize_with(len, || {
            RwLock::new(MutableGraphNode::with_capacity(max_degree.get() * 2))
        });
        Self {
            max_degree: max_degree.get(),
            entry_point: AtomicU64::new(INITIAL_ENTRY_POINT.into()),
            nodes,
        }
    }

    fn try_set_entry_point(&self, entry_point: Neighbor) -> bool {
        self.entry_point
            .compare_exchange(
                INITIAL_ENTRY_POINT.into(),
                entry_point.into(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            )
            .is_ok()
    }

    fn maybe_update_entry_point(&self, entry_point: Neighbor) {
        let mut current = Neighbor::from(self.entry_point.load(Ordering::Relaxed));
        while entry_point.score > current.score {
            match self.entry_point.compare_exchange_weak(
                current.into(),
                entry_point.into(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(raw_ep) => current = raw_ep.into(),
            }
        }
    }

    fn insert_neighbor<P>(&self, node_id: usize, neighbor: Neighbor, pruner: &P)
    where
        P: EdgePruner,
    {
        let mut node = self.nodes[node_id].write().unwrap();
        assert_ne!(node_id, neighbor.id as usize);
        if node.add_neighbor(neighbor) == self.max_degree * 2 {
            node.prune(pruner);
        }
    }

    /// Insert all of the neighbors in the set as edges on `node_id`, then prunes edges if needed.
    /// Returns the set of neighbors after any pruning.
    fn insert_neighbors<P>(&self, node_id: usize, neighbors: NeighborSet, pruner: &P) -> NeighborSet
    where
        P: EdgePruner,
    {
        let mut node = self.nodes[node_id].write().unwrap();
        if node.add_neighbors(neighbors) >= self.max_degree * 2 {
            node.prune(pruner);
        }
        node.edges.clone()
    }

    fn maybe_prune_neighbors<P>(&self, node_id: usize, pruner: &P)
    where
        P: EdgePruner,
    {
        let mut node = self.nodes[node_id].write().unwrap();
        if node.len() >= self.max_degree {
            node.prune(pruner);
        }
    }

    pub fn write(&self, out: &mut impl Write) -> std::io::Result<()> {
        out.write_all(&(self.nodes.len() as u32).to_le_bytes())?;
        out.write_all(&((self.entry_point.load(Ordering::Relaxed) as u32).to_le_bytes()))?;
        // Write the number of nodes at the start of each edge and at the end. This way we will
        // write self.nodes.len() + 1 values for edge extents.
        out.write_all(&0u64.to_le_bytes())?;
        // This is ~8MB per ~1M nodes
        let mut num_edges = 0u64;
        for n in self.nodes.iter().map(|n| n.read().unwrap()) {
            num_edges += n.len() as u64;
            out.write_all(&num_edges.to_le_bytes())?;
        }
        // This is ~120MB per ~1M nodes. If we did simple bit-width compression it would be ~75MB.
        for n in self.nodes.iter().map(|n| n.read().unwrap()) {
            for e in n.iter() {
                out.write_all(&e.id.to_le_bytes())?;
            }
        }
        Ok(())
    }
}

pub struct NeighborNodeIterator<'a> {
    guard: RwLockReadGuard<'a, MutableGraphNode>,
    it: Range<usize>,
}

impl<'a> NeighborNodeIterator<'a> {
    fn new(neighbors: &'a RwLock<MutableGraphNode>) -> Self {
        let guard = neighbors.read().unwrap();
        let it = 0..guard.len();
        Self { guard, it }
    }
}

impl Iterator for NeighborNodeIterator<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|i| self.guard.edges[i].id as usize)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl ExactSizeIterator for NeighborNodeIterator<'_> {}

impl FusedIterator for NeighborNodeIterator<'_> {}

impl Graph for MutableGraph {
    type NeighborEdgeIterator<'c>
        = NeighborNodeIterator<'c>
    where
        Self: 'c;

    fn entry_point(&self) -> Option<usize> {
        let entry_point: Neighbor = self.entry_point.load(Ordering::Relaxed).into();
        if entry_point != INITIAL_ENTRY_POINT {
            Some(entry_point.id as usize)
        } else {
            None
        }
    }

    fn neighbors_iter(&self, ord: usize) -> Self::NeighborEdgeIterator<'_> {
        NeighborNodeIterator::new(&self.nodes[ord])
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }
}

/// In immutable graph implementation that can be loaded from a byte slice.
pub struct ImmutableMemoryGraph<D> {
    // doing weird stuff with static slices to allow self referencing in an unprincipled way.
    #[allow(dead_code)]
    data: D,
    entry_point: Option<u32>,
    node_extents: &'static [u64],
    edges: &'static [u32],
}

impl<D> ImmutableMemoryGraph<D> {
    pub fn new(data: D) -> Result<Self, &'static str>
    where
        D: Deref<Target = [u8]>,
    {
        let rep: &'static [u8] = unsafe { std::slice::from_raw_parts(data.as_ptr(), data.len()) };
        if rep.len() < 8 {
            return Err("rep too short to contain a valid graph");
        }
        let (num_nodes, rep) = Self::decode_u32(rep);
        if num_nodes == 0 {
            return Ok(Self {
                data,
                entry_point: None,
                node_extents: &[],
                edges: &[],
            });
        }
        let (entry_point, rep) = Self::decode_u32(rep);
        if entry_point >= num_nodes {
            return Err("invalid entry point");
        }
        let node_extent_bytes = std::mem::size_of::<u64>() * (num_nodes as usize + 1);
        if rep.len() < node_extent_bytes {
            return Err("rep too short to contain all node extents");
        }
        let (prefix, node_extents, suffix) = unsafe { rep[0..node_extent_bytes].align_to::<u64>() };
        if !prefix.is_empty() || !suffix.is_empty() {
            return Err("rep does not align to u64 for node_extents");
        }
        let num_edges = node_extents.last().unwrap().to_le();
        let edges_bytes = num_edges as usize * std::mem::size_of::<u32>();
        if rep.len() < node_extent_bytes + edges_bytes {
            return Err("rep too short to contain all edges");
        }
        let (prefix, edges, suffix) =
            unsafe { rep[node_extent_bytes..(node_extent_bytes + edges_bytes)].align_to::<u32>() };
        if !prefix.is_empty() || !suffix.is_empty() {
            return Err("rep does not align to u32 for edges");
        }
        Ok(Self {
            data,
            entry_point: Some(entry_point),
            node_extents,
            edges,
        })
    }

    fn decode_u32(rep: &[u8]) -> (u32, &[u8]) {
        let (u32rep, rep) = rep.split_at(std::mem::size_of::<u32>());
        (u32::from_le_bytes(u32rep.try_into().unwrap()), rep)
    }
}

pub struct ImmutableEdgeIterator<'a> {
    it: std::slice::Iter<'a, u32>,
}

impl ImmutableEdgeIterator<'_> {
    fn new<D>(graph: &ImmutableMemoryGraph<D>, node: usize) -> Self {
        let begin = graph.node_extents[node] as usize;
        let end = graph.node_extents[node + 1] as usize;
        Self {
            it: graph.edges[begin..end].iter(),
        }
    }
}

impl Iterator for ImmutableEdgeIterator<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|e| *e as usize)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl ExactSizeIterator for ImmutableEdgeIterator<'_> {}

impl FusedIterator for ImmutableEdgeIterator<'_> {}

impl<D> Graph for ImmutableMemoryGraph<D> {
    type NeighborEdgeIterator<'c>
        = ImmutableEdgeIterator<'c>
    where
        Self: 'c;

    fn entry_point(&self) -> Option<usize> {
        self.entry_point.map(|p| p as usize)
    }

    fn neighbors_iter(&self, node: usize) -> Self::NeighborEdgeIterator<'_> {
        ImmutableEdgeIterator::new(self, node)
    }

    fn len(&self) -> usize {
        self.node_extents.len() - 1
    }
}

#[cfg(test)]
mod test {
    use std::{cmp::Ordering, num::NonZero};

    use ordered_float::NotNan;

    use crate::{
        graph::Graph,
        scorer::{DefaultQueryScorer, VectorScorer},
        store::MeanVectorStore,
    };

    use rayon::prelude::*;

    use super::{GraphBuilder, GraphSearcher, ImmutableMemoryGraph, MutableGraph, VectorStore};

    impl VectorStore for Vec<u64> {
        type Vector = u64;

        fn len(&self) -> usize {
            self.len()
        }

        fn iter(&self) -> impl ExactSizeIterator<Item = &Self::Vector> {
            self.as_slice().iter()
        }
    }

    impl MeanVectorStore for Vec<u64> {
        fn mean_vector(&self) -> u64 {
            let mut counts = [0usize; 64];
            for i in 0..self.len() {
                let mut v = self[i];
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

    #[derive(Copy, Clone)]
    struct Hamming64;

    impl VectorScorer for Hamming64 {
        type Vector = u64;

        fn score(&self, a: &Self::Vector, b: &Self::Vector) -> ordered_float::NotNan<f32> {
            let distance = (a ^ b).count_ones();
            let score = ((u64::BITS - distance) as f32) / u64::BITS as f32;
            NotNan::new(score).expect("constant")
        }
    }

    fn make_builder<'a, V: MeanVectorStore<Vector = u64> + Sync + Send>(
        store: &'a V,
    ) -> GraphBuilder<'a, V, Hamming64, u64> {
        GraphBuilder::new(
            NonZero::new(4).expect("constant"),
            NonZero::new(10).expect("constant"),
            1.2f32,
            store,
            Hamming64,
        )
    }

    fn build_graph<B: MeanVectorStore<Vector = u64> + Sync + Send>(store: &B) -> MutableGraph {
        let builder = GraphBuilder::new(
            NonZero::new(4).expect("constant"),
            NonZero::new(10).expect("constant"),
            1.2f32,
            store,
            Hamming64,
        );
        (0..store.len()).into_par_iter().panic_fuse().for_each(|i| {
            builder.add_node(i);
        });
        builder.finish()
    }

    fn get_neighbors(graph: &impl Graph, node: usize) -> Vec<usize> {
        let mut edges: Vec<usize> = graph.neighbors_iter(node).collect();
        edges.sort();
        edges
    }

    #[test]
    fn empty_store_and_graph() {
        let store: Vec<u64> = vec![];
        let graph: MutableGraph = build_graph(&store);
        assert_eq!(graph.len(), 0);
        assert_eq!(graph.entry_point(), None);
    }

    #[test]
    fn single_node() {
        let store: Vec<u64> = vec![0, 1, 2, 3];
        let builder = make_builder(&store);
        builder.add_node(2);
        let graph: MutableGraph = builder.finish();
        assert_eq!(graph.len(), 4);
        assert_eq!(graph.entry_point(), Some(2));
    }

    #[test]
    fn tiny_graph() {
        let store: Vec<u64> = (0u64..5u64).collect();
        let graph: MutableGraph = build_graph(&store);
        assert_eq!(graph.len(), 5);
        assert_eq!(get_neighbors(&graph, 0), vec![1, 2, 3, 4]);
        assert_eq!(get_neighbors(&graph, 1), vec![0, 2, 3, 4]);
        assert_eq!(get_neighbors(&graph, 2), vec![0, 1, 3, 4]);
        assert_eq!(get_neighbors(&graph, 3), vec![0, 1, 2, 4]);
        assert_eq!(get_neighbors(&graph, 4), vec![0, 1, 2, 3]);
    }

    #[test]
    fn pruned_graph() {
        let store: Vec<u64> = (0u64..16u64).collect();
        let graph: MutableGraph = build_graph(&store);
        assert_eq!(get_neighbors(&graph, 0), vec![1, 2, 4, 8]);
        assert_eq!(get_neighbors(&graph, 7), vec![3, 5, 6, 15]);
        assert_eq!(get_neighbors(&graph, 8), vec![0, 9, 10, 12]);
        assert_eq!(get_neighbors(&graph, 15), vec![7, 11, 13, 14]);
        assert_eq!(graph.entry_point(), Some(5));
    }

    #[test]
    fn search_graph() {
        let store: Vec<u64> = (0u64..16u64).collect();
        let graph: MutableGraph = build_graph(&store);
        let mut searcher = GraphSearcher::new(NonZero::new(8).expect("constant"));
        let query_scorer = DefaultQueryScorer::new(&64, &Hamming64);
        let results = searcher.search(&graph, &store, &query_scorer);
        assert_eq!(
            results
                .into_iter()
                .map(|n| (n.id, n.score.into_inner()))
                .collect::<Vec<_>>(),
            vec![
                (0, 0.984375),
                (1, 0.96875),
                (2, 0.96875),
                (4, 0.96875),
                (8, 0.96875),
                (3, 0.953125),
                (5, 0.953125),
                (6, 0.953125),
            ]
        );
    }

    #[test]
    fn serialize_graph() {
        let store: Vec<u64> = (0u64..16u64).collect();
        let mutable_graph: MutableGraph = build_graph(&store);
        let mut serialized_graph = Vec::new();
        mutable_graph.write(&mut serialized_graph).unwrap();
        let immutable_graph = ImmutableMemoryGraph::new(serialized_graph).unwrap();

        assert_eq!(get_neighbors(&immutable_graph, 0), vec![1, 2, 4, 8]);
        assert_eq!(get_neighbors(&immutable_graph, 7), vec![3, 5, 6, 15]);
        assert_eq!(get_neighbors(&immutable_graph, 8), vec![0, 9, 10, 12]);
        assert_eq!(get_neighbors(&immutable_graph, 15), vec![7, 11, 13, 14]);
        let mut searcher = GraphSearcher::new(NonZero::new(8).expect("constant"));
        let query_scorer = DefaultQueryScorer::new(&64, &Hamming64);
        let results = searcher.search(&immutable_graph, &store, &query_scorer);
        assert_eq!(
            results
                .into_iter()
                .map(|n| (n.id, n.score.into_inner()))
                .collect::<Vec<_>>(),
            vec![
                (0, 0.984375),
                (1, 0.96875),
                (2, 0.96875),
                (4, 0.96875),
                (8, 0.96875),
                (3, 0.953125),
                (5, 0.953125),
                (6, 0.953125),
            ]
        );
    }
}
