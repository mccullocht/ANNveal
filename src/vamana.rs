// XXX filtered search
//
// we want to avoid scoring a lot of stuff we can't collect.
// * read edges of candidates and filter.
// * if there are too few neighbors (none?) automatically traverse neighbors-of-neighbors.
// * get greedy: insert edges that don't match the filter at a very low or partial score.
// * use a much longer candidate list.
// * choose an arbitrary entrypoint if the preset one cannot be used?

use std::{
    cell::RefCell,
    cmp::Reverse,
    collections::BinaryHeap,
    io::Write,
    iter::FusedIterator,
    num::NonZeroUsize,
    sync::{
        atomic::{AtomicUsize, Ordering},
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

/// Dense store of vector data, analogous to a `Vec``.
pub trait VectorStore {
    /// Type of the underlying vector data.
    type Vector: ?Sized;

    /// Obtain a reference to the contents of the vector by VectorId.
    /// *Panics* if `i`` is out of bounds.
    fn get(&self, i: usize) -> &Self::Vector;

    // XXX method to approximate centroid. would compute this value up front when building the
    // graph and periodically search to improve entry point.

    /// Return the total number of vectors in the store.
    fn len(&self) -> usize;
}

/// Trait for scoring vectors against one another.
// XXX maybe the args shouldn't be by ref so the vector type doesn't have to be unref slice?
pub trait Scorer {
    /// Type for the underlying vector data.
    type Vector: ?Sized;

    /// Return the non-nan score of the two vectors. Larger values are better.
    fn score(&self, a: &Self::Vector, b: &Self::Vector) -> NotNan<f32>;
}

/// Information about a neighbor of a node.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Neighbor {
    id: u32,
    score: NotNan<f32>,
}

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

/// Graph access.
pub trait Graph {
    type NeighborEdgeIterator<'c>: Iterator<Item = usize>
    where
        Self: 'c;

    // Returns the entry point to the graph if any nodes have been populated.
    fn entry_point(&self) -> Option<usize>;
    /// Return an iterate over the list of nodes neighboring `ord`.
    fn neighbors_iter(&self, i: usize) -> Self::NeighborEdgeIterator<'_>;
    /// Return the number of nodes in the graph.
    fn len(&self) -> usize;
}

#[derive(Debug)]
pub struct GraphSearcher {
    k: NonZeroUsize,
    candidates: BinaryHeap<Reverse<Neighbor>>,
    results: BinaryHeap<Neighbor>,
    seen: AHashSet<usize>,
    visited: Vec<Neighbor>,
}

impl GraphSearcher {
    /// Create a new GraphSearcher that will return k results searching.
    pub fn new(k: NonZeroUsize) -> Self {
        Self {
            k,
            candidates: BinaryHeap::with_capacity(k.into()),
            results: BinaryHeap::with_capacity(k.into()),
            seen: AHashSet::new(),
            visited: vec![],
        }
    }

    /// Search `graph` over vector `store` for `query` and return the `k` nearest neighbors in
    /// descending order by score.
    pub fn search<G, V, Q, S>(
        &mut self,
        graph: &G,
        store: &V,
        query: &Q,
        scorer: &S,
    ) -> Vec<Neighbor>
    where
        G: Graph,
        V: VectorStore<Vector = Q>,
        S: Scorer<Vector = Q>,
    {
        self.search_internal(graph, store, query, scorer, false);
        // NB: we might get some mileage out of using a min-max heap for this, if for no other
        // reason than avoiding Reverse and attendant unwrapping.
        self.results.clone().into_sorted_vec()
    }

    /// Search `graph` over vector `store` for `query` using `scorer` to compute scores.
    /// Access visited to view results in descending order by score.
    fn search_for_insertion<G, V, Q: ?Sized, S>(
        &mut self,
        graph: &G,
        store: &V,
        query: &Q,
        scorer: &S,
    ) where
        G: Graph,
        V: VectorStore<Vector = Q>,
        S: Scorer<Vector = Q>,
    {
        self.search_internal(graph, store, query, scorer, true);
        self.visited.sort();
    }

    fn search_internal<G, V, Q: ?Sized, S>(
        &mut self,
        graph: &G,
        store: &V,
        query: &Q,
        scorer: &S,
        collect_visited: bool,
    ) where
        G: Graph,
        V: VectorStore<Vector = Q>,
        S: Scorer<Vector = Q>,
    {
        if graph.entry_point().is_none() {
            return;
        }
        let entry_point = graph.entry_point().unwrap();
        let score = scorer.score(query, store.get(entry_point));
        self.seen.insert(entry_point);
        self.candidates
            .push(Reverse((entry_point as u32, score).into()));

        while let Some(Reverse(candidate)) = self.candidates.pop() {
            if !self.collect(candidate) {
                break;
            }
            if collect_visited {
                self.visited.push(candidate);
            }

            for neighbor_id in graph.neighbors_iter(candidate.id as usize) {
                // skip candidates we've already visited.
                if !self.seen.insert(neighbor_id) {
                    continue;
                }

                let score = scorer.score(query, store.get(neighbor_id));
                let neighbor = Neighbor::from((neighbor_id as u32, score));
                // Insert only neighbors that could possibly make it to the result set.
                // XXX this makes the candidates heap ~unbounded.
                // if we use a min-max heap here we could bound the size of the heap.
                if score >= self.min_score() {
                    self.candidates.push(Reverse(neighbor));
                }
            }
        }
    }

    fn collect(&mut self, result: Neighbor) -> bool {
        if self.results.len() < self.results.capacity() {
            self.results.push(result);
            true
        } else {
            let mut min_result = self.results.peek_mut().expect("results is not empty");
            if result.score > min_result.score {
                *min_result = result;
                true
            } else {
                false
            }
        }
    }

    fn min_score(&self) -> NotNan<f32> {
        if self.results.len() < self.results.capacity() {
            NotNan::new(f32::NEG_INFINITY).expect("negative infinity is not nan")
        } else {
            self.results.peek().expect("results not empty").score
        }
    }

    pub fn clear(&mut self) {
        self.candidates.clear();
        self.results.clear();
        self.seen.clear();
        self.visited.clear();
    }
}

/// `GraphBuilder` is used to build a vamana graph.
pub struct GraphBuilder<'a, V, S> {
    max_degree: usize,
    beam_width: NonZeroUsize,
    alpha: f32,
    store: &'a V,
    scorer: S,

    graph: MutableGraph,
    searcher: ThreadLocal<RefCell<GraphSearcher>>,
    in_flight: SkipSet<u32>,
}

impl<'a, V, S> GraphBuilder<'a, V, S>
where
    V: VectorStore + Sync + Send,
    S: Scorer<Vector = V::Vector> + Sync + Send,
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
        max_degree: NonZeroUsize,
        beam_width: NonZeroUsize,
        alpha: f32,
        store: &'a V,
        scorer: S,
    ) -> Self {
        u32::try_from(store.len()).expect("ordinal count limited to 32 bits");
        Self {
            max_degree: max_degree.into(),
            beam_width,
            alpha,
            store,
            scorer,
            graph: MutableGraph::new(store.len(), max_degree),
            searcher: ThreadLocal::with_capacity(current_num_threads()),
            in_flight: SkipSet::new(),
        }
    }

    /// Add all vectors from the backing store.
    fn add_all_vectors(&self) {
        (0..self.store.len()).into_par_iter().for_each(|i| {
            self.add_node(i);
        });
    }

    /// Add a single vector from the backing store.
    ///
    /// *Panics* if `node` is out of range.
    pub fn add_node(&self, node: usize) {
        assert!(node < self.store.len());
        if self.graph.try_set_entry_point(node) {
            return;
        }

        let mut searcher = self
            .searcher
            .get_or(|| RefCell::new(GraphSearcher::new(self.beam_width)))
            .borrow_mut();
        searcher.clear();
        // Update the set of in-flight nodes to include this one. All in-flight nodes will be added
        // to the visited set to ensure we generate links.
        let query = self.store.get(node);
        for n in self.in_flight.iter() {
            searcher.visited.push(Neighbor::from((
                *n as u32,
                self.scorer.score(query, self.store.get(*n as usize)),
            )))
        }
        self.in_flight.insert(node as u32);
        searcher.search_for_insertion(&self.graph, self.store, query, &self.scorer);
        searcher.visited.dedup();
        searcher.visited.retain(|n| n.id as usize != node);
        let neighbors = self.robust_prune(&searcher.visited);
        self.graph
            .insert_neighbors(node, &neighbors, |unpruned| self.robust_prune(unpruned));
        for n in neighbors {
            self.graph.insert_neighbor(
                n.id as usize,
                Neighbor::from((node as u32, n.score)),
                |unpruned| self.robust_prune(unpruned),
            );
        }

        self.in_flight.remove(&(node as u32));
    }

    /// Wraps up graph building and prepares for search.
    pub fn finish(self) -> MutableGraph {
        (0..self.store.len()).into_par_iter().for_each(|i| {
            self.graph
                .maybe_prune_neighbors(i, |unpruned| self.robust_prune(unpruned))
        });
        self.graph
    }

    /// REQUIRES: neighbors is sorted in descending order by score.
    fn robust_prune(&self, neighbors: &[Neighbor]) -> Vec<Neighbor> {
        let mut prune_factor =
            vec![NotNan::new(0.0f32).expect("not NaN constant"); neighbors.len()];
        let mut pruned = Vec::with_capacity(self.max_degree);
        let mut cur_alpha = NotNan::new(1.0f32).expect("not NaN constant");
        let max_alpha = NotNan::new(self.alpha).expect("not NaN constant");
        while cur_alpha <= max_alpha && pruned.len() < self.max_degree {
            for (i, n) in neighbors.iter().enumerate() {
                if pruned.len() >= self.max_degree {
                    break;
                }
                if prune_factor[i] > max_alpha {
                    continue;
                }
                prune_factor[i] = NotNan::new(f32::MAX).unwrap();
                pruned.push(*n);
                // Update prune factor for all subsequent neighbors.
                let n_vector = self.store.get(n.id as usize);
                for (j, o) in neighbors.iter().enumerate().skip(i + 1) {
                    if prune_factor[j] > max_alpha {
                        continue;
                    }
                    // This inverts the order from the paper because score metrics favor larger
                    // values. This might still need tuning.
                    let score_n_o = self.scorer.score(n_vector, self.store.get(o.id as usize));
                    prune_factor[j] = prune_factor[j].max(score_n_o / o.score)
                }
            }
            cur_alpha *= 1.2;
        }
        pruned
    }
}

#[derive(Debug)]
pub struct MutableGraph {
    max_degree: usize,
    entry_point: AtomicUsize,
    // XXX Vec<Neighbor> ought to be something else that applies an ordering constraint, probably
    // just default neighbor ordering.
    // XXX have all updates replace the values rather than updates in place and use crossbeam::epoch.
    nodes: Vec<RwLock<Vec<Neighbor>>>,
}

impl MutableGraph {
    fn new(len: usize, max_degree: NonZeroUsize) -> Self {
        let mut nodes = Vec::with_capacity(len);
        nodes.resize_with(len, || {
            RwLock::new(Vec::with_capacity(max_degree.get() * 2))
        });
        Self {
            max_degree: max_degree.get(),
            entry_point: AtomicUsize::new(usize::MAX),
            nodes,
        }
    }

    fn try_set_entry_point(&self, node: usize) -> bool {
        self.entry_point
            .compare_exchange(usize::MAX, node, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
    }

    fn insert_neighbor<P>(&self, node: usize, neighbor: Neighbor, prune: P)
    where
        P: Fn(&[Neighbor]) -> Vec<Neighbor>,
    {
        let mut edges = self.nodes[node].write().unwrap();
        assert_ne!(node, neighbor.id as usize);
        if let Err(index) = edges.binary_search(&neighbor) {
            edges.insert(index, neighbor);
            if edges.len() >= edges.capacity() {
                let pruned = prune(&edges);
                edges.clear();
                edges.extend_from_slice(&pruned);
            }
        }
    }

    fn insert_neighbors<P>(&self, node: usize, neighbors: &[Neighbor], prune: P)
    where
        P: Fn(&[Neighbor]) -> Vec<Neighbor>,
    {
        let mut edges = self.nodes[node].write().unwrap();
        let cur = edges.clone();
        edges.clear();
        for n in itertools::kmerge(vec![&cur, neighbors].into_iter()) {
            edges.push(*n);
        }
        if edges.len() >= edges.capacity() {
            let pruned = prune(&edges);
            edges.clear();
            edges.extend_from_slice(&pruned);
        }
    }

    fn maybe_prune_neighbors<P>(&self, node: usize, prune: P)
    where
        P: Fn(&[Neighbor]) -> Vec<Neighbor>,
    {
        let mut edges = self.nodes[node].write().unwrap();
        if edges.len() >= self.max_degree {
            let pruned = prune(&edges);
            edges.clear();
            edges.extend_from_slice(&pruned);
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
    guard: RwLockReadGuard<'a, Vec<Neighbor>>,
    next: usize,
}

impl<'a> NeighborNodeIterator<'a> {
    fn new(neighbors: &'a RwLock<Vec<Neighbor>>) -> Self {
        Self {
            guard: neighbors.read().unwrap(),
            next: 0,
        }
    }
}

impl<'a> Iterator for NeighborNodeIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.guard.len() {
            return None;
        }

        let i = self.next;
        self.next += 1;
        Some(self.guard[i].id as usize)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = if self.next < self.guard.len() {
            self.guard.len() - self.next
        } else {
            0
        };
        (size, Some(size))
    }
}

impl<'a> ExactSizeIterator for NeighborNodeIterator<'a> {}

impl<'a> FusedIterator for NeighborNodeIterator<'a> {}

impl Graph for MutableGraph {
    type NeighborEdgeIterator<'c> = NeighborNodeIterator<'c> where Self: 'c;

    fn entry_point(&self) -> Option<usize> {
        let entry_point = self.entry_point.load(Ordering::Relaxed);
        if entry_point != usize::MAX {
            Some(entry_point)
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

pub struct ImmutableMemoryGraph<'a> {
    entry_point: Option<u32>,
    node_extents: &'a [u64],
    edges: &'a [u32],
}

impl<'a> ImmutableMemoryGraph<'a> {
    pub fn new(rep: &'a [u8]) -> Result<Self, &'static str> {
        if rep.len() < 8 {
            return Err("rep too short to contain a valid graph");
        }
        let (num_nodes, rep) = Self::decode_u32(rep);
        if num_nodes == 0 {
            return Ok(Self {
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

impl<'a, 'b> ImmutableEdgeIterator<'a> {
    fn new(graph: &'b ImmutableMemoryGraph<'a>, node: usize) -> Self {
        let begin = graph.node_extents[node] as usize;
        let end = graph.node_extents[node + 1] as usize;
        Self {
            it: graph.edges[begin..end].iter(),
        }
    }
}

impl<'a> Iterator for ImmutableEdgeIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|e| *e as usize)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a> ExactSizeIterator for ImmutableEdgeIterator<'a> {}

impl<'a> FusedIterator for ImmutableEdgeIterator<'a> {}

impl<'a> Graph for ImmutableMemoryGraph<'a> {
    type NeighborEdgeIterator<'c> = ImmutableEdgeIterator<'c> where Self: 'c;

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
    use std::num::NonZeroUsize;

    use ordered_float::NotNan;

    use crate::vamana::Graph;

    use super::{
        GraphBuilder, GraphSearcher, ImmutableMemoryGraph, MutableGraph, Scorer, VectorStore,
    };

    impl VectorStore for Vec<u64> {
        type Vector = u64;

        fn get(&self, i: usize) -> &Self::Vector {
            &self[i]
        }

        fn len(&self) -> usize {
            self.len()
        }
    }

    struct Hamming64;

    impl Scorer for Hamming64 {
        type Vector = u64;

        fn score(&self, a: &Self::Vector, b: &Self::Vector) -> ordered_float::NotNan<f32> {
            let distance = (a ^ b).count_ones();
            let score = ((u64::BITS - distance) as f32) / u64::BITS as f32;
            NotNan::new(score).expect("constant")
        }
    }

    fn make_builder<'a, V: VectorStore<Vector = u64> + Sync + Send>(
        store: &'a V,
    ) -> GraphBuilder<'a, V, Hamming64> {
        GraphBuilder::new(
            NonZeroUsize::new(4).expect("constant"),
            NonZeroUsize::new(10).expect("constant"),
            1.2f32,
            store,
            Hamming64,
        )
    }

    fn get_neighbors(graph: &impl Graph, node: usize) -> Vec<usize> {
        let mut edges: Vec<usize> = graph.neighbors_iter(node).collect();
        edges.sort();
        edges
    }

    #[test]
    fn empty_store_and_graph() {
        let store: Vec<u64> = vec![];
        let builder = make_builder(&store);
        builder.add_all_vectors();
        let graph: MutableGraph = builder.finish();
        assert_eq!(graph.len(), 0);
        assert_eq!(graph.entry_point(), None);
    }

    #[test]
    fn empty_graph() {
        let store: Vec<u64> = vec![0, 1, 2, 3];
        let builder = make_builder(&store);
        let graph: MutableGraph = builder.finish();
        assert_eq!(graph.len(), 4);
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
        let builder = make_builder(&store);
        builder.add_all_vectors();
        let graph: MutableGraph = builder.finish();
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
        let builder = make_builder(&store);
        builder.add_all_vectors();
        let graph: MutableGraph = builder.finish();
        assert_eq!(get_neighbors(&graph, 0), vec![1, 2, 4, 8]);
        assert_eq!(get_neighbors(&graph, 7), vec![3, 5, 6, 15]);
        assert_eq!(get_neighbors(&graph, 8), vec![0, 9, 10, 12]);
        assert_eq!(get_neighbors(&graph, 15), vec![7, 11, 13, 14]);
    }

    #[test]
    fn search_graph() {
        let store: Vec<u64> = (0u64..16u64).collect();
        let builder = make_builder(&store);
        builder.add_all_vectors();
        let graph: MutableGraph = builder.finish();
        let mut searcher = GraphSearcher::new(NonZeroUsize::new(8).expect("constant"));
        let results = searcher.search(&graph, &store, &64, &Hamming64);
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
        let builder = make_builder(&store);
        builder.add_all_vectors();
        let mutable_graph: MutableGraph = builder.finish();
        let mut serialized_graph = Vec::new();
        mutable_graph.write(&mut serialized_graph).unwrap();
        let immutable_graph = ImmutableMemoryGraph::new(&serialized_graph).unwrap();

        assert_eq!(get_neighbors(&immutable_graph, 0), vec![1, 2, 4, 8]);
        assert_eq!(get_neighbors(&immutable_graph, 7), vec![3, 5, 6, 15]);
        assert_eq!(get_neighbors(&immutable_graph, 8), vec![0, 9, 10, 12]);
        assert_eq!(get_neighbors(&immutable_graph, 15), vec![7, 11, 13, 14]);
        let mut searcher = GraphSearcher::new(NonZeroUsize::new(8).expect("constant"));
        let results = searcher.search(&immutable_graph, &store, &64, &Hamming64);
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
