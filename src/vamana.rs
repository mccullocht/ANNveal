// XXX filtered search
// the idea is that we can use multiple (random) entry points to the graph to seed the candidate
// list, where the number of candidates is between 1/f and num_candidates/f. We would apply the
// filter before we traverse any nodes. if the distribution is roughly random across edges we'd
// expect to search the same number of nodes. We would use the same result set size but increase
// the size of the candidates heap proportionally to avoid dropping stuff.

use core::num;
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
    io::Write,
    iter::FusedIterator,
    num::NonZeroUsize,
};

use ordered_float::NotNan;

/// Dense store of vector data, analogous to a `Vec``.
pub trait VectorStore {
    /// Type of the underlying vector data.
    type Vector;
    type VectorIter<'v>: Iterator<Item = &'v Self::Vector>
    where
        Self: 'v;

    /// Obtain a reference to the contents of the vector by VectorId.
    /// *Panics* if `i`` is out of bounds.
    fn get(&self, i: usize) -> &Self::Vector;

    /// Obtain an iterator over vector data.
    fn vector_iter(&self) -> Self::VectorIter<'_>;

    /// Compute the mean point from the data set.
    fn compute_mean(&self) -> Self::Vector;

    /// Uses `scorer` to find the point nearest to `point` or `None` if the
    /// store is empty.
    fn find_nearest_point<S>(&self, point: &Self::Vector, scorer: &S) -> Option<usize>
    where
        S: Scorer<Vector = Self::Vector>;

    /// Return the total number of vectors in the store.
    fn len(&self) -> usize;
}

/// Trait for scoring vectors against one another.
pub trait Scorer {
    /// Type for the underlying vector data.
    type Vector;

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

// XXX should this be public?
pub struct GraphSearcher {
    k: NonZeroUsize,
    candidates: BinaryHeap<Reverse<Neighbor>>,
    results: BinaryHeap<Neighbor>,
    seen: HashSet<usize>,
    visited: Vec<Neighbor>,
}

impl GraphSearcher {
    /// Create a new GraphSearcher that will return k results searching.
    pub fn new(k: NonZeroUsize) -> Self {
        Self {
            k,
            candidates: BinaryHeap::with_capacity(k.into()),
            results: BinaryHeap::with_capacity(k.into()),
            seen: HashSet::new(),
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
    fn search_for_insertion<G, V, Q, S>(&mut self, graph: &G, store: &V, query: &Q, scorer: &S)
    where
        G: Graph,
        V: VectorStore<Vector = Q>,
        S: Scorer<Vector = Q>,
    {
        self.search_internal(graph, store, query, scorer, true);
        self.visited.sort();
    }

    fn search_internal<G, V, Q, S>(
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
    alpha: f32,
    store: &'a V,
    scorer: S,

    graph: MutableGraph,
    searcher: GraphSearcher,
}

impl<'a, V, S> GraphBuilder<'a, V, S>
where
    V: VectorStore,
    S: Scorer<Vector = V::Vector>,
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
            alpha,
            store,
            scorer,
            graph: MutableGraph::new(store.len(), max_degree),
            searcher: GraphSearcher::new(beam_width),
        }
    }

    /// Add all vectors from the backing store.
    pub fn add_all_vectors(&mut self) {
        // Choose a good entry point and add it first.
        if let Some(mediod) = self
            .store
            .find_nearest_point(&self.store.compute_mean(), &self.scorer)
        {
            self.add_node(mediod);
        }
        for i in 0..self.store.len() {
            self.add_node(i);
        }
    }

    /// Add a single vector from the backing store.
    ///
    /// *Panics* if `node` is out of range.
    pub fn add_node(&mut self, node: usize) {
        assert!(node < self.store.len());
        if self.graph.entry_point.is_none() {
            self.graph.entry_point = Some(node as u32);
            return;
        }

        if !self.graph.get_neighbors(node).is_empty() {
            return;
        }

        self.searcher.clear();
        self.searcher.search_for_insertion(
            &self.graph,
            self.store,
            self.store.get(node),
            &self.scorer,
        );
        let neighbors = self.robust_prune(&self.searcher.visited);
        self.graph.set_neighbors(node, &neighbors);
        for n in neighbors {
            if self
                .graph
                .add_neighbor(n.id as usize, Neighbor::from((node as u32, n.score)))
            {
                self.graph.sort_neighbors(n.id as usize);
                let pruned = self.robust_prune(self.graph.get_neighbors(n.id as usize));
                self.graph.set_neighbors(n.id as usize, &pruned);
            }
        }
    }

    /// Wraps up graph building and prepares for search.
    pub fn finish(mut self) -> MutableGraph {
        self.prune_all();
        self.graph
    }

    fn prune_all(&mut self) {
        for id in 0..self.graph.len() {
            let neighbors = self.graph.get_neighbors(id);
            if neighbors.len() > self.max_degree {
                self.graph.sort_neighbors(id);
                let pruned = self.robust_prune(self.graph.get_neighbors(id));
                self.graph.set_neighbors(id, &pruned);
            }
        }
    }

    /// REQUIRES: neighbors is sorted in descending order by score.
    fn robust_prune(&self, neighbors: &[Neighbor]) -> Vec<Neighbor> {
        let mut prune_factor =
            vec![NotNan::new(0.0f32).expect("not NaN constant"); neighbors.len()];
        let mut pruned = Vec::with_capacity(self.max_degree);
        let mut cur_alpha = NotNan::new(1.0f32).expect("not NaN constant");
        let max_alpha = NotNan::new(self.alpha).expect("not NaN constant");
        // XXX may want a list of alpha values intead of one and a multiplier.
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
                for (j, o) in neighbors.iter().enumerate().skip(i + 1) {
                    if prune_factor[j] > max_alpha {
                        continue;
                    }
                    // This inverts the order from the paper because score metrics favor larger
                    // values. This might still need tuning.
                    let score_n_o = self
                        .scorer
                        .score(self.store.get(n.id as usize), self.store.get(o.id as usize));
                    prune_factor[j] = prune_factor[j].max(score_n_o / o.score)
                }
            }
            cur_alpha *= 1.2;
        }
        pruned
    }
}

#[derive(Clone, Debug)]
pub struct MutableGraph {
    entry_point: Option<u32>,
    nodes: Vec<Vec<Neighbor>>,
}

impl MutableGraph {
    fn new(len: usize, max_degree: NonZeroUsize) -> Self {
        Self {
            entry_point: None,
            nodes: vec![Vec::with_capacity(max_degree.get() * 2); len],
        }
    }

    fn get_neighbors(&self, node: usize) -> &[Neighbor] {
        &self.nodes[node]
    }

    fn sort_neighbors(&mut self, node: usize) {
        self.nodes[node].sort();
    }

    fn add_neighbor(&mut self, node: usize, neighbor: Neighbor) -> bool {
        let node = &mut self.nodes[node];
        node.push(neighbor);
        node.len() == node.capacity()
    }

    fn set_neighbors(&mut self, node: usize, neighbors: &[Neighbor]) {
        let node = &mut self.nodes[node];
        node.clear();
        node.extend_from_slice(neighbors)
    }

    pub fn write(&self, out: &mut impl Write) -> std::io::Result<()> {
        out.write_all(&(self.nodes.len() as u32).to_le_bytes())?;
        out.write_all(&(self.entry_point.unwrap_or(0) as u32).to_le_bytes())?;
        // Write the number of nodes at the start of each edge and at the end. This way we will
        // write self.nodes.len() + 1 values for edge extents.
        out.write_all(&0u64.to_le_bytes())?;
        // This is ~8MB per ~1M nodes
        let mut num_edges = 0u64;
        for n in self.nodes.iter() {
            num_edges += n.len() as u64;
            out.write_all(&num_edges.to_le_bytes())?;
        }
        // This is ~120MB per ~1M nodes. If we did simple bit-width compression it would be ~75MB.
        for n in self.nodes.iter() {
            for e in n {
                out.write_all(&e.id.to_le_bytes())?;
            }
        }
        Ok(())
    }
}

pub struct NeighborNodeIterator<'a> {
    it: std::slice::Iter<'a, Neighbor>,
}

impl<'a> NeighborNodeIterator<'a> {
    fn new(neighbors: &'a [Neighbor]) -> Self {
        Self {
            it: neighbors.iter(),
        }
    }
}

impl<'a> Iterator for NeighborNodeIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|x| x.id as usize)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a> ExactSizeIterator for NeighborNodeIterator<'a> {}

impl<'a> FusedIterator for NeighborNodeIterator<'a> {}

impl Graph for MutableGraph {
    type NeighborEdgeIterator<'c> = NeighborNodeIterator<'c> where Self: 'c;

    fn entry_point(&self) -> Option<usize> {
        self.entry_point.map(|p| p as usize)
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
    use std::{cmp::Ordering, num::NonZeroUsize};

    use ordered_float::NotNan;

    use crate::vamana::Graph;

    use super::{
        GraphBuilder, GraphSearcher, ImmutableMemoryGraph, MutableGraph, Neighbor, Scorer,
        VectorStore,
    };

    impl VectorStore for Vec<u64> {
        type Vector = u64;
        type VectorIter<'v> = std::slice::Iter<'v, Self::Vector>;

        fn get(&self, i: usize) -> &Self::Vector {
            &self[i]
        }

        fn vector_iter<'v>(&'v self) -> Self::VectorIter<'v> {
            self.iter()
        }

        fn compute_mean(&self) -> Self::Vector {
            let mut counts = [0usize; 64];
            for mut v in self.iter().copied() {
                while v != 0 {
                    let i = v.trailing_zeros() as usize;
                    counts[i] += 1;
                    v ^= 1u64 << i;
                }
            }

            let mean = counts
                .into_iter()
                .enumerate()
                .filter(|(i, c)| match (*c as usize).cmp(&(self.len() / 2)) {
                    Ordering::Less => false,
                    Ordering::Greater => true,
                    Ordering::Equal => i % 2 == 0,
                })
                .map(|(i, _)| 1u64 << i)
                .fold(0u64, |m, x| m | x);
            mean
        }

        fn find_nearest_point<S>(&self, point: &Self::Vector, scorer: &S) -> Option<usize>
        where
            S: Scorer<Vector = Self::Vector>,
        {
            self.iter()
                .enumerate()
                .map(|(i, v)| Neighbor::from((i as u32, scorer.score(point, v))))
                .min()
                .map(|n| n.id as usize)
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

    fn make_builder<'a, V: VectorStore<Vector = u64>>(
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
        let mut builder = make_builder(&store);
        builder.add_all_vectors();
        builder.prune_all();
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
        let mut builder = make_builder(&store);
        builder.add_node(2);
        let graph: MutableGraph = builder.finish();
        assert_eq!(graph.len(), 4);
        assert_eq!(graph.entry_point(), Some(2));
    }

    #[test]
    fn tiny_graph() {
        let store: Vec<u64> = (0u64..5u64).collect();
        let mut builder = make_builder(&store);
        builder.add_all_vectors();
        let graph: MutableGraph = builder.finish();
        assert_eq!(graph.len(), 5);
        assert_eq!(graph.entry_point(), Some(1));
        assert_eq!(get_neighbors(&graph, 0), vec![1, 2, 3, 4]);
        assert_eq!(get_neighbors(&graph, 1), vec![0, 2, 3, 4]);
        assert_eq!(get_neighbors(&graph, 2), vec![0, 1, 3, 4]);
        assert_eq!(get_neighbors(&graph, 3), vec![0, 1, 2, 4]);
        assert_eq!(get_neighbors(&graph, 4), vec![0, 1, 2, 3]);
    }

    #[test]
    fn pruned_graph() {
        let store: Vec<u64> = (0u64..16u64).collect();
        let mut builder = make_builder(&store);
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
        let mut builder = make_builder(&store);
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
        let mut builder = make_builder(&store);
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
