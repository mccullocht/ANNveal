// XXX filtered search
// the idea is that we can use multiple (random) entry points to the graph to seed the candidate
// list, where the number of candidates is between 1/f and num_candidates/f. We would apply the
// filter before we traverse any nodes. if the distribution is roughly random across edges we'd
// expect to search the same number of nodes. We would use the same result set size but increase
// the size of the candidates heap proportionally to avoid dropping stuff.

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
    num::NonZeroUsize,
};

use ordered_float::NotNan;

/// Ordinal identifier for vectors in the data store and graph.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct VectorId(u32);

/// Ordinal indexed store for vector data.
// XXX I need a way to iterate over vector ids. Range uses an unstabilized trait.
pub trait VectorStore {
    /// Type of the underlying vector data.
    type Vector;

    /// Obtain a reference to the contents of the vector by VectorId.
    /// *Panics* if `ord`` is out of bounds.
    fn get(&self, ord: VectorId) -> &Self::Vector;

    /// Compute the mean point from the data set.
    fn compute_mean(&self) -> Self::Vector;

    /// Uses `scorer` to find the point nearest to `point` or `None` if the
    /// store is empty.
    fn find_nearest_point<S>(&self, point: &Self::Vector, scorer: &S) -> Option<VectorId>
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
    id: VectorId,
    score: NotNan<f32>,
}

impl From<(VectorId, NotNan<f32>)> for Neighbor {
    fn from(value: (VectorId, NotNan<f32>)) -> Self {
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
    type NeighborOrdIterator<'c>: Iterator<Item = VectorId>
    where
        Self: 'c;

    // Returns the entry point to the graph if any nodes have been populated.
    fn entry_point(&self) -> Option<VectorId>;
    /// Return an iterate over the list of nodes neighboring `ord`.
    fn neighbors_iter<'c>(&'c self, ord: VectorId) -> Self::NeighborOrdIterator<'c>;
    /// Return the number of nodes in the graph.
    fn len(&self) -> usize;
}

// XXX should this be public?
pub struct GraphSearcher {
    k: NonZeroUsize,
    candidates: BinaryHeap<Reverse<Neighbor>>,
    results: BinaryHeap<Neighbor>,
    seen: HashSet<VectorId>,
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
        self.visited.sort_by(|a, b| a.cmp(b));
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
        // XXX exit cleanly if the graph is empty.
        let entry_point = graph.entry_point().unwrap();
        let score = scorer.score(query, store.get(entry_point));
        self.seen.insert(entry_point);
        self.candidates.push(Reverse((entry_point, score).into()));

        while let Some(Reverse(candidate)) = self.candidates.pop() {
            if !self.collect(candidate) {
                break;
            }
            if collect_visited {
                self.visited.push(candidate);
            }

            for neighbor_id in graph.neighbors_iter(candidate.id) {
                // skip candidates we've already visited.
                if !self.seen.insert(neighbor_id) {
                    continue;
                }

                let score = scorer.score(query, store.get(neighbor_id));
                let neighbor = Neighbor::from((neighbor_id, score));
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
            self.add_node(VectorId(i as u32));
        }
    }

    /// Add a single vector from the backing store.
    ///
    /// *Panics* if `node` is out of range.
    pub fn add_node(&mut self, node: VectorId) {
        if self.graph.entry_point.is_none() {
            self.graph.entry_point = Some(node);
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
                .add_neighbor(n.id, Neighbor::from((node, n.score)))
            {
                self.graph.sort_neighbors(n.id);
                let pruned = self.robust_prune(self.graph.get_neighbors(n.id));
                self.graph.set_neighbors(n.id, &pruned);
            }
        }
    }

    /// Wraps up graph building and prepares for search.
    pub fn finish(mut self) -> MutableGraph {
        self.prune_all();
        self.graph
    }

    fn prune_all(&mut self) {
        for id in (0..self.graph.len()).map(|i| VectorId(i as u32)) {
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
                        .score(self.store.get(n.id), self.store.get(o.id));
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
    entry_point: Option<VectorId>,
    nodes: Vec<Vec<Neighbor>>,
    edges: usize,
}

impl MutableGraph {
    fn new(len: usize, max_degree: NonZeroUsize) -> Self {
        Self {
            entry_point: None,
            nodes: vec![Vec::with_capacity(max_degree.get() * 2); len],
            edges: 0usize,
        }
    }

    fn get_neighbors(&self, node: VectorId) -> &[Neighbor] {
        &self.nodes[node.0 as usize]
    }

    fn sort_neighbors(&mut self, node: VectorId) {
        self.nodes[node.0 as usize].sort();
    }

    fn add_neighbor(&mut self, node: VectorId, neighbor: Neighbor) -> bool {
        let node = &mut self.nodes[node.0 as usize];
        node.push(neighbor);
        self.edges += 1;
        node.len() == node.capacity()
    }

    fn set_neighbors(&mut self, node: VectorId, neighbors: &[Neighbor]) {
        let node = &mut self.nodes[node.0 as usize];
        self.edges += neighbors.len();
        self.edges -= node.len();
        node.clear();
        node.extend_from_slice(neighbors)
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
    type Item = VectorId;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|x| x.id)
    }
}

impl Graph for MutableGraph {
    type NeighborOrdIterator<'c> = NeighborNodeIterator<'c> where Self: 'c;

    fn entry_point(&self) -> Option<VectorId> {
        self.entry_point
    }

    fn neighbors_iter<'c>(&'c self, ord: VectorId) -> Self::NeighborOrdIterator<'c> {
        NeighborNodeIterator::new(&self.nodes[ord.0 as usize])
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod test {
    use std::{cmp::Ordering, num::NonZeroUsize};

    use ordered_float::NotNan;

    use crate::vamana::Graph;

    use super::{
        GraphBuilder, GraphSearcher, MutableGraph, Neighbor, Scorer, VectorId, VectorStore,
    };

    impl VectorStore for Vec<u64> {
        type Vector = u64;

        fn get(&self, ord: VectorId) -> &Self::Vector {
            &self[ord.0 as usize]
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

        fn find_nearest_point<S>(&self, point: &Self::Vector, scorer: &S) -> Option<VectorId>
        where
            S: Scorer<Vector = Self::Vector>,
        {
            self.iter()
                .enumerate()
                .map(|(i, v)| Neighbor::from((VectorId(i as u32), scorer.score(point, v))))
                .min()
                .map(|n| n.id)
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

    fn get_neighbors(graph: &impl Graph, node: VectorId) -> Vec<u32> {
        let mut edges: Vec<u32> = graph.neighbors_iter(node).map(|n| n.0).collect();
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
        builder.add_node(VectorId(2));
        let graph: MutableGraph = builder.finish();
        assert_eq!(graph.len(), 4);
        assert_eq!(graph.entry_point(), Some(VectorId(2)));
    }

    #[test]
    fn tiny_graph() {
        let store: Vec<u64> = (0u64..5u64).collect();
        let mut builder = make_builder(&store);
        builder.add_all_vectors();
        let graph: MutableGraph = builder.finish();
        assert_eq!(graph.len(), 5);
        assert_eq!(graph.entry_point(), Some(VectorId(1)));
        assert_eq!(get_neighbors(&graph, VectorId(0)), vec![1u32, 2, 3, 4]);
        assert_eq!(get_neighbors(&graph, VectorId(1)), vec![0u32, 2, 3, 4]);
        assert_eq!(get_neighbors(&graph, VectorId(2)), vec![0u32, 1, 3, 4]);
        assert_eq!(get_neighbors(&graph, VectorId(3)), vec![0u32, 1, 2, 4]);
        assert_eq!(get_neighbors(&graph, VectorId(4)), vec![0u32, 1, 2, 3]);
    }

    #[test]
    fn pruned_graph() {
        let store: Vec<u64> = (0u64..16u64).collect();
        let mut builder = make_builder(&store);
        builder.add_all_vectors();
        let graph: MutableGraph = builder.finish();
        assert_eq!(get_neighbors(&graph, VectorId(0)), vec![1u32, 2, 4, 8]);
        assert_eq!(get_neighbors(&graph, VectorId(7)), vec![3u32, 5, 6, 15]);
        assert_eq!(get_neighbors(&graph, VectorId(8)), vec![0u32, 9, 10, 12]);
        assert_eq!(get_neighbors(&graph, VectorId(15)), vec![7u32, 11, 13, 14]);
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
                .map(|n| (n.id.0 as usize, n.score.into_inner()))
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
