use std::{
    num::NonZero,
    ops::{Index, IndexMut, Range},
};

use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use simsimd::SpatialSimilarity;

use crate::VectorStore;

pub struct KMeansTreeParams {
    /// Maximum number of vectors in a single leaf node.
    pub max_leaf_size: usize,
    // XXX need to limit the depth of the tree, especially because the algorithm used to create
    // the tree does not terminate even if the subset is empty.
    // Google recommends a 2-level tree if there are < 10M vectors and a 3-level index for > 100M.
    // Otherwise use more levels to optimize for indexing time and fewer to optimize for recall.
    // https://cloud.google.com/alloydb/docs/ai/tune-indexes?resource=scann#tune-scann-indexes
}

impl Default for KMeansTreeParams {
    fn default() -> Self {
        Self { max_leaf_size: 1 }
    }
}

#[derive(Serialize, Deserialize)]
pub struct KMeansTreeNode {
    centers: MutableFloatVectorStore,
    node_type: NodeType,
}

#[derive(Serialize, Deserialize)]
enum NodeType {
    Parent(Vec<KMeansTreeNode>),
    Leaf(usize),
}

impl KMeansTreeNode {
    pub fn train<V: VectorStore<Vector = [f32]> + Send + Sync>(
        training_data: &V,
        k_per_node: NonZero<usize>,
        tree_params: &KMeansTreeParams,
        kmeans_params: &KMeansParams,
        level: usize,
        next_leaf_id: &mut usize,
    ) -> Self {
        assert!(level < 4);
        if training_data.len() <= tree_params.max_leaf_size {
            let leaf_id = *next_leaf_id;
            *next_leaf_id += 1;
            return KMeansTreeNode {
                centers: MutableFloatVectorStore::from_store(training_data),
                node_type: NodeType::Leaf(leaf_id),
            };
        }

        let (centers, subpartitions) = kmeans(training_data, k_per_node, kmeans_params);
        let children = subpartitions
            .into_iter()
            .map(|p| {
                KMeansTreeNode::train(
                    &training_data.subset_view(p),
                    k_per_node,
                    tree_params,
                    kmeans_params,
                    level + 1,
                    next_leaf_id,
                )
            })
            .collect::<Vec<_>>();
        KMeansTreeNode {
            centers,
            node_type: NodeType::Parent(children),
        }
    }

    // TODO: trivial search without spilling: calculate the distance to all centers and search the
    // closest one. Recurse until you find the closest leaf.
}

/// Parameters for computing the kmeans
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KMeansParams {
    /// Maximum number of iterations to run. May run converge in fewer iterations.
    pub max_iters: usize,
    /// Minimum number of samples in each cluster. If any clustes have fewer than this many samples
    /// the computation will not converge.
    pub min_cluster_size: usize,
    /// If the difference of cluster means between iterations is greater than epsilon the
    /// computation will not converge.
    pub epsilon: f64,
    /// Adjustment when reinitializing centroids for clusters that have too few samples.
    pub perturbation: f32,
}

impl Default for KMeansParams {
    fn default() -> Self {
        Self {
            max_iters: 10,
            min_cluster_size: 1,
            epsilon: 0.000_01,
            perturbation: 0.000_000_1,
        }
    }
}

/// Compute `clusters` centroids over `training_data` using `params` configuration.
///
/// Returns the centroids as well as the set of samples in `training_data` that appear in each
/// cluster.
pub fn kmeans<V: VectorStore<Vector = [f32]> + Send + Sync>(
    training_data: &V,
    clusters: NonZero<usize>,
    params: &KMeansParams,
) -> (MutableFloatVectorStore, Vec<Vec<usize>>) {
    let mut centroids = initialize_centroids(training_data, clusters.get());

    let mut means = vec![0.0; clusters.get()];
    let mut cluster_sizes = vec![0usize; clusters.get()];
    let mut assignments: Vec<(usize, f64)> = vec![];

    for _ in 0..params.max_iters {
        assignments = compute_cluster_assignments(training_data, &centroids);
        let mut new_means = vec![0.0; clusters.get()];
        cluster_sizes.fill(0);
        for (cluster, distance) in assignments.iter() {
            new_means[*cluster] += *distance;
            cluster_sizes[*cluster] += 1;
        }
        for (m, c) in new_means.iter_mut().zip(cluster_sizes.iter_mut()) {
            if *c > 0 {
                *m /= *c as f64;
            }
        }

        // We've converged if none of the centers have moved substantially.
        if means
            .iter()
            .zip(new_means.iter())
            .zip(cluster_sizes.iter())
            .all(|((om, nm), s)| *s >= params.min_cluster_size && (nm - om).abs() <= params.epsilon)
        {
            break;
        }

        // Recompute centroids. Start by summing input vectors for each cluster and dividing by count.
        centroids.fill(0.0);
        for (i, (cluster, _)) in assignments.iter().enumerate() {
            for (c, v) in centroids
                .get_mut(*cluster)
                .iter_mut()
                .zip(&training_data[i])
            {
                *c += v;
            }
        }
        let min_cluster_size = std::cmp::min(
            params.min_cluster_size,
            training_data.len() / clusters.get(),
        );
        for (cluster, cluster_size) in cluster_sizes.iter().enumerate() {
            if *cluster_size >= min_cluster_size {
                for d in centroids[cluster].iter_mut() {
                    *d /= *cluster_size as f32;
                }
            } else {
                new_means[cluster] = -1.0;
                let (sample_index, sample_cluster) = loop {
                    let i = thread_rng().gen_range(0..training_data.len());
                    let cluster = assignments[i].0;
                    if cluster_sizes[cluster] >= params.min_cluster_size {
                        break (i, cluster);
                    }
                };

                let sample_point = &training_data[sample_index];
                let sample_centroid = &centroids[sample_cluster];
                let new_centroid: Vec<f32> = sample_centroid
                    .iter()
                    .zip(sample_point.iter())
                    .map(|(c, s)| *c + params.perturbation * (*s - *c))
                    .collect();
                centroids[cluster].copy_from_slice(&new_centroid);
            }
        }
        means = new_means;
    }

    let mut partitions = cluster_sizes
        .into_iter()
        .map(Vec::with_capacity)
        .collect::<Vec<_>>();
    for (i, (c, _)) in assignments.into_iter().enumerate() {
        partitions[c].push(i);
    }

    (centroids, partitions)
}

/// Create `clusters` initial centroids from `training_data` by the kmeans++ scheme.
fn initialize_centroids<V: VectorStore<Vector = [f32]> + Send + Sync>(
    training_data: &V,
    clusters: usize,
) -> MutableFloatVectorStore {
    // Use kmeans++ initialization.
    let mut centroids = MutableFloatVectorStore::with_capacity(clusters, training_data[0].len());
    centroids.push(&training_data[thread_rng().gen_range(0..training_data.len())]);
    while centroids.len() < clusters {
        let assignments = compute_cluster_assignments(training_data, &centroids);
        let selected = WeightedIndex::new(assignments.iter().map(|(_, w)| w))
            .unwrap()
            .sample(&mut thread_rng());
        centroids.push(&training_data[selected]);
    }
    centroids
}

/// Compute the `centroid` that each sample in `training_data` is closest to as well as the distance
/// between the sample and the centroid.
fn compute_cluster_assignments<
    V: VectorStore<Vector = [f32]> + Send + Sync,
    C: VectorStore<Vector = [f32]> + Send + Sync,
>(
    training_data: &V,
    centroids: &C,
) -> Vec<(usize, f64)> {
    (0..training_data.len())
        .into_par_iter()
        .map(|i| {
            let v = &training_data[i];
            centroids
                .iter()
                .enumerate()
                .map(|(ci, cv)| {
                    (
                        ci,
                        SpatialSimilarity::l2(v, cv).expect("same vector length"),
                    )
                })
                .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
                .expect("non-zero clusters")
        })
        .collect()
}

/// A mutable `VectorStore` backed by a `Vec`.
#[derive(Serialize, Deserialize, Clone)]
pub struct MutableFloatVectorStore {
    dim: usize,
    data: Vec<f32>,
}

impl MutableFloatVectorStore {
    /// Create a store with `len` vectors of `dim` length.
    ///
    /// All vectors in the store are zero-initialized.
    pub fn new(len: usize, dim: usize) -> Self {
        Self {
            dim,
            data: vec![0.0; len * dim],
        }
    }

    /// Create an empty store pre-allocated for `capacity` vectors of `dim` length.
    pub fn with_capacity(capacity: usize, dim: usize) -> Self {
        Self {
            dim,
            data: Vec::with_capacity(capacity * dim),
        }
    }

    /// Create a mutable copy of another store.
    pub fn from_store<V: VectorStore<Vector = [f32]>>(store: &V) -> Self {
        if store.len() > 0 {
            let mut s = MutableFloatVectorStore::new(store.len(), store[0].len());
            for (i, o) in store.iter().zip(s.iter_mut()) {
                o.copy_from_slice(i);
            }
            s
        } else {
            MutableFloatVectorStore::new(0, 0)
        }
    }

    /// Get the mutable vector at `index`.
    ///
    /// *Panics* if `index >= len()`.`
    pub fn get_mut(&mut self, index: usize) -> &mut [f32] {
        let r = self.range(index);
        &mut self.data[r]
    }

    /// Return an iterator over mutable vectors.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut [f32]> {
        self.data.chunks_mut(self.dim)
    }

    /// Append a new vector to this store.
    ///
    /// *Panics* if `vector.len() != dim()`
    pub fn push(&mut self, vector: &[f32]) {
        assert_eq!(vector.len(), self.dim);
        self.data.extend_from_slice(vector);
    }

    /// Return the number of dimensions in each vector.
    pub fn dim(&self) -> usize {
        self.dim
    }

    fn fill(&mut self, value: f32) {
        self.data.fill(value);
    }

    fn range(&self, index: usize) -> Range<usize> {
        let start = index * self.dim;
        let end = start + self.dim;
        start..end
    }
}

impl VectorStore for MutableFloatVectorStore {
    type Vector = [f32];

    fn len(&self) -> usize {
        self.data.len() / self.dim
    }

    fn iter(&self) -> impl ExactSizeIterator<Item = &Self::Vector> {
        self.data.chunks(self.dim)
    }
}

impl Index<usize> for MutableFloatVectorStore {
    type Output = [f32];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[self.range(index)]
    }
}

impl IndexMut<usize> for MutableFloatVectorStore {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let r = self.range(index);
        &mut self.data[r]
    }
}
