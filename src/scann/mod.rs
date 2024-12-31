use std::{num::NonZero, ops::Range};

use rayon::prelude::*;
use simsimd::SpatialSimilarity;

use crate::{quantization::SampleIterator, VectorStore};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KMeansParams {
    /// Maximum number of iterations to run. May terminate earlier than this.
    pub max_iters: usize,
    /// Minimum number of samples in each cluster.
    pub min_cluster_size: usize,
    /// If the difference between iterations is within epsilon we may terminate early.
    pub epsilon: f64,
    /// If true, return the mapping from cluster to vectors in the training data.
    pub want_partitions: bool,
}

impl Default for KMeansParams {
    fn default() -> Self {
        Self {
            max_iters: 15,
            min_cluster_size: 2,
            epsilon: 0.001,
            want_partitions: true,
        }
    }
}

pub fn kmeans<V: VectorStore<Vector = [f32]> + Send + Sync>(
    training_data: &V,
    clusters: NonZero<usize>,
    params: &KMeansParams,
) -> (MutableFloatVectorStore, Option<Vec<Vec<usize>>>) {
    let dim = training_data.get(0).len();
    let mut centroids = MutableFloatVectorStore::new(clusters.get(), dim);
    for (i, s) in SampleIterator::new(training_data, clusters.get()).enumerate() {
        centroids.get_mut(i).copy_from_slice(s);
    }
    let mut means = vec![0.0; clusters.get()];
    let mut cluster_sizes = vec![0usize; clusters.get()];
    let mut assignments: Vec<(usize, f64)> = vec![];

    for _ in 0..params.max_iters {
        assignments = (0..training_data.len())
            .into_par_iter()
            .map(|i| {
                let v = training_data.get(i);
                centroids
                    .iter()
                    .enumerate()
                    .map(|(ci, cv)| {
                        (
                            ci,
                            SpatialSimilarity::l2(v, cv).expect("same vector length"),
                        )
                    })
                    .max_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
                    .expect("non-zero clusters")
            })
            .collect();
        let mut new_means = vec![0.0; clusters.get()];
        cluster_sizes.fill(0);
        for (cluster, distance) in assignments.iter() {
            new_means[*cluster] += *distance;
            cluster_sizes[*cluster] += 1;
        }
        for (m, c) in new_means.iter_mut().zip(cluster_sizes.iter_mut()) {
            *m /= *c as f64;
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
        for (cluster, _) in assignments.iter() {
            let centroid = centroids.get_mut(*cluster);
            for (c, v) in centroid.iter_mut().zip(training_data.get(*cluster)) {
                *c += v;
            }
        }
        for (centroid, cluster_size) in centroids.iter_mut().zip(cluster_sizes.iter()) {
            centroid.iter_mut().for_each(|d| *d /= *cluster_size as f32);
        }
        means = new_means;
    }

    let partitions = if params.want_partitions {
        let mut partitions = cluster_sizes
            .into_iter()
            .map(|l| Vec::with_capacity(l))
            .collect::<Vec<_>>();
        for (i, (c, _)) in assignments.into_iter().enumerate() {
            partitions[c].push(i);
        }
        Some(partitions)
    } else {
        None
    };

    (centroids, partitions)
}

pub struct MutableFloatVectorStore {
    dim: usize,
    data: Vec<f32>,
}

impl MutableFloatVectorStore {
    fn new(len: usize, dim: usize) -> Self {
        Self {
            dim,
            data: vec![0.0; len * dim],
        }
    }

    fn get_mut(&mut self, i: usize) -> &mut [f32] {
        let r = self.range(i);
        &mut self.data[r]
    }

    fn fill(&mut self, value: f32) {
        self.data.fill(value);
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut [f32]> {
        self.data.chunks_mut(self.dim)
    }

    fn range(&self, i: usize) -> Range<usize> {
        let start = i * self.dim;
        let end = start + self.dim;
        start..end
    }
}

impl VectorStore for MutableFloatVectorStore {
    type Vector = [f32];

    fn get(&self, i: usize) -> &Self::Vector {
        &self.data[self.range(i)]
    }

    fn len(&self) -> usize {
        self.data.len() / self.dim
    }

    fn mean_vector(&self) -> <Self::Vector as ToOwned>::Owned
    where
        Self::Vector: ToOwned,
    {
        unimplemented!()
    }
}
