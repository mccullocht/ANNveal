mod quantization;
mod scorer;
mod store;
mod vamana;

use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    fs::File,
    io::{BufWriter, Write},
    num::NonZeroUsize,
    path::PathBuf,
    str::FromStr,
};

use clap::{Args, Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use ordered_float::NotNan;
use quantization::binary_quantize_f32;
use rayon::prelude::*;
use scorer::{EuclideanScorer, VectorScorer};
use store::{SliceFloatVectorStore, SliceU32VectorStore};
use vamana::{GraphSearcher, Neighbor};

use crate::{
    scorer::HammingScorer,
    store::{SliceBitVectorStore, VectorStore},
    vamana::{GraphBuilder, ImmutableMemoryGraph},
};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Binary quantize an input vector file.
    BinaryQuantize {
        /// Path to input flat vector file.
        #[arg(short, long)]
        vectors: PathBuf,
        /// Coding of vectors in input flat vector file.
        #[arg(short, long)]
        coding: VectorCoding,
        /// Output path of binary coded vectors.
        #[arg(short, long)]
        output: PathBuf,
        // Number of dimensions in input vectors.
        #[arg(short, long)]
        dimensions: NonZeroUsize,
    },
    /// Search an input vector file using queries from another file.
    Search(SearchArgs),
    /// Build a Vamana vector index from an input flat vector file.
    VamanaBuild(VamanaBuildArgs),
    /// Search a Vamana vector index stored on disk.
    VamanaSearch(VamanaSearchArgs),
}

#[derive(Args)]
struct SearchArgs {
    /// Path to input query flat vector file.
    #[arg(short, long)]
    queries: PathBuf,
    /// Number of queries to run. If unset, run all input queries.
    #[arg(short, long)]
    num_queries: Option<NonZeroUsize>,
    /// Number of results to retrieve for each query.
    #[arg(short, long)]
    num_candidates: NonZeroUsize,
    /// Path to input flat vector file.
    #[arg(short, long)]
    vectors: PathBuf,
    /// Coding of vectors in input flat vector file.
    #[arg(short, long)]
    coding: VectorCoding,
    /// Path to parallel flat vector file.
    #[arg(short, long)]
    binary_vectors: Option<PathBuf>,
    /// Amount to oversample by when performing binary vector search.
    /// By default no oversampling occurs.
    #[arg(short, long, default_value_t = 1.0f32)]
    oversample: f32,
    // Number of dimensions in input vectors.
    #[arg(short, long)]
    dimensions: NonZeroUsize,
}

#[derive(Args)]
struct VamanaBuildArgs {
    /// Path to the input flat vector file.
    #[arg(short, long)]
    vectors: PathBuf,
    /// Coding of vectors in input flat vector file.
    #[arg(short, long)]
    coding: VectorCoding,
    /// Number of dimensions in input vectors.
    #[arg(short, long)]
    dimensions: NonZeroUsize,
    /// Path to output vamana index file.
    #[arg(short, long)]
    index_file: PathBuf,
    /// Maximum degree of each vertex in the vamana graph.
    #[arg(short, long, default_value_t = NonZeroUsize::new(32).unwrap())]
    max_degree: NonZeroUsize,
    /// Beam width for search during index build.
    #[arg(short, long, default_value_t = NonZeroUsize::new(200).unwrap())]
    beam_width: NonZeroUsize,
    /// Alpha value for pruning; must be >= 1.0
    #[arg(short, long, default_value_t = NotNan::new(1.2f32).unwrap())]
    alpha: NotNan<f32>,
}

// XXX need query vectors and printing support.
#[derive(Args)]
struct VamanaSearchArgs {
    /// Path to the serialized vamana graph.
    #[arg(short, long)]
    graph: PathBuf,
    /// Path to the binary vectors.
    #[arg(short, long)]
    bin_vectors: PathBuf,
    /// Path to the float32 vectors. If this is set we will re-rank the results.
    #[arg(short, long)]
    f32_vectors: Option<PathBuf>,
    /// Number of dimensions in input vectors.
    #[arg(short, long)]
    dimensions: NonZeroUsize,
    /// Beam width for search during index build.
    #[arg(short, long, default_value_t = NonZeroUsize::new(100).unwrap())]
    num_candidates: NonZeroUsize,

    /// Path to input query flat vector file.
    #[arg(short, long)]
    queries: PathBuf,
    /// Number of queries to run. If unset, run all input queries.
    #[arg(short, long)]
    num_queries: Option<NonZeroUsize>,
    /// List of 100 neighbors for each input query, as an array of `u32`s.
    #[arg(short, long)]
    neighbors: Option<PathBuf>,
    /// Compute recall in top K results. Requires that --neighbors is also set.
    #[arg(short, long)]
    recall_k: Option<NonZeroUsize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VectorCoding {
    F32,
    Bin,
}

impl FromStr for VectorCoding {
    type Err = &'static str;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "f32" => Ok(VectorCoding::F32),
            "bin" => Ok(VectorCoding::Bin),
            _ => Err("unknown vector coding format"),
        }
    }
}

fn binary_quantize(
    vectors: PathBuf,
    output: PathBuf,
    dimensions: NonZeroUsize,
) -> std::io::Result<()> {
    let float_vector_store = SliceFloatVectorStore::new(
        unsafe { memmap2::Mmap::map(&File::open(vectors)?)? },
        dimensions,
    );
    let mut vectors_out = BufWriter::new(File::create(output)?);
    for fv in float_vector_store.iter() {
        for b in binary_quantize_f32(fv) {
            vectors_out.write_all(std::slice::from_ref(&b))?;
        }
    }
    Ok(())
}

fn search(args: SearchArgs) -> std::io::Result<()> {
    let query_store = SliceFloatVectorStore::new(
        unsafe { memmap2::Mmap::map(&File::open(args.queries)?)? },
        args.dimensions,
    );
    let limit = args
        .num_queries
        .map(NonZeroUsize::get)
        .unwrap_or(query_store.len());

    let vector_store = SliceFloatVectorStore::new(
        unsafe { memmap2::Mmap::map(&File::open(args.vectors)?)? },
        args.dimensions,
    );
    let bin_vector_store = match args.binary_vectors {
        Some(p) => Some(SliceBitVectorStore::new(
            unsafe { memmap2::Mmap::map(&File::open(p)?)? },
            args.dimensions,
        )),
        None => None,
    };
    search_queries(
        &query_store,
        args.num_candidates,
        limit,
        &vector_store,
        bin_vector_store.as_ref(),
        args.oversample,
    );
    Ok(())
}

/// Retrieve num_candidates for query in store and return the result in any order.
fn retrieve<V, B, S>(
    query: &V,
    store: &B,
    scorer: &S,
    num_candidates: usize,
) -> Vec<Reverse<(NotNan<f32>, usize)>>
where
    V: ?Sized,
    B: VectorStore<Vector = V>,
    S: VectorScorer<Vector = V>,
{
    let mut heap: BinaryHeap<Reverse<(NotNan<f32>, usize)>> =
        BinaryHeap::with_capacity(num_candidates + 1);
    for (i, v) in store.iter().enumerate() {
        let score = scorer.score(query, v);
        if heap.len() >= num_candidates {
            if score > heap.peek().unwrap().0 .0 {
                heap.pop();
            } else {
                continue;
            }
        }
        heap.push(Reverse((score, i)));
    }
    heap.into_vec()
}

fn full_retrieve(
    query: &[f32],
    store: &impl VectorStore<Vector = [f32]>,
    num_candidates: usize,
) -> Vec<Reverse<(NotNan<f32>, usize)>> {
    let mut results = retrieve(query, store, &EuclideanScorer, num_candidates);
    results.sort();
    results
}

fn binary_retrieve(
    query: &[f32],
    store: &impl VectorStore<Vector = [f32]>,
    bin_store: &impl VectorStore<Vector = [u8]>,
    num_candidates: usize,
    oversample: f32,
) -> Vec<Reverse<(NotNan<f32>, usize)>> {
    let binary_query: Vec<u8> = binary_quantize_f32(query).collect();
    let results = retrieve(
        binary_query.as_ref(),
        bin_store,
        &HammingScorer,
        (num_candidates as f32 * oversample) as usize,
    );
    let mut full_results: Vec<Reverse<(NotNan<f32>, usize)>> = results
        .into_iter()
        .map(|e| Reverse((EuclideanScorer.score(query, store.get(e.0 .1)), e.0 .1)))
        .collect();
    full_results.sort();
    full_results.truncate(num_candidates);
    full_results
}

fn search_queries(
    query_store: &impl VectorStore<Vector = [f32]>,
    num_candidates: NonZeroUsize,
    limit: usize,
    vector_store: &impl VectorStore<Vector = [f32]>,
    bin_vector_store: Option<&impl VectorStore<Vector = [u8]>>,
    oversample: f32,
) {
    match bin_vector_store {
        Some(bin_store) => {
            for q in query_store.iter().take(limit) {
                assert_ne!(
                    binary_retrieve(q, vector_store, bin_store, num_candidates.get(), oversample)
                        .len(),
                    0
                );
            }
        }
        None => {
            for q in query_store.iter().take(limit) {
                assert_ne!(
                    full_retrieve(q, vector_store, num_candidates.get(),).len(),
                    0
                );
            }
        }
    }
}

fn vamana_build(args: VamanaBuildArgs) -> std::io::Result<()> {
    assert_eq!(args.coding, VectorCoding::Bin);
    let store = SliceBitVectorStore::new(
        unsafe { memmap2::Mmap::map(&File::open(args.vectors)?)? },
        args.dimensions,
    );
    let builder = GraphBuilder::new(
        args.max_degree,
        args.beam_width,
        *args.alpha,
        &store,
        HammingScorer,
    );
    let progress = ProgressBar::new(store.len() as u64)
        .with_style(
            ProgressStyle::default_bar()
                .template("{wide_bar} {pos}/{len} ETA: {eta_precise} Elapsed: {elapsed_precise}")
                .unwrap(),
        )
        .with_finish(indicatif::ProgressFinish::AndLeave);
    (0..store.len()).into_par_iter().for_each(|i| {
        builder.add_node(i);
        progress.inc(1);
    });
    progress.finish_using_style();
    let graph = builder.finish();
    let mut w = BufWriter::new(File::create(args.index_file)?);
    graph.write(&mut w)?;
    Ok(())
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct RerankedResult {
    result: Neighbor,
    rank: u32,
    approx_score: NotNan<f32>,
}

impl Ord for RerankedResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.result.cmp(&other.result)
    }
}

impl PartialOrd for RerankedResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

struct RecallState {
    neighbors: SliceU32VectorStore<memmap2::Mmap>,
    k: usize,
}

fn vamana_search(args: VamanaSearchArgs) -> std::io::Result<()> {
    let graph_backing = unsafe { memmap2::Mmap::map(&File::open(args.graph)?)? };
    let graph = ImmutableMemoryGraph::new(&graph_backing).unwrap();
    let bin_vectors = SliceBitVectorStore::new(
        unsafe { memmap2::Mmap::map(&File::open(args.bin_vectors)?)? },
        args.dimensions,
    );
    let f32_vectors = match args.f32_vectors {
        Some(vectors) => Some(SliceFloatVectorStore::new(
            unsafe { memmap2::Mmap::map(&File::open(vectors)?)? },
            args.dimensions,
        )),
        None => None,
    };
    let queries = SliceFloatVectorStore::new(
        unsafe { memmap2::Mmap::map(&File::open(args.queries)?)? },
        args.dimensions,
    );
    let limit = std::cmp::min(
        args.num_queries
            .map(NonZeroUsize::get)
            .unwrap_or(queries.len()),
        queries.len(),
    );

    let recall_state = match (args.neighbors, args.recall_k) {
        (Some(n), Some(k)) => Some(RecallState {
            neighbors: SliceU32VectorStore::new(
                unsafe { memmap2::Mmap::map(&File::open(n)?)? },
                NonZeroUsize::new(100).unwrap(),
            ),
            k: k.get(),
        }),
        (Some(_), None) => panic!("--neighbors set without --recall_k"),
        (None, Some(_)) => panic!("--recall_k set without --neighbors"),
        (None, None) => None,
    };

    let progress = ProgressBar::new(limit as u64)
        .with_style(
            ProgressStyle::default_bar()
                .template("{wide_bar} {pos}/{len} ETA: {eta_precise} Elapsed: {elapsed_precise}")
                .unwrap(),
        )
        .with_finish(indicatif::ProgressFinish::AndLeave);
    let mut searcher = GraphSearcher::new(args.num_candidates);
    let mut rerank_stats: Option<(usize, usize)> = None;
    let mut recall_stats: Option<(usize, usize)> = None;
    for (i, query) in queries.iter().enumerate().take(limit) {
        let bin_query: Vec<u8> = binary_quantize_f32(query).collect();
        let mut results = searcher.search(&graph, &bin_vectors, &bin_query, &HammingScorer);
        assert_ne!(results.len(), 0);

        if let Some(store) = f32_vectors.as_ref() {
            let mut reranked = results
                .par_iter()
                .enumerate()
                .map(|(i, n)| RerankedResult {
                    result: Neighbor {
                        id: n.id,
                        score: EuclideanScorer.score(query, store.get(n.id as usize)),
                    },
                    rank: i as u32,
                    approx_score: n.score,
                })
                .collect::<Vec<_>>();
            reranked.sort();
            let (ref mut rank_diff, ref mut count) = rerank_stats.get_or_insert((0, 0));
            *count += reranked.len();
            for (i, r) in reranked.iter().enumerate() {
                *rank_diff += i.abs_diff(r.rank as usize);
            }
            results = reranked.into_iter().map(|e| e.result).collect();
        };

        if let Some(recall_state) = recall_state.as_ref() {
            let (ref mut matched, ref mut total) = recall_stats.get_or_insert((0, 0));
            let mut expected: Vec<u32> = recall_state
                .neighbors
                .get(i)
                .iter()
                .take(recall_state.k)
                .copied()
                .collect();
            expected.sort();
            for n in results.iter().take(recall_state.k) {
                *total += 1;
                if expected.binary_search(&n.id).is_ok() {
                    *matched += 1;
                }
            }
        }

        searcher.clear();
        progress.inc(1);
    }

    progress.finish_using_style();
    println!(
        "queries {} avg duration {:.3} ms",
        limit,
        progress.elapsed().div_f32(limit as f32).as_micros() as f64 / 1_000f64
    );
    if let Some((rerank_diff, count)) = rerank_stats {
        println!(
            "result_count {} rerank_diff {} avg {:.2}",
            count,
            rerank_diff,
            rerank_diff as f64 / count as f64
        );
    }
    if let Some((matched, total)) = recall_stats {
        println!("recall {:.5}", matched as f64 / total as f64);
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    let args = Cli::parse();
    match args.command {
        Commands::BinaryQuantize {
            vectors,
            coding: _,
            output,
            dimensions,
        } => binary_quantize(vectors, output, dimensions),
        Commands::Search(args) => search(args),
        Commands::VamanaBuild(args) => vamana_build(args),
        Commands::VamanaSearch(args) => vamana_search(args),
    }
}
