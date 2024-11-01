mod graph;
mod quantization;
mod scorer;
mod store;
mod utils;

use std::{
    fs::File,
    io::{BufWriter, Write},
    num::NonZeroUsize,
    ops::Range,
    path::{Path, PathBuf},
};

use clap::{Args, Parser, Subcommand, ValueEnum};
use graph::{
    vamana::{GraphBuilder, GraphSearcher, ImmutableMemoryGraph},
    Neighbor, NeighborSet,
};
use indicatif::{ProgressBar, ProgressStyle};
use ordered_float::NotNan;
use quantization::{QuantizationAlgorithm, Quantizer};
use rayon::prelude::*;
use scorer::{
    DefaultQueryScorer, EuclideanDequantizeScorer, EuclideanScorer, QuantizedEuclideanQueryScorer,
    QuantizedEuclideanScorer, QueryScorer,
};
use store::{SliceFloatVectorStore, SliceQuantizedVectorStore, SliceU32VectorStore, VectorStore};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Quantize an input vector file.
    Quantize(QuantizeArgs),
    /// Build a Vamana vector index from an input flat vector file.
    VamanaBuild(VamanaBuildArgs),
    /// Search a Vamana vector index stored on disk.
    VamanaSearch(VamanaSearchArgs),
}

#[derive(Args)]
struct QuantizeArgs {
    /// Path to input flat vector file.
    #[arg(short, long)]
    vectors: PathBuf,
    /// Output path of binary coded vectors.
    #[arg(short, long)]
    output: PathBuf,
    /// Number of dimensions in input vectors.
    #[arg(short, long)]
    dimensions: NonZeroUsize,
    /// Quantization algorithm to use.
    #[arg(short, long)]
    quantizer: QuantizerAlgorithm,
    /// Number of bits to quantize to.
    ///
    /// When using scalar quantizer, must be in 2..=8.
    #[arg(short, long)]
    bits: Option<usize>,
}

#[derive(Clone, ValueEnum)]
enum QuantizerAlgorithm {
    /// Quantize each dimension into a single bit.
    Binary,
    /// Quantize each dimension into --bits based on mean and standard deviation.
    StatisticalBinary,
    /// Quantize each dimension into --bits linearly based on min and max values.
    Scalar,
}

fn quantize(args: QuantizeArgs) -> std::io::Result<()> {
    let float_vector_store = SliceFloatVectorStore::new(
        unsafe { memmap2::Mmap::map(&File::open(args.vectors)?)? },
        args.dimensions,
    );
    let mut vectors_out = BufWriter::new(File::create(args.output)?);
    let progress = ProgressBar::new(float_vector_store.len() as u64)
        .with_style(
            ProgressStyle::default_bar()
                .template("{wide_bar} {pos}/{len} ETA: {eta_precise} Elapsed: {elapsed_precise}")
                .unwrap(),
        )
        .with_finish(indicatif::ProgressFinish::AndLeave);
    let algo = match args.quantizer {
        QuantizerAlgorithm::Binary => QuantizationAlgorithm::Binary,
        QuantizerAlgorithm::StatisticalBinary => match args.bits {
            Some(1) => QuantizationAlgorithm::BinaryMean,
            Some(b) => QuantizationAlgorithm::StatisticalBinary(b),
            None => panic!("bits must be set"),
        },
        QuantizerAlgorithm::Scalar => QuantizationAlgorithm::Scalar(args.bits.unwrap()),
    };
    let quantizer = Quantizer::from_store(algo, &float_vector_store, args.dimensions.get());
    let mut buf = quantizer.quantization_buffer(args.dimensions.get());
    for v in float_vector_store.iter() {
        quantizer.quantize_to(v, &mut buf);
        vectors_out.write_all(&buf)?;
        progress.inc(1);
    }
    quantizer.write_footer(&mut vectors_out)?;
    progress.finish_using_style();
    Ok(())
}

#[derive(Args)]
struct VamanaBuildArgs {
    /// Path to the input flat vector file.
    #[arg(short, long)]
    vectors: PathBuf,
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
    /// Number of graph shards.
    #[arg(short, long, default_value_t = NonZeroUsize::new(1).unwrap())]
    shards: NonZeroUsize,
}

fn progress_bar(len: usize, message: &'static str) -> ProgressBar {
    let progress = ProgressBar::new(len as u64)
        .with_style(
            ProgressStyle::default_bar()
                .template(
                    "{msg} {wide_bar} {pos}/{len} ETA: {eta_precise} Elapsed: {elapsed_precise}",
                )
                .unwrap(),
        )
        .with_finish(indicatif::ProgressFinish::AndLeave);
    progress.set_message(message);
    progress
}

fn shard_id_range(len: usize, shards: NonZeroUsize) -> Vec<Range<usize>> {
    (0..shards.get())
        .map(|s| {
            let start = len / shards.get() * s;
            let end = if s + 1 == shards.get() {
                len
            } else {
                len / shards.get() * (s + 1)
            };
            start..end
        })
        .collect()
}

fn shard_path(path: &Path, shard: usize, shards: NonZeroUsize) -> PathBuf {
    let mut shard_path = path.to_owned();
    if shards.get() != 1 {
        shard_path.set_file_name(format!(
            "{}-{:05}-of-{:05}",
            shard_path.file_name().unwrap_or_default().to_string_lossy(),
            shard,
            shards.get()
        ));
    }
    shard_path
}

fn vamana_build(args: VamanaBuildArgs) -> std::io::Result<()> {
    let store = SliceQuantizedVectorStore::new(
        unsafe { memmap2::Mmap::map(&File::open(args.vectors)?)? },
        args.dimensions,
    );

    for (shard, range) in shard_id_range(store.len(), args.shards)
        .into_iter()
        .enumerate()
    {
        // TODO: allow building a vamana graph from a float vector store.
        let builder = GraphBuilder::new(
            args.max_degree,
            args.beam_width,
            *args.alpha,
            &store,
            QuantizedEuclideanScorer::new(store.quantizer()),
        );
        let mut progress = progress_bar(range.len(), "build ");
        builder.add_nodes_with_progress(range.clone(), || progress.inc(1));
        progress.finish_using_style();

        progress = progress_bar(store.len(), "finish");
        let graph = builder.finish_with_progress(|| progress.inc(1));
        progress.finish_using_style();

        let mut w = BufWriter::new(File::create(shard_path(
            &args.index_file,
            shard,
            args.shards,
        ))?);
        graph.write(&mut w)?;
    }
    Ok(())
}

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
    /// Number of vectors to rerank using f32 x bit scoring.
    #[arg(long)]
    bit_rerank_budget: Option<usize>,
    /// Number of vectors to rerank with f32 when --f32-vectors is set.
    #[arg(long)]
    f32_rerank_budget: Option<usize>,
    /// Number of graph shards.
    #[arg(short, long, default_value_t = NonZeroUsize::new(1).unwrap())]
    shards: NonZeroUsize,
}

struct Reranker<'a, B> {
    k: usize,
    budget: usize,
    store: &'a B,

    rank_diff: usize,
    count: usize,
}

impl<'a, B> Reranker<'a, B>
where
    B: VectorStore + Send + Sync,
{
    fn new(k: usize, budget: usize, store: &'a B) -> Self {
        Self {
            k,
            budget,
            store,
            rank_diff: 0,
            count: 0,
        }
    }

    fn rerank<Q>(&mut self, results: NeighborSet, query_scorer: &Q) -> NeighborSet
    where
        Q: QueryScorer<Vector = B::Vector> + Send + Sync,
    {
        // XXX this is fluent but does some unnecessary copying.
        // if I record (id, rank) tuples after scoring and sorting then look up the diff from that.
        // we'd still need NeighborSet::from_sorted() which is what I was hoping to avoid.

        let mut reranked = Vec::from(results)
            .into_par_iter()
            .take(self.budget)
            .enumerate()
            .map(|(i, n)| {
                (
                    i,
                    Neighbor::new(n.id, query_scorer.score(self.store.get(n.id as usize))),
                )
            })
            .collect::<Vec<_>>();
        reranked.sort_by_key(|r| r.1);
        NeighborSet::from_sorted(
            reranked
                .into_iter()
                .enumerate()
                .map(|(i, (r, n))| {
                    if i < self.k {
                        self.rank_diff += i.abs_diff(r);
                        self.count += 1;
                    }
                    n
                })
                .collect(),
        )
    }

    fn count(&self) -> usize {
        self.count
    }

    fn rank_diff(&self) -> usize {
        self.rank_diff
    }

    fn avg_rank_diff(&self) -> f64 {
        self.rank_diff as f64 / self.count as f64
    }
}

struct RecallState {
    neighbors: SliceU32VectorStore<memmap2::Mmap>,
    k: usize,
}

fn vamana_search(args: VamanaSearchArgs) -> std::io::Result<()> {
    let mut graphs = Vec::with_capacity(args.shards.get());
    for shard in 0..args.shards.get() {
        graphs.push(
            ImmutableMemoryGraph::new(unsafe {
                memmap2::Mmap::map(&File::open(shard_path(&args.graph, shard, args.shards))?)?
            })
            .unwrap(),
        );
    }
    let qvectors = SliceQuantizedVectorStore::new(
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

    let recall_state = if let Some((p, k)) = args.neighbors.zip(args.recall_k) {
        Some(RecallState {
            neighbors: SliceU32VectorStore::new(
                unsafe { memmap2::Mmap::map(&File::open(p)?)? },
                NonZeroUsize::new(100).unwrap(),
            ),
            k: k.get(),
        })
    } else {
        None
    };

    let progress = ProgressBar::new(limit as u64)
        .with_style(
            ProgressStyle::default_bar()
                .template("{wide_bar} {pos}/{len} ETA: {eta_precise} Elapsed: {elapsed_precise}")
                .unwrap(),
        )
        .with_finish(indicatif::ProgressFinish::AndLeave);
    let mut searcher = GraphSearcher::new(args.num_candidates);
    let mut bit_reranker = args
        .recall_k
        .zip(args.bit_rerank_budget)
        .map(|(k, budget)| Reranker::new(k.get(), budget, &qvectors));
    let mut f32_reranker = args
        .recall_k
        .zip(
            args.f32_rerank_budget
                .or(args.bit_rerank_budget)
                .or(Some(args.num_candidates.get())),
        )
        .zip(f32_vectors.as_ref())
        .map(|((k, budget), store)| Reranker::new(k.get(), budget, store));
    let quantizer = qvectors.quantizer();
    let mut recall_stats: Option<(usize, usize)> = None;
    for (i, query) in queries.iter().enumerate().take(limit) {
        let quantized_query_scorer = QuantizedEuclideanQueryScorer::new(quantizer, query);
        let mut results = searcher.search(&graphs[0], &qvectors, &quantized_query_scorer);
        for graph in graphs.iter().skip(1) {
            searcher.clear();
            let sub_results = searcher.search(graph, &qvectors, &quantized_query_scorer);
            for result in sub_results {
                if let Some(worst) = results.0.last() {
                    if result < *worst {
                        results.0.truncate(args.num_candidates.get() - 1);
                        results.insert(result);
                    } else {
                        break;
                    }
                }
            }
        }
        assert_ne!(results.len(), 0);
        assert!(results.len() <= args.num_candidates.get());

        if let Some(rr) = bit_reranker.as_mut() {
            let query_scorer = EuclideanDequantizeScorer::new(qvectors.quantizer(), query);
            results = rr.rerank(results, &query_scorer);
        }

        if let Some(rr) = f32_reranker.as_mut() {
            let query_scorer = DefaultQueryScorer::new(query, &EuclideanScorer);
            results = rr.rerank(results, &query_scorer);
        }

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
    if let Some(rr) = bit_reranker.as_ref() {
        println!(
            "bit rerank result_count {} rank_diff {} avg {:.2}",
            rr.count(),
            rr.rank_diff(),
            rr.avg_rank_diff(),
        );
    }
    if let Some(rr) = f32_reranker.as_ref() {
        println!(
            "f32 rerank result_count {} rank_diff {} avg {:.2}",
            rr.count(),
            rr.rank_diff(),
            rr.avg_rank_diff(),
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
        Commands::Quantize(args) => quantize(args),
        Commands::VamanaBuild(args) => vamana_build(args),
        Commands::VamanaSearch(args) => vamana_search(args),
    }
}
