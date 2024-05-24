// XXX remove this
#[allow(dead_code)]
mod vamana;
mod vec;

use std::{
    cmp::Reverse, collections::BinaryHeap, fs::File, io::BufWriter, num::NonZeroUsize,
    path::PathBuf, str::FromStr,
};

use clap::{Args, Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use ordered_float::NotNan;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use simsimd::BinarySimilarity;
use vamana::Scorer;
use vec::{BinaryVectorView, FloatVectorView, VectorView, VectorViewStore};

use crate::vamana::GraphBuilder;

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
    // Number of dimensions in input vectors.
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
    let vectors_in = unsafe { memmap2::Mmap::map(&File::open(vectors)?)? };
    let vector_store =
        vec::VectorViewStore::<'_, FloatVectorView<'_>>::new(vectors_in.as_ref(), dimensions.get());
    let mut vectors_out = BufWriter::new(File::create(output)?);
    for v in vector_store.iter() {
        v.write_binary_quantized(&mut vectors_out)?;
    }
    Ok(())
}

fn search(args: SearchArgs) -> std::io::Result<()> {
    let query_backing = unsafe { memmap2::Mmap::map(&File::open(args.queries)?)? };
    let query_store = vec::VectorViewStore::<'_, FloatVectorView<'_>>::new(
        query_backing.as_ref(),
        args.dimensions.get(),
    );
    let limit = args
        .num_queries
        .map(NonZeroUsize::get)
        .unwrap_or(query_store.len());

    let vector_backing = unsafe { memmap2::Mmap::map(&File::open(args.vectors)?)? };
    let vector_store = vec::VectorViewStore::<'_, FloatVectorView<'_>>::new(
        vector_backing.as_ref(),
        args.dimensions.get(),
    );
    let bin_vector_backing = match args.binary_vectors {
        Some(p) => Some(unsafe { memmap2::Mmap::map(&File::open(p)?)? }),
        None => None,
    };
    let bin_vector_store = bin_vector_backing.as_ref().map(|b| {
        vec::VectorViewStore::<'_, BinaryVectorView<'_>>::new(b.as_ref(), args.dimensions.get())
    });
    search_queries(
        query_store,
        args.num_candidates,
        limit,
        vector_store,
        bin_vector_store,
        args.oversample,
    );
    Ok(())
}

/// Retrieve num_candidates for query in store and return the result in any order.
fn retrieve<'a, V: VectorView<'a>>(
    query: V,
    store: &VectorViewStore<'a, V>,
    num_candidates: usize,
) -> Vec<Reverse<(NotNan<f32>, usize)>> {
    let mut heap: BinaryHeap<Reverse<(NotNan<f32>, usize)>> =
        BinaryHeap::with_capacity(num_candidates + 1);
    for (i, v) in store.iter().enumerate() {
        let score = NotNan::new(v.score(&query)).unwrap();
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

fn full_retrieve<'a>(
    query: FloatVectorView<'a>,
    store: &VectorViewStore<'a, FloatVectorView<'a>>,
    num_candidates: usize,
) -> Vec<Reverse<(NotNan<f32>, usize)>> {
    let mut results = retrieve(query, store, num_candidates);
    results.sort();
    results
}

fn binary_retrieve<'a>(
    query: FloatVectorView<'a>,
    store: &VectorViewStore<'a, FloatVectorView<'a>>,
    bin_store: &VectorViewStore<'a, BinaryVectorView<'a>>,
    num_candidates: usize,
    oversample: f32,
) -> Vec<Reverse<(NotNan<f32>, usize)>> {
    let binary_query: Vec<u8> = query.binary_quantize().collect();
    let results = retrieve(
        BinaryVectorView::new(binary_query.as_ref(), query.dimensions()),
        bin_store,
        (num_candidates as f32 * oversample) as usize,
    );
    let mut full_results: Vec<Reverse<(NotNan<f32>, usize)>> = results
        .into_iter()
        .map(|e| {
            Reverse((
                NotNan::new(store.get(e.0 .1).score(&query)).unwrap(),
                e.0 .1,
            ))
        })
        .collect();
    full_results.sort();
    full_results.truncate(num_candidates);
    full_results
}

fn search_queries<'a>(
    query_store: VectorViewStore<'a, FloatVectorView<'a>>,
    num_candidates: NonZeroUsize,
    limit: usize,
    vector_store: VectorViewStore<'a, FloatVectorView<'a>>,
    bin_vector_store: Option<VectorViewStore<'a, BinaryVectorView<'a>>>,
    oversample: f32,
) {
    match bin_vector_store {
        Some(bin_store) => {
            for q in query_store.iter().take(limit) {
                assert_ne!(
                    binary_retrieve(
                        q,
                        &vector_store,
                        &bin_store,
                        num_candidates.get(),
                        oversample
                    )
                    .len(),
                    0
                );
            }
        }
        None => {
            for q in query_store.iter().take(limit) {
                assert_ne!(
                    full_retrieve(q, &vector_store, num_candidates.get(),).len(),
                    0
                );
            }
        }
    }
}

struct BinaryVectorStore<'a> {
    vector_data: &'a [u8],
    stride: usize,
}

impl<'a> BinaryVectorStore<'a> {
    fn new(vector_data: &'a [u8], dimensions: NonZeroUsize) -> Self {
        let stride = (dimensions.get() + 7) / 8;
        assert_eq!(vector_data.len() % stride, 0);
        Self {
            vector_data,
            stride,
        }
    }

    fn len(&self) -> usize {
        self.vector_data.len() / self.stride
    }
}

// rayon par_chunks(n) is probably what I want for parallel processing.
// i'm not entirely sure how i model concurrent build. i guess i could internalize it?
// if I do that i don't need par_chunks, i can just use a parallel range iterator and then I have
// to wrap node state in a Mutex
impl<'a> vamana::VectorStore for BinaryVectorStore<'a> {
    // XXX this is kind of awkward. I don't know if this is a good idea.
    type Vector = [u8];

    fn get(&self, i: usize) -> &Self::Vector {
        let offset = i * self.stride;
        &self.vector_data[offset..(offset + self.stride)]
    }

    fn len(&self) -> usize {
        self.vector_data.len() / self.stride
    }
}

struct HammingScorer;

impl Scorer for HammingScorer {
    type Vector = [u8];

    fn score(&self, a: &Self::Vector, b: &Self::Vector) -> NotNan<f32> {
        let dim = (a.len() * 8) as f32;
        let distance = BinarySimilarity::hamming(a, b).unwrap() as f32;
        NotNan::new((dim - distance) / dim).unwrap()
    }
}

fn vamana_build(args: VamanaBuildArgs) -> std::io::Result<()> {
    assert_eq!(args.coding, VectorCoding::Bin);
    let vectors_in = unsafe { memmap2::Mmap::map(&File::open(args.vectors)?)? };
    let store = BinaryVectorStore::new(&vectors_in, args.dimensions);
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
    }
}
