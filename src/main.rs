mod vec;

use std::{
    cmp::Reverse, collections::BinaryHeap, fs::File, io::BufWriter, num::NonZeroUsize,
    path::PathBuf, str::FromStr,
};

use clap::{Args, Parser, Subcommand};
use ordered_float::NotNan;
use vec::{FloatVectorView, VectorView, VectorViewStore};

use crate::vec::BinaryVectorView;

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

#[derive(Clone)]
enum VectorCoding {
    F32,
}

impl FromStr for VectorCoding {
    type Err = &'static str;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "f32" => Ok(VectorCoding::F32),
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
    }
}
