// XXX remove this
#[allow(dead_code)]
mod vamana;
mod vec;

use std::{
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap, HashMap, HashSet},
    fs::File,
    io::BufWriter,
    num::NonZeroUsize,
    path::PathBuf,
    str::FromStr,
};

use clap::{Args, Parser, Subcommand};
use hnsw::{Hnsw, Searcher};
use ordered_float::NotNan;
use rand_pcg::Pcg64;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::Serialize;
use space::{Metric, Neighbor};
use vec::{BinaryVectorView, FloatVectorView, VectorView, VectorViewStore};

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
    /// Analyze binary vector data.
    Analyze(AnalyzerArgs),
    /// Build an HNSW index of random vectors from a binary query file.
    BuildCentroidIndex(BuildCentroidIndex),
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
struct AnalyzerArgs {
    /// Path to flat vector file.
    #[arg(short, long)]
    binary_vectors: PathBuf,
    /// Number of dimensions in input vectors.
    #[arg(short, long)]
    dimensions: NonZeroUsize,
    /// Distribution of population count of each input vector.
    #[arg(short, long, default_value_t = false)]
    popcnt: bool,
    /// Distribution of vectors with a particular bit set.
    #[arg(short, long, default_value_t = false)]
    set_dist: bool,
    /// Distribution of the first word of various sizes.
    #[arg(short, long, default_value_t = false)]
    word_dist: bool,
    /// Distribution of closest centroids.
    #[arg(short, long, default_value_t = false)]
    rand_centroids: bool,
}

#[derive(Args)]
struct BuildCentroidIndex {
    /// Path to flat binary vector file to use as input.
    #[arg(short, long)]
    binary_vectors: PathBuf,
    /// Number of dimensions in input vectors.
    #[arg(short, long)]
    dimensions: NonZeroUsize,
    /// Path to output index file.
    #[arg(short, long)]
    index: PathBuf,
    /// Fraction of input vector to select for the index.
    #[arg(short, long)]
    frac: f64,
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

#[derive(Serialize)]
struct Hamming;

impl<'a> Metric<BinaryVectorView<'a>> for Hamming {
    type Unit = u32;

    fn distance(&self, a: &BinaryVectorView<'a>, b: &BinaryVectorView<'a>) -> Self::Unit {
        a.distance(b)
    }
}

fn analyze(args: AnalyzerArgs) -> std::io::Result<()> {
    let bin_vector_backing = unsafe { memmap2::Mmap::map(&File::open(args.binary_vectors)?)? };
    let bin_vector_store = vec::VectorViewStore::<'_, BinaryVectorView<'_>>::new(
        bin_vector_backing.as_ref(),
        args.dimensions.get(),
    );

    if args.popcnt {
        // This produces a bell curve around the center of the dimensionality.
        let mut count_dist = Vec::with_capacity(args.dimensions.get() + 1);
        count_dist.resize(args.dimensions.get() + 1, 0usize);
        for v in bin_vector_store.iter() {
            count_dist[v.count_ones() as usize] += 1;
        }
        for (i, c) in count_dist.into_iter().enumerate().filter(|(_, c)| c > &0) {
            println!("{:4} {}", i, c);
        }
    }

    if args.set_dist {
        // Some values occur very infrequently but most are 45-55%
        let mut set_dist = Vec::with_capacity(args.dimensions.get() + 1);
        set_dist.resize(args.dimensions.get() + 1, 0usize);
        for v in bin_vector_store.iter() {
            for b in v.ones_iter() {
                set_dist[b] += 1;
            }
        }
        for (i, p) in set_dist
            .into_iter()
            .enumerate()
            .filter(|(_, c)| c > &0)
            .map(|(i, c)| (i, c as f64 * 100f64 / bin_vector_store.len() as f64))
        {
            println!("{:4} {:.2}%", i, p);
        }
    }

    if args.word_dist {
        // Vector words aren't super random, the data set is ~1M but 2^20 bits only covers a quarter
        // of this or so. At 16 bits it divides the space roughly in half. At 12 bits it covers most of
        // the space, at 8 bits it covers everything.
        // If I do LSH with 16 I end up with 96 different "hash functions" I have to perform a lookup
        // in, which seems prohibitive if I am a completionist. Double the hash size and I only do
        // 48 posting lookups but each one should have only one result.
        let mut h8 = HashSet::new();
        let mut h12 = HashSet::new();
        let mut h16 = HashSet::new();
        let mut h20 = HashSet::new();
        let mut h24 = HashSet::new();
        let mut h28 = HashSet::new();
        let mut h32 = HashSet::new();
        for v in bin_vector_store.iter() {
            let w0 = v.word_iter().next().unwrap();
            h8.insert(w0 & 0xff);
            h12.insert(w0 & 0xfff);
            h16.insert(w0 & 0xffff);
            h20.insert(w0 & 0xffff_f);
            h24.insert(w0 & 0xffff_ff);
            h28.insert(w0 & 0xffff_fff);
            h32.insert(w0 & 0xffff_ffff);
        }
        println!("h8  {}", h8.len());
        println!("h12 {}", h12.len());
        println!("h16 {}", h16.len());
        println!("h20 {}", h20.len());
        println!("h24 {}", h24.len());
        println!("h28 {}", h28.len());
        println!("h32 {}", h32.len());
    }

    if args.rand_centroids {
        // XXX I can probably revert almost everything I'm just getting a version mismatch since
        // hnsw doesn't re-export space::Metric
        let mut hnsw: Hnsw<Hamming, BinaryVectorView, Pcg64, 12, 24> = Hnsw::new(Hamming);
        let mut searcher = Searcher::default();
        println!("building hnsw index");
        for (i, v) in bin_vector_store
            .iter()
            .filter(|v| v.hash() % 6 == 0)
            .enumerate()
        {
            if i % 10000 == 0 {
                println!("{}", i);
            }
            hnsw.insert(v, &mut searcher);
        }
        println!("done building hnsw index");
        let closest_centroids: Vec<usize> = (0..bin_vector_store.len())
            .into_par_iter()
            .map(|i| {
                let p = bin_vector_store.get(i);
                let mut searcher = Searcher::default();
                let mut closest = [Neighbor {
                    index: 0usize,
                    distance: 0u32,
                }; 1];
                hnsw.nearest(&p, 25, &mut searcher, &mut closest);
                closest[0].index
            })
            .collect();
        let mut centroid_counts: HashMap<usize, usize> = HashMap::new();
        for c in closest_centroids {
            centroid_counts
                .entry(c)
                .and_modify(|n| *n += 1)
                .or_insert(1);
        }
        let mut pl_lengths: BTreeMap<usize, usize> = BTreeMap::new();
        for v in centroid_counts.values() {
            pl_lengths.entry(*v).and_modify(|n| *n += 1).or_insert(1);
        }
        let num_pls: usize = pl_lengths.values().sum();
        let mut total = 0usize;
        for (l, c) in pl_lengths {
            total += c;
            println!(
                "{:3} {:7} {:.2}%",
                l,
                c,
                total as f64 * 100.0 / num_pls as f64
            );
        }
    }

    Ok(())
}

fn build_centroid_index(args: BuildCentroidIndex) -> std::io::Result<()> {
    let bin_vector_backing = unsafe { memmap2::Mmap::map(&File::open(args.binary_vectors)?)? };
    let bin_vector_store = vec::VectorViewStore::<'_, BinaryVectorView<'_>>::new(
        bin_vector_backing.as_ref(),
        args.dimensions.get(),
    );
    let max_hash = u64::MAX / 1000 * ((args.frac * 1000.0) as u64);
    let mut hnsw: Hnsw<Hamming, BinaryVectorView, Pcg64, 12, 24> = Hnsw::new(Hamming);
    let mut searcher = Searcher::default();
    for (i, v) in bin_vector_store
        .iter()
        .filter(|v| v.hash() <= max_hash)
        .enumerate()
    {
        if i % 10000 == 0 {
            println!("{}", i);
        }
        hnsw.insert(v, &mut searcher);
    }

    let mut w = BufWriter::new(File::create(args.index)?);
    rmp_serde::encode::write(&mut w, &hnsw).unwrap();
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
        Commands::Analyze(args) => analyze(args),
        Commands::BuildCentroidIndex(args) => build_centroid_index(args),
    }
}
