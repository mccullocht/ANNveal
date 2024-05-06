import java.lang.Comparable;
import java.lang.Iterable;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Implement a Vamana graph for approximate nearest neighbor (ANN) search as
 * described in the DiskANN paper:
 * https://suhasjs.github.io/files/diskann_neurips19.pdf. This is similar to
 * HNSW but the graph is flat rather than having multiple levels.
 * 
 * For this graph vectors will be 64-bit integers and use hamming distance as
 * the distance metric. Hamming distance counts the number of bits that differ
 * between two values, so for instance 1, 2, 4, and 8 are all distance 1 from 0.
 * 
 * For testing search(), consider using VamanaGraph.createTestGraph() to make
 * a small graph with some constant values or a simple sequence.
 */
public class VamanaGraph {
    /**
     * A single node in the graph.
     * 
     * @param vector the vector value for this node.
     * @param edges  a list of outbound links to other nodes. This list should be no
     *               longer than maxDegree.
     */
    record Node(long vector, List<Node> edges) {
        @Override
        public final boolean equals(Object o) {
            return o instanceof Node && this.vector == ((Node) o).vector;
        }

        @Override
        public final int hashCode() {
            return Long.hashCode(this.vector);
        }
    }

    /**
     * A neighbor node and its distance from the query. Neighbors are sorted first
     * by distance then by node.
     */
    public record Neighbor(Node node, int distance) implements Comparable<Neighbor> {
        @Override
        public int compareTo(Neighbor o) {
            int cmp = Integer.compare(this.distance, o.distance);
            return cmp == 0 ? Long.compare(this.node.vector, o.node.vector) : cmp;
        }
    }

    /**
     * A list of the top k search results and the set of all nodes visited.
     */
    public record SearchResult(List<Neighbor> searchResults, List<Node> visited) {
    }

    /**
     * Search for the query vector, returning an approximation of the closest
     * results and the set of node visited.
     * 
     * This is listed as Algorithm 1 (GreedySearch()) in the paper.
     * In addition to the function parameters this takes a start node s; you may use
     * the first node in the graph as the entry point.
     * 
     * <code>
     * Algorithm 1: GreedySearch(s, q, k, L)
     *   Data: Graph G with start node s, query q, result size k, search list size L ≥ k
     *   Result: Result set R containing k-approx NNs, and a set V containing all the visited nodes
     *   begin
     *     initialize sets R ← {s} and V ← {}
     *     while R - V != {} do
     *       let p ← r where r has min distance(q, r) for all r in (R - V)
     *       R.addAll(p.edges)
     *       V.add(p)
     *       if R.size > L then
     *         update R to retain closest searchListSize points to q
     *       return [closest k points from R; V]
     * </code>
     * 
     * @param query          query vector
     * @param k              number of results to return
     * @param searchListSize number of candidates to consider when computing top k
     *                       results. Must be >= k.
     * @return the top K results from the search list and all of the visited
     *         candidates.
     */
    public SearchResult search(long query, int k, int searchListSize) {
        if (k > searchListSize) {
            throw new IllegalArgumentException("require: k <= searchListSize");
        }
        throw new UnsupportedOperationException("unimplemented");
    }

    /**
     * Add a vector to the graph.
     * 
     * @param vector the vector to add to the graph.
     */
    public void add(long vector) {
        Node n = new Node(vector, search(vector, params.beamWidth, params.beamWidth).visited);
        robustPrune(n);
        this.nodes.add(n);
        for (Node e : n.edges) {
            e.edges.add(n);
            robustPrune(e);
        }
    }

    /**
     * Prunes the list of outbound edges from node to at most maxDegree length.
     * 
     * This is listed as Algorithm 2 (RobustPrune()) in the paper.
     * <code>
     * Algorithm 2: RobustPrune(p, V, α, maxDegree)
     *  Data: Graph G, point p , candidate set V, distance threshold α ≥ 1, degree bound maxDegree
     *  Result: G is modified by setting at most maxDegree new out-neighbors for p
     *  begin
     *    V.addAll(p.edges)
     *    p.edges.clear();
     *    while V != {} do
     *      q ← r where r has min distance(p, r) for all r in V
     *      p.edges.add(q)
     *      if p.edges.size() == maxDegree then
     *        break
     *      for r in V do
     *        if α · distance(r, q) ≤ distance(p, q) then
     *          remove r from V
     * </code>
     * 
     * @param neighbors
     * @return
     */
    void robustPrune(Node node) {
        if (node.edges.size() <= this.params.maxDegree) {
            return;
        }

        throw new UnsupportedOperationException("unimplemented");
    }

    /**
     * Parameters for constructing a Vamana graph.
     * 
     * @param maxDegree the maximum number of outbound edges at each node.
     * @param beadWidth the number of candidates in the search during graph
     *                  construction.
     * @param alpha     robust pruning parameter.
     */
    public record ConstructionParams(int maxDegree, int beamWidth, float alpha) {
        /**
         * Construction parameters appropriate for a very small graph (< 256 values).
         */
        public ConstructionParams() {
            this(4, 10, 1.2f);
        }
    }

    final ConstructionParams params;

    /**
     * The list of nodes in the graph.
     */
    final List<Node> nodes;

    /**
     * Initialize a new empty graph.
     * 
     * @param params constructions parameters used when adding nodes to the graph.
     */
    public VamanaGraph(ConstructionParams params) {
        this(params, new ArrayList<>());
    }

    private VamanaGraph(ConstructionParams params, List<Node> nodes) {
        this.params = params;
        this.nodes = nodes;
    }

    /**
     * Create a new graph from the provided vectors for testing search().
     * 
     * This exhaustively computes node distance (O(N^2)) and crudely prunes the
     * set of edges to the closest nodes.
     * 
     * @param vectors to include in the graph
     * @return a graph containing all the vectors with edges between them.
     */
    public static VamanaGraph createTestGraph(ConstructionParams params, List<Long> vectors) {
        // Insert every vector into the graph with no edges.
        var graph = new VamanaGraph(params, vectors.stream().map(v -> new Node(v, new ArrayList<>()))
                .collect(Collectors.toCollection(() -> new ArrayList<>(params.maxDegree))));
        for (Node node : graph.nodes) {
            // Compute the distance between this node and every other node in the graph,
            // then take the maxDegree closest values and insert those nodes as edges.
            node.edges.addAll(graph.nodes.stream().filter(n -> node != n)
                    .map(n -> new Neighbor(n, hammingDistance(node.vector, n.vector))).sorted().limit(params.maxDegree)
                    .map(neighbor -> neighbor.node)
                    .collect(Collectors.toCollection(ArrayList::new)));
        }
        return graph;
    }

    /**
     * @return the number of vectors in the graph.
     */
    public int size() {
        return this.nodes.size();
    }

    /**
     * @param i index; must be less than size()
     * @return the vector value at that index.
     */
    public long getVector(int i) {
        return this.nodes.get(i).vector;
    }

    /**
     * Compute hamming distance between two vectors. This is a count of the number
     * of bits that are different between the two parameters.
     * 
     * @param a first vector.
     * @param b second vector.
     * @return a distance metric (smaller is closer/better).
     */
    public static int hammingDistance(long a, long b) {
        return Long.bitCount(a ^ b);
    }
}

//
// NB: do not paste this into the prompt!
//
// Additional notes:
// * Insertion is implemented by first searching the graph. Nudge toward doing
// search first.
// * It's useful to have a prebuilt graph to test for search. It shouldn't be
// too hard for the candidate to come up with one, but one is also provided
// below with certain properties.
// * Looking for data structure and algorithmic choices around how to perform
// the search and prune routines; wouldn't expect candidates to know the core
// algorithm.
//
// Question extensions:
// * Pick a better entry point node.
// * Compute a recall figure.
// * alpha > 1.0
// * Concurrent graph build
// * Partitioned graph build using kmeans clustering; described in the paper.
