import java.lang.Comparable;
import java.lang.Iterable;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Implement a Vamana graph for approximate nearest neighbor (ANN) search as
 * described in the DiskANN paper:
 * https://suhasjs.github.io/files/diskann_neurips19.pdf. This is similar to
 * HNSW but the graph is flat rather than having multiple levels.
 * 
 * For this graph vectors will be 64-bit integers and use hamming distance as
 * the distance metric.
 */
public class VamanaGraph {
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
     *     initialize sets L ← {s} and V ← {}
     *     while R - V != {} do
     *       let p ← q where q min distance(p, q) in (R - V)
     *       R.addAll(p.edges)
     *       V.add(p)
     *       if R.size > L then
     *         update R to retain closest L points to q
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
    public SearchResult search(long query, int k, int numCandidates) {
        if (k > numCandidates) {
            throw new IllegalArgumentException("k must be less than numCandidates");
        }
        throw new UnsupportedOperationException("unimplemented");
    }

    /**
     * Add a vector to the graph.
     * 
     * <code>
     * Algorithm:
     *   * search for vector to obtain visited set
     *   * prune visited set to params.maxDegrees
     *   * insert node in graph with pruned neighbors
     *   * update each neighbor to point back to the new node
     * </code>
     * 
     * @param vector the vector to add to the graph.
     */
    public void add(long vector) {
        throw new UnsupportedOperationException("unimplemented");
    }

    /**
     * Prunes the list of outbound edges from node to at most maxDegree length.
     * 
     * This is listed as Algorithm 2 (RobustPrune()) in the paper.
     * <code>
     * Algorithm 2: RobustPrune(p, V, α, maxDegree)
     *  Data: Graph G, point p ∈ P, candidate set V, distance threshold α ≥ 1, degree bound maxDegree
     *  Result: G is modified by setting at most maxDegree new out-neighbors for p
     *  begin
     *    V.addAll(p.edges)
     *    p.edges.clear();
     *    while V != {} do
     *      q ← r where r min distance(p, r) in V
     *      p.edges.add(q)
     *      if p.edges.size() == maxDegree then
     *        break
     *      for r in V do
     *        if α · distance(r, q) ≤ d(p, q) then
     *          remove r from V
     * </code>
     * 
     * @param neighbors
     * @return
     */
    void robustPrune(Node node) {
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

    /**
     * A single node in the graph.
     * 
     * @param vector the vector value for this node.
     * @param edges  a list of other nodes by index ordinal. This list should be no
     *               longer than params.maxDegree.
     */
    record Node(Long vector, List<Integer> edges) {
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
        this(params, List.of());
    }

    /**
     * Initialize with a pre-built graph.
     * 
     * @param params constructions parameters used when adding nodes to the graph.
     * @param nodes  list of nodes and edges.
     */
    public VamanaGraph(ConstructionParams params, List<Node> nodes) {
        this.params = params;
        this.nodes = new ArrayList<>(nodes);
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
     * A neighbor, expressed as an index ordinal in the node graph and a distance.
     * 
     * By default, Neighbors are sorted first by distance then by index.
     */
    public record Neighbor(int index, int distance) implements Comparable<Neighbor> {
        @Override
        public int compareTo(Neighbor o) {
            int cmp = Integer.compare(this.distance, o.distance);
            return cmp == 0 ? Integer.compare(this.index, o.index) : cmp;
        }
    }

    /**
     * A list of the top k search results and the set of all nodes visited.
     */
    public record SearchResult(List<Neighbor> searchResults, List<Integer> visited) {
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
class OmitFromPrompt {
    // Small graph with vector values in 0..16 with maxDegree of 4.
    // In this graph the distance from each node to all of its edges is 1.
    static VamanaGraph smallGraph() {
        return new VamanaGraph(
                new VamanaGraph.ConstructionParams(4, 10, 1.0f),
                List.of(new VamanaGraph.Node(0L, List.of(1, 2, 4, 8)),
                        new VamanaGraph.Node(1L, List.of(0, 3, 5, 9)),
                        new VamanaGraph.Node(2L, List.of(0, 3, 6, 10)),
                        new VamanaGraph.Node(3L, List.of(1, 2, 7, 11)),
                        new VamanaGraph.Node(4L, List.of(0, 5, 6, 12)),
                        new VamanaGraph.Node(5L, List.of(1, 4, 7, 13)),
                        new VamanaGraph.Node(6L, List.of(2, 4, 7, 14)),
                        new VamanaGraph.Node(7L, List.of(3, 5, 6, 15)),
                        new VamanaGraph.Node(8L, List.of(0, 9, 10, 12)),
                        new VamanaGraph.Node(9L, List.of(1, 8, 11, 13)),
                        new VamanaGraph.Node(10L, List.of(2, 8, 11, 14)),
                        new VamanaGraph.Node(11L, List.of(3, 9, 10, 15)),
                        new VamanaGraph.Node(12L, List.of(4, 8, 13, 14)),
                        new VamanaGraph.Node(13L, List.of(5, 9, 12, 15)),
                        new VamanaGraph.Node(14L, List.of(6, 10, 12, 15)),
                        new VamanaGraph.Node(15L, List.of(7, 11, 13, 14))));
    }
}