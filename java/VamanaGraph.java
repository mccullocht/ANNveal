import java.lang.Comparable;
import java.util.ArrayList;
import java.util.List;
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
     * by distance then by vector value.
     */
    public record Neighbor(Node node, int distance) implements Comparable<Neighbor> {
        /**
         * Create a new Neighbor from a query vector and a node.
         *
         * @param query vector
         * @param node  in graph
         * @return a Neighbor containing the node and measured distance between the
         *         query and the node.
         */
        public static Neighbor create(long query, Node node) {
            return new Neighbor(node, hammingDistance(query, node.vector));
        }

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

        if (this.nodes.isEmpty()) {
            return new SearchResult(List.of(), List.of()); // cannot search an empty graph.
        }

        // Entry point to the graph is chosen arbitrarily.
        Node startNode = this.nodes.get(0);

        // initialize sets resultSet ← {startNode} and visitedSet ← {}
        // while resultSet - visitedSet != {} do
        // let point ← r where r has min distance(query, r) for all r in (resultSet -
        // visitedSet)
        // resultSet.addAll(point.edges)
        // visitedSet.add(point)
        // if resultSet.size > searchListSize then
        // update resultSet to retain closest searchListSize points to query
        // return [closest k points from resultSet; visitedSet]

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
     *
     * @param neighbors
     * @return
     */
    void robustPrune(Node node) {
        if (node.edges.size() <= this.params.maxDegree) {
            return;
        }

        // visitedSet ← node.edges
        // node.edges = {}
        // while visitedSet != {} do
        // q ← r where r has min distance(node, r) for all r in visitedSet
        // node.edges.add(q)
        // if node.edges.size() == maxDegree then
        // break
        // for r in visitedSet do
        // if α · distance(r, q) ≤ distance(node, q) then
        // remove r from V

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
     * Return hamming distance (count of differing bits) between 2 long values.
     *
     * @param a
     * @param b
     */
    public static int hammingDistance(long a, long b) {
        return Long.bitCount(a ^ b);
    }
}

//
// NB: do not paste this into the prompt!
//
// Additional notes:
// * Looking for data structure and algorithmic choices around how to perform
// the search and prune routines; wouldn't expect candidates to know the core
// algorithm.
// * createTestGraph() is there to make something simple that can be used for
// testing the search algorithm without dealing with pruning yet.
// * add() has already been implemented because it is pretty small and doesn't
// involve any interesting algorithmic choices.
//
// Question extensions:
// * Pick a better entry point node.
// * Compute a recall figure.
// * alpha > 1.0
// * Concurrent graph build
// * Partitioned graph build using kmeans clustering; described in the paper.
