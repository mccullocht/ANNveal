from typing import Self

class VamanaGraph:
    """ Implement a Vamana graph for approximate nearest neighbor (ANN) search.

    The algorithm is described in the DiskANN paper: https://suhasjs.github.io/files/diskann_neurips19.pdf.
    This is similar to HNSW but the graph is flat rather than having multiple levels.
    
    For this graph vectors will be integers and use hamming distance as the distance metric. Hamming
    distance counts the number of bits that differ between two values, so for instance 1, 2, 4, and
    8 are all distance 1 from 0.

    For testing search(), consider using VamanaGraph.createTestGraph() to make a small graph with
    some constant values or a simple sequence.

    """

    class Node:
        """A single node in the graph.

        Each node consists of an integer vector value and a list of edges to other nodes.
        Node are identified/compared/hashed by the vector value.
        """
        vector: int
        edges: list[Self]

        def __init__(self, vector: int) -> None:
            self.vector = vector
            self.edges = []
        
        def compute_result(self, query_vector: int) -> tuple[int, Self]:
            """Create a result as a (distance, Node) tuple.

            Hamming distance ius used to provide the distance between query_vector and self.vector.
            """
            return (self.vector ^ query_vector).bit_count(), self
        
        def __repr__(self) -> str:
            return f"Node[vector:{self.vector} edges:{[e.vector for e in self.edges]}]"
        
        def __eq__(self, value: object) -> bool:
            return isinstance(value, self.__class__) and self.vector == value.vector
        
        def __lt__(self, value: object) -> bool:
            return isinstance(value, self.__class__) and self.vector < value.vector

        def __hash__(self) -> int:
            return hash(self.vector)
    
    """All nodes in the graph."""
    nodes: list[Node]
    """Maximum number of edges for each node."""
    max_degree: int = 4
    """Size of the search list during graph construction."""
    beam_width: int = 10
    """Parameter for sparse neighborhood graph (SNG) pruning."""
    alpha: float = 1.2

    def __init__(self, max_degree: int = 4, beam_width: int = 10, alpha: float = 1.2) -> None:
        self.nodes = []
        self.max_degree = max_degree
        self.beam_width = beam_width
        self.alpha = alpha
    
    @staticmethod
    def create_test_graph(vectors: list[int]) -> Self:
        """Create a test graph from a list of vectors.
        
        This function exhaustively computes distances between each node and all other nodes (N^2)
        and crudely prunes to the set of closest edges. This is useful for testing a search()
        implementation.
        """
        graph = VamanaGraph()
        graph.nodes = [VamanaGraph.Node(v) for v in vectors]
        for node in graph.nodes:
            candidates = [n.compute_result(node.vector) for n in graph.nodes if n != node]
            candidates.sort()
            candidates = candidates[:graph.max_degree]
            node.edges = [c[1] for c in candidates]
        return graph
    
    def search(self, query_vector: int, num_results: int, search_list_size: int) -> tuple[list[tuple[int, Node]], list[Node]]:
        """search this graph for query_vector.
        
        This function returns a result set in ascending order by distance and a set of all visited
        nodes which may be used during graph construction.

        num_results indicates how many results should be returned; search_list_size controls how
        deep/expensive the search is. search_list_size must be >= num_results.
        """
        if search_list_size < num_results:
            raise ValueError("search_list_size must be >= num_results")
        
        if not self.nodes:
            return [], []

        # Arbitrarily choose the first node as the entry point to the graph.
        start_node: VamanaGraph.Node = self.nodes[0]

        """
        initialize sets resultSet ← {startNode} and visitedSet ← {}
        while resultSet - visitedSet != {} do
          let point ← r where r has min distance(query, r) for all r in (resultSet - visitedSet)
          add all of point.edges to resultSet
          add point to visitedSet
          if resultSet.size > searchListSize then
            update resultSet to retain closest searchListSize points to query
        return [closest k points from resultSet; visitedSet]
        """

        raise NotImplementedError()
    
    def robust_prune(self, node: Node) -> None:
        """Prune node.edges such that there are at most max_degree entries.

        In general closer nodes are preferred although a sparse neighborhood graph (SNG) metric is
        applied to avoid too many links to similar neighborhoods.
        """
        if len(node.edges) < self.max_degree:
            return

        """
        visitedSet ← node.edges
        node.edges = {}
        while visitedSet != {} do
          q ← r where r has min distance(node, r) for all r in visitedSet
          node.edges.add(q)
          if node.edges.size() == maxDegree then
            break
          for r in visitedSet do
            if α · distance(r, q) ≤ distance(node, q) then
              remove r from V
        """
        raise NotImplementedError()
    
    def add(self, vector: int) -> None:
        """Add a new vector to the graph.
        
        A search is performed for the new vector to produce a list of candidate edges that are then
        pruned to a list of at most max_degree length. Once edges are selected we add reciprocal
        edges to neighbors of the new node and prune those nodes too.
        """
        node = VamanaGraph.Node(vector, self.search(vector, self.beam_width, self.beam_width)[1])
        self.robust_prune(node)
        self.nodes.append(node)
        for neighbor in node.edges:
            neighbor.edges.append(node)
            self.robust_prune(neighbor)
