import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """

        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        n = self.adj_mat.shape[0] #number of vertices in the graph
        self.mst = np.full((n,n), 0) #empty set/matrix MST, store as self.mst
        S = n * [False]  #explored vertices, start with none
        pq = [] #priority queue... put in (weight, start_vertex, end_vertex)
        
        S[0] = True  #start

        #add edges from start to pq
        for v in range(n):
            edge_weight = self.adj_mat[0][v]
            if edge_weight != 0: # can we have negative edge weights...?
                heapq.heappush(pq, (edge_weight, 0, v)) # heapq to pq!

        edge_count = 0 #track so we can repeat loop n - 1 times

        while pq and edge_count < n - 1:
            weight, u, v = heapq.heappop(pq) #get the minimum cost edge
            if not S[v]: #NOT explored
                S[v] = True #explored
                self.mst[u][v] = weight #add edge to MST
                self.mst[v][u] = weight #undirected graph, add both ways...?
                edge_count += 1 #edge count goes up!

                #must add new edges from v, e for edges lol
                for e in range(n):
                    new_weight = self.adj_mat[v][e]
                    if new_weight != 0 and not S[e]:
                        heapq.heappush(pq, (new_weight, v, e)) 