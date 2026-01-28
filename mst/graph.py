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
        pq = []  #priority queue
        start = 0  #starting vertex
        n = self.adj_mat.shape[0] #number of vertices

        key = [float('inf')] * n  #list for keys and initialize as inf
        parent = [-1] * n  #store parent of each vertex
        
        S = [False] * n #explored vertices, start with none

        heapq.heappush(pq, (0, start)) #push start vertex to pq with 0 weight
        key[start] = 0 

        while pq:
            u = heapq.heappop(pq)[1] #get vertex with min key

            if S[u]: #already explored
                continue

            S[u] = True #mark as explored
            neighbors = self.adj_mat[u].nonzero()[0]  #get neighbors of u, indices

            for v in neighbors:
                weight = self.adj_mat[u, v]

                if not S[v] and key[v] > weight: #v not explored, cost less than current
                    key[v] = weight #update cost
                    heapq.heappush(pq, (key[v], v))
                    parent[v] = u
        
        #construct MST adjacency matrix
        self.mst = np.zeros((n,n)) #empty set/matrix MST, store as self.mst
        for v in range(n):
            if parent[v] != -1:
                self.mst[parent[v], v] = key[v]
                self.mst[v, parent[v]] = key[v] #undirected graph, add both ways