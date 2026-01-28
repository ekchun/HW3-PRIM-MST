import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'

    # easiest... check that MST has exactly n-1 edges
    edges = np.sum(mst > 0)
    assert edges == mst.shape[0] - 1, 'Proposed MST has incorrect edge count'

    # should always be connected
    degrees = np.sum(mst > 0, axis = 1)
    assert np.all(degrees > 0), 'Proposed MST has disconnected vertices'

    # symmetrical bc we defined undirected graph
    assert np.allclose(mst, mst.T), 'Proposed MST is not symmetrical'

    #trying to use the arguements? adj_mat
    edge = mst > 0
    assert np.all(approx_equal(mst[edge], adj_mat[edge])), "MST edges don't match original graph edges"


def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student1():
    """
    
    Simple two-node graph test case. Is this too similar to test_mst_small?
    
    """
    adjmat = np.array([[0, 3],
                       [3, 0]])
    g = Graph(adjmat)
    g.construct_mst()

    check_mst(g.adj_mat, g.mst, 3)


def test_mst_student2():
    """
    
    n - 1 edges again...
    
    """
    g = Graph('../data/small.csv')
    g.construct_mst()

    n = g.mst.shape[0]
    num_edges = np.sum(g.mst > 0)

    assert num_edges == n - 1

def test_mst_student3():
    """
    
    Check for symmetry and no self-loops.
    I don't know what else to do...
    
    """
    coords = np.loadtxt('../data/slingshot_example.txt')
    dist_mat = pairwise_distances(coords)

    g = Graph(dist_mat)
    g.construct_mst()

    # symmetry
    assert np.allclose(g.mst, g.mst.T)
    # no self-loops
    assert np.all(np.diag(g.mst) == 0)
