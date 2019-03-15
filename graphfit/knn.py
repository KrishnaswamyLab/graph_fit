import pygsp

def get_knn(X, k=10):
    G = pygsp.graphs.NNGraph(X, NNtype='knn', k=k)
    return G.W.toarray()