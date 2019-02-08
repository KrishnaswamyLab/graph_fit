import time
import numpy as np
from scipy import spatial
from sklearn.covariance import graph_lasso
from pyunlocbox import functions, solvers
from .utils import weight2degmap

def l2_reg(X, dist_type='sqeuclidean', alpha=1, s=None, step=0.5,
                  w0=None, maxit=1000, rtol=1e-5, retall=False,
                  verbosity='NONE'):
    r"""
    Learn graph by regularizing the l2-norm of the degrees.

    FROM DONG.

    This is done by solving
    :math:`\tilde{W} = \underset{W \in \mathcal{W}_m}{\text{arg}\min} \,
    \|W \odot Z\|_{1,1} + \alpha \|W1}\|^2 + \alpha \| W \|_{F}^{2}`, subject
    to :math:`\|W\|_{1,1} = s`, where :math:`Z` is a pairwise distance matrix,
    and :math:`\mathcal{W}_m`is the set of valid symmetric weighted adjacency
    matrices.

    Parameters
    ----------
    X : array_like
        An N-by-M data matrix of N variable observations in an M-dimensional
        space. The learned graph will have N nodes.
    dist_type : string
        Type of pairwise distance between variables. See
        :func:`spatial.distance.pdist` for the possible options.
    alpha : float, optional
        Regularization parameter acting on the l2-norm.
    s : float, optional
        The "sparsity level" of the weight matrix, as measured by its l1-norm.
    step : float, optional
        A number between 0 and 1 defining a stepsize value in the admissible
        stepsize interval (see [Komodakis & Pesquet, 2015], Algorithm 6)
    w0 : array_like, optional
        Initialization of the edge weights. Must be an N(N-1)/2-dimensional
        vector.
    maxit : int, optional
        Maximum number of iterations.
    rtol : float, optional
        Stopping criterion. Relative tolerance between successive updates.
    retall : boolean
        Return solution and problem details. See output of
        :func:`pyunlocbox.solvers.solve`.
    verbosity : {'NONE', 'LOW', 'HIGH', 'ALL'}, optional
        Level of verbosity of the solver. See :func:`pyunlocbox.solvers.solve`.

    Returns
    -------
    W : array_like
        Learned weighted adjacency matrix
    problem : dict, optional
        Information about the solution of the optimization. Only returned if
        retall == True.
    """

    # Parse X
    N = X.shape[0]
    E = int(N * (N - 1.) / 2.)
    z = spatial.distance.pdist(X, dist_type)  # Pairwise distances

    # Parse s
    s = N if s is None else s

    # Parse step
    if (step <= 0) or (step > 1):
        raise ValueError("step must be a number between 0 and 1.")

    # Parse initial weights
    w0 = np.zeros(z.shape) if w0 is None else w0
    if (w0.shape != z.shape):
        raise ValueError("w0 must be of dimension N(N-1)/2.")

    # Get primal-dual linear map
    one_vec = np.ones(E)

    def K(w):
        return np.array([2 * np.dot(one_vec, w)])

    def Kt(n):
        return 2 * n * one_vec

    norm_K = 2 * np.sqrt(E)

    # Get weight-to-degree map
    S, St = weight2degmap(N)

    # Assemble functions in the objective
    f1 = functions.func()
    f1._eval = lambda w: 2 * np.dot(w, z)
    f1._prox = lambda w, gamma: np.maximum(0, w - (2 * gamma * z))

    f2 = functions.func()
    f2._eval = lambda w: 0.
    f2._prox = lambda d, gamma: s

    f3 = functions.func()
    f3._eval = lambda w: alpha * (2 * np.sum(w**2) + np.sum(S(w)**2))
    f3._grad = lambda w: alpha * (4 * w + St(S(w)))
    lipg = 2 * alpha * (N + 1)

    # Rescale stepsize
    stepsize = step / (1 + lipg + norm_K)

    # Solve problem
    solver = solvers.mlfbf(L=K, Lt=Kt, step=stepsize)
    problem = solvers.solve([f1, f2, f3], x0=w0, solver=solver, maxit=maxit,
                            rtol=rtol, verbosity=verbosity)

    # Transform weight matrix from vector form to matrix form
    W = spatial.distance.squareform(problem['sol'])

    if retall:
        return W, problem
    else:
        return W
