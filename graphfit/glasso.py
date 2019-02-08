import time
import numpy as np
from sklearn.covariance import graph_lasso

def glasso(X, alpha=1, w0=None, maxit=1000, rtol=1e-5, retall=False,
           verbosity='NONE'):
    r"""
    Learn graph by imposing promoting sparsity in the inverse covariance.

    This is done by solving
    :math:`\tilde{W} = \underset{W \succeq 0}{\text{arg}\min} \,
    -\log \det W - \text{tr}(SW) + \alpha\|W \|_{1,1},
    where :math:`S` is the empirical (sample) covariance matrix.

    Parameters
    ----------
    X : array_like
        An N-by-M data matrix of N variable observations in an M-dimensional
        space. The learned graph will have N nodes.
    alpha : float, optional
        Regularization parameter acting on the l1-norm
    w0 : array_like, optional
        Initialization of the inverse covariance. Must be an N-by-N symmetric
        positive semi-definite matrix.
    maxit : int, optional
        Maximum number of iterations.
    rtol : float, optional
        Stopping criterion. If the dual gap goes below this value, iterations
        are stopped. See :func:`sklearn.covariance.graph_lasso`.
    retall : boolean
        Return solution and problem details.
    verbosity : {'NONE', 'ALL'}, optional
        Level of verbosity of the solver.
        See :func:`sklearn.covariance.graph_lasso`/

    Returns
    -------
    W : array_like
        Learned inverse covariance matrix
    problem : dict, optional
        Information about the solution of the optimization. Only returned if
        retall == True.

    """

    # Parse X
    S = np.cov(X)

    # Parse initial point
    w0 = np.ones(S.shape) if w0 is None else w0
    if (w0.shape != S.shape):
        raise ValueError("w0 must be of dimension N-by-N.")

    # Solve problem
    tstart = time.time()
    res = graph_lasso(emp_cov=S,
                      alpha=alpha,
                      cov_init=w0,
                      mode='cd',
                      tol=rtol,
                      max_iter=maxit,
                      verbose=(verbosity == 'ALL'),
                      return_costs=True,
                      return_n_iter=True)

    problem = {'sol':       res[1],
               'dual_sol':  res[0],
               'solver':    'sklearn.covariance.graph_lasso',
               'crit':      'dual_gap',
               'niter':     res[3],
               'time':      time.time() - tstart,
               'objective': np.array(res[2])[:, 0]}

    W = problem['sol']

    if retall:
        return W, problem
    else:
        return W
