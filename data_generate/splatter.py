import numpy as np
import pandas as pd
import scprep


class RFunction(object):

    def __init__(self, name, args, setup, body, quiet_setup=True):
        self.name = name
        self.args = args
        self.setup = setup
        self.body = body
        if quiet_setup:
            self.setup = """
                suppressPackageStartupMessages(suppressMessages(
                    suppressWarnings({{
                        {setup}
                    }})))""".format(setup=self.setup)

    @property
    def function(self):
        try:
            return self._function
        except AttributeError:
            from rpy2.robjects.packages import STAP  # NOQA
            import rpy2.robjects.numpy2ri  # NOQA
            function_text = """
            {setup}
            {name} <- function({args}) {{
              {body}
            }}
            """.format(setup=self.setup, name=self.name,
                       args=self.args, body=self.body)
            self._function = getattr(STAP(function_text, self.name), self.name)
            rpy2.robjects.numpy2ri.activate()
            return self._function

    def is_r_object(self, obj):
        return "rpy2.robjects" in str(type(obj))

    def convert(self, robject):
        if self.is_r_object(robject):
            import rpy2.robjects.numpy2ri  # NOQA
            import rpy2.robjects  # NOQA
            if isinstance(robject, rpy2.robjects.vectors.ListVector):
                names = self.convert(robject.names)
                if names is rpy2.rinterface.NULL or \
                        len(names) != len(np.unique(names)):
                    # list
                    robject = [self.convert(obj) for obj in robject]
                else:
                    # dictionary
                    robject = {name: self.convert(
                        obj) for name, obj in zip(robject.names, robject)}
            else:
                # try numpy first
                robject = rpy2.robjects.numpy2ri.ri2py(robject)
                if self.is_r_object(robject):
                    # try regular conversion
                    robject = rpy2.robjects.conversion.ri2py(robject)
        return robject

    def __call__(self, *args, **kwargs):
        import rpy2.robjects.numpy2ri  # NOQA
        robject = self.function(*args, **kwargs)
        return self.convert(robject)


def _load_splat(dropout=0.5, sigma=0.18, method="paths", n_genes=17580,
                seed=None,
                kwargs="group.prob=c(0.3, 0.1, 0.2, 0.2, 0.1, 0.1)"):
    assert seed is not None
    np.random.seed(seed)
    splat = RFunction(name="splat", setup="""
                                library(splatter)
                                library(scater)
                                library(magrittr)""",
                      args="", body="""
                        sim <- splatSimulate(
                            method="{method}", seed={seed},
                            batchCells=3000, #16825,
                            nGenes=17580,
                            mean.shape=6.6, mean.rate = 0.45,
                            lib.loc=8.4 + log(2), lib.scale=0.33,
                            out.prob=0.016, out.facLoc=5.4, out.facScale=0.90,
                            bcv.common={sigma}, bcv.df=21.6,
                            de.prob=0.2,
                            {kwargs},
                            dropout.type="none"
                            )
                        data <- sim %>%
                            counts() %>%
                            t()
                        list(data=data, time=sim$Step, branch=sim$Group)
                        """.format(method=method,
                                   seed=seed,
                                   sigma=sigma,
                                   n_genes=n_genes, kwargs=kwargs))
    data = splat()
    branch, labels = pd.factorize(data['branch'])
    time = data['time']
    data = data['data']
    if dropout > 0:
        data = np.random.binomial(n=data, p=1 - dropout,
                                  size=data.shape)
    if n_genes < data.shape[1]:
        data = data[:, np.random.choice(data.shape[1], n_genes, replace=False)]
    data = scprep.normalize.library_size_normalize(data)
    data = scprep.transform.sqrt(data)
    data = scprep.reduce.pca(data, n_components=100)
    return data, branch, time


def load_splat_paths(dropout=0.5, sigma=0.18, n_genes=17580, seed=np.random.randint(1000)):
    return _load_splat(seed=seed, dropout=dropout, sigma=sigma, n_genes=n_genes,
                       method="paths", kwargs="""
                       group.prob=c(0.3, 0.1, 0.2, 0.2, 0.1, 0.1),
                       path.from=c(0, 0, 1, 1, 3, 3),
                       path.nonlinearProb=0.2,
                       path.skew=c(0.4, 0.5, 0.6, 0.4, 0.5, 0.4)""")


def load_splat_groups(dropout=0.5, sigma=0.18, n_genes=17580, seed=np.random.randint(1000)):
    return _load_splat(seed=seed, dropout=dropout, sigma=sigma, n_genes=n_genes,
                       method="groups", kwargs="""
                       group.prob=c(0.3, 0.1, 0.05, 0.15, 0.05, 0.05,
                                    0.05, 0.15, 0.05, 0.05)""")