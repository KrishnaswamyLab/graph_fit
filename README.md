# Graph Fit

Different ways to learn or fit a graph to vector data.

## Usage

Your data should be

An *n*-by-*m* data matrix of *n* variable observations in an *m*-dimensional space. Your graph will have *n* nodes.

## Example

```
pip install -r requirements.txt
```

and then

```
import graphfit as gf
X = my_data_from_somewhere()
W = gf.log_barrier(X)
```

is the general idea.
