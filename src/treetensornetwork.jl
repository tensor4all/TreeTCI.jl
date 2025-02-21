"""
    struct TreeTensorNetwork

Represents a tree tensor network.
TODO: Check! we first implemented this code in the assumption where each tensor has site index that mean all tensors are site tensors. However, in the general case, there could be tensor has only auxiliary bonds.


The tree tensor network can be evaluated (or call value) using standard function call notation:
```julia
    ttn = TreeTensorNetwork(...)
    value = ttn([1, 2, 3, 4])
```
The corresponding function is:

    function (ttn::TreeTensorNetwork)(indexset)

Evaluates the tensor train `ttn` at indices given by `indexset`.
"""

mutable struct TreeTensorNetwork
    tn::TensorNetwork

    function TreeTensorNetwork(tn::TensorNetwork)
        !Graphs.is_cyclic(tn) || error("TreeTensorNetwork is not supported for loopy tensor network.")
        # TODO: we need to check each site tensor has the site index at first dimension.
        new(tn)
    end
end

