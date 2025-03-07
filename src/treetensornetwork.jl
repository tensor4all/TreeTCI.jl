mutable struct TreeTensorNetwork{ValueType} <: AbstractTreeTensorNetwork{ValueType}
    g::NamedGraph
    sitetensors::Vector{Pair{Array{ValueType},Vector{NamedEdge}}}

    function TreeTensorNetwork(
        g::NamedGraph,
        sitetensors::Vector{Pair{Array{ValueType},Vector{NamedEdge}}},
    ) where {ValueType}
        !Graphs.is_cyclic(g) ||
            error("TreeTensorNetwork is not supported for loopy tensor network.")
        new{ValueType}(g, sitetensors)
    end
end
