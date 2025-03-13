mutable struct TreeTensorNetwork{ValueType}
    tensornetwork::TensorNetwork

    function TreeTensorNetwork(
        g::NamedGraph,
        sitetensors::Vector{Pair{Array{ValueType},Vector{NamedEdge}}},
    ) where {ValueType}
        !Graphs.is_cyclic(g) ||
            error("TreeTensorNetwork is not supported for loopy tensor network.")
        ttntensors = Vector{IndexedArray}()
        for (i, (T, edges)) in enumerate(sitetensors)
            indexs = vcat(
                Index(size(T)[1], "s$i"),
                [
                    Index(size(T)[j+1], "$(src(edges[j]))=>$(dst(edges[j]))") for
                    j = 1:length(edges)
                ],
            )
            t = IndexedArray(T, indexs)
            push!(ttntensors, t)
        end
        tensornetwork = TensorNetwork(ttntensors)
        new{ValueType}(tensornetwork)
    end
end

function crossinterpolate(
    ::Type{ValueType},
    f,
    localdims::Union{Vector{Int},NTuple{N,Int}},
    g::NamedGraph,
    initialpivots::Vector{MultiIndex} = [ones(Int, length(localdims))];
    kwargs...,
) where {ValueType,N}
    tci = SimpleTCI{ValueType}(f, localdims, g, initialpivots)
    ranks, errors = optimize!(tci, f; kwargs...)
    sitetensors = fillsitetensors(tci, f)
    return TreeTensorNetwork(tci.g, sitetensors), ranks, errors
end

function evaluate(
    ttn::TreeTensorNetwork{ValueType},
    indexset::Union{AbstractVector{Int},NTuple{N,Int}},
) where {N,ValueType}
    tn = deepcopy(ttn.tensornetwork)
    if length(indexset) != length(vertices(tn.data_graph))
        throw(
            ArgumentError(
                "To evaluate a tt of length $(length(ttn)), you have to provide $(length(ttn)) indices, but there were $(length(indexset)).",
            ),
        )
    end
    for i = 1:length(vertices(tn.data_graph))
        t = tn[i]
        site = IndexedArray(
            [j == indexset[i] ? 1.0 : 0.0 for j = 1:t.indices[1].dim],
            [t.indices[1]],
        )
        tn[i] = contract(t, site)
    end
    return only(complete_contraction(tn))
end

function (ttn::TreeTensorNetwork{V})(indexset) where {V}
    return evaluate(ttn, indexset)
end
