abstract type AbstractTreeTensorNetwork{V} <: Function end

"""
    function evaluate(
        ttn::TreeTensorNetwork{V},
        indexset::Union{AbstractVector{Int}, NTuple{N, Int}}
    )::V where {V}

Evaluates the tensor train `tt` at indices given by `indexset`.
"""
function evaluate(
    ttn::AbstractTreeTensorNetwork{V},
    indexset::Union{AbstractVector{Int},NTuple{N,Int}},
)::V where {N,V}
    if length(indexset) != length(ttn.sitetensors)
        throw(
            ArgumentError(
                "To evaluate a tt of length $(length(ttn)), you have to provide $(length(ttn)) indices, but there were $(length(indexset)).",
            ),
        )
    end
    sitetensors = IndexedArray[]
    # TODO: site tensorを作る関数を作成
    for (Tinfo, i) in zip(ttn.sitetensors, indexset)
        T, edges = Tinfo
        inds = (i, ntuple(_ -> :, ndims(T) - 1)...)
        T = T[inds...]
        indexs = [
            Index(size(T)[j], "$(src(edges[j]))=>$(dst(edges[j]))") for j = 1:length(edges)
        ]
        t = IndexedArray(T, indexs)
        push!(sitetensors, t)
    end
    tn = TensorNetwork(sitetensors)
    return only(complete_contraction(tn))
end

function (ttn::AbstractTreeTensorNetwork{V})(indexset) where {V}
    return evaluate(ttn, indexset)
end
