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

@doc raw"""
    function crossinterpolate(
        ::Type{ValueType},
        f,
        localdims::Union{Vector{Int},NTuple{N,Int}},
        g::NamedGraph,
        initialpivots::Vector{MultiIndex}=[ones(Int, length(localdims))];
        kwargs...
    ) where {ValueType,N}

Cross interpolate a function ``f(\mathbf{u})`` using the 2-site TCI algorithm.

# Arguments:
- `ValueType` is the return type of `f`. Automatic inference is too error-prone.
- `localdims::Union{Vector{Int},NTuple{N,Int}}` is a `Vector` (or `Tuple`) that contains the local dimension of each index of `f`.
- `f` is the function to be interpolated. `f` should have a single parameter, which is a vector of the same length as `localdims`. The return type should be `ValueType`.
- `g::NamedGraph` is the graph on which the function is defined.
- `initialpivots::Vector{MultiIndex}` is a vector of pivots to be used for initialization. Default: `[1, 1, ...]`.

# Keywords
- `tolerance::Union{Float64,Nothing} = nothing`: Error tolerance for convergence
- `maxbonddim::Int = typemax(Int)`: Maximum bond dimension
- `maxiter::Int = 20`: Maximum number of iterations
- `sweepstrategy::AbstractSweep2sitePathProposer = DefaultSweep2sitePathProposer()`: Strategy for sweeping
- `pivotstrategy::AbstractPivotCandidateProposer = DefaultPivotCandidateProposer()`: Strategy for proposing pivot candidates
- `verbosity::Int = 0`: Verbosity level
- `loginterval::Int = 10`: Interval for logging
- `normalizeerror::Bool = true`: Whether to normalize errors
- `ncheckhistory::Int = 3`: Number of history steps to check

Notes:
- Set `tolerance` to be > 0 or `maxbonddim` to some reasonable value. Otherwise, convergence is not reachable.
- By default, no caching takes place. Use the [`CachedFunction`](@ref) wrapper if your function is expensive to evaluate.


See also: [`optimize!`](@ref)
"""

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

# Add length method for TreeTensorNetwork
function Base.length(ttn::TreeTensorNetwork)
    return length(vertices(ttn.tensornetwork.data_graph))
end
