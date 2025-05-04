"""
Abstract type for pivot candidate generation strategies
"""
abstract type AbstractPivotCandidateProposer end

"""
Default strategy that uses kronecker product and union with extra indices
"""
struct DefaultPivotCandidateProposer <: AbstractPivotCandidateProposer end

"""
Truncated default strategy that uses kronecker product and union with extra indices
"""
struct TruncatedDefaultPivotCandidateProposer <: AbstractPivotCandidateProposer end

"""
Default strategy that runs through within all indices of site tensor according to the bond and connect them with IJSet from neighbors
"""
function generate_pivot_candidates(
    ::DefaultPivotCandidateProposer,
    tci::SimpleTCI{ValueType},
    site::Int,
    subkeys::Vector{SubTreeVertex},
) where {ValueType}

    if isempty(subkeys)
        Outkey = [site]
        pivots = [fill(0, 1)]
        site_index = 1
    else
        Outkey, pivots = pivotset_with_site_index(tci.IJset, subkeys, site)
        site_index = findfirst(==(site), Outkey)
    end

    Iset = kronecker(pivots, site_index, tci.localdims[site])

    extraIJset = !isempty(tci.IJset_history) ?
        tci.IJset_history[end] :
        Dict(key => MultiIndex[] for key in keys(tci.IJset))

    extraIJset = get(extraIJset, Outkey, MultiIndex[])

    return Outkey => union(Iset, extraIJset)
end


function generate_pivot_candidates(
    ::TruncatedDefaultPivotCandidateProposer,
    tci::SimpleTCI{ValueType},
    site::Int,
    subkeys::Vector{SubTreeVertex},
) where {ValueType}
    outkey, pivots = generate_pivot_candidates(DefaultPivotCandidateProposer(), tci, site, subkeys)
    chis = tci.localdims[site] * length(tci.IJset[outkey])
    pivots = sample_ordered_pivots(pivots, chis)
    return outkey => pivots
end

function pivotset_with_site_index(
    IJset::Dict{SubTreeVertex, Vector{MultiIndex}},
    Inkeys::Vector{SubTreeVertex},
    site::Int,
)::Tuple{SubTreeVertex, Vector{MultiIndex}}
    if isempty(Inkeys)
        # Only the site index remains, which implies it's the root or core
        return [[0 for _ in 1:1]], 1  # 1-dimensional zero index, index 1 is the site
    end

    all_keys = sort(unique(reduce(vcat, Inkeys)))
    idx = searchsortedfirst(all_keys, site)
    Outkey = insert!(copy(all_keys), idx, site)

    pivots = MultiIndex[]
    for indices in Iterators.product((IJset[inkey] for inkey in Inkeys)...)
        indexset = zeros(Int, length(Outkey))
        for (inkey, index) in zip(Inkeys, indices)
            for (j, key) in enumerate(inkey)
                id = searchsortedfirst(Outkey, key)
                indexset[id] = index[j]
            end
        end
        push!(pivots, indexset)
    end

    return Outkey, pivots
end

function sample_ordered_pivots(pivots::Vector{MultiIndex}, maxsize::Int)
    n = length(pivots)
    @show n, maxsize
    if n ≤ maxsize
        return pivots
    end
    selected_indices = shuffle(1:n)[1:maxsize]
    return pivots[sort(selected_indices)]
end

function kronecker(
    pivotset::Vector{MultiIndex},
    site_index::Union{Int,Nothing},
    localdims::Int,
)
    isnothing(site_index) && return MultiIndex[]
    return MultiIndex[
        [is[1:site_index-1]..., j, is[site_index+1:end]...] for is in pivotset,
        j = 1:localdims
    ][:]
end
