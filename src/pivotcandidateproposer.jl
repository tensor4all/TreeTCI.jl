"""
Abstract type for pivot candidate generation strategies
"""
abstract type AbstractPivotCandidateProposer end

"""
Default strategy that uses kronecker product and union with extra indices
"""
struct DefaultPivotCandidateProposer <: AbstractPivotCandidateProposer end


"""
Simple strategy that uses kronecker product and union with extra indices
"""
struct SimplePivotCandidateProposer <: AbstractPivotCandidateProposer end


function generate_pivot_candidates(
    ::DefaultPivotCandidateProposer,
    tci::SimpleTCI{ValueType},
    edge::NamedEdge;
    maxbonddim::Int = typemax(Int),
) where {ValueType}
    (Ikey, Jkey), (Ipivots, Jpivots), (Isite_index, Jsite_index), (vp, vq) =
        _prepare_keys_and_pivots(tci, edge)

    Iset = kronecker(Ipivots, Isite_index, tci.localdims[vp])
    Jset = kronecker(Jpivots, Jsite_index, tci.localdims[vq])

    extraIJset = if length(tci.IJset_history) > 0
        tci.IJset_history[end]
    else
        Dict(key => MultiIndex[] for key in keys(tci.IJset))
    end

    Icombined = union(Iset, extraIJset[Ikey])
    Jcombined = union(Jset, extraIJset[Jkey])
    return Dict(Ikey => Icombined, Jkey => Jcombined)
end

function generate_pivot_candidates(
    ::SimplePivotCandidateProposer,
    tci::SimpleTCI{ValueType},
    edge::NamedEdge;
    maxbonddim::Int = typemax(Int),
) where {ValueType}
    (Ikey, Jkey), (Ipivots, Jpivots), (Isite_index, Jsite_index), (vp, vq) =
        _prepare_keys_and_pivots(tci, edge)

    Icombined = sample_ordered_pivots(Ipivots, maxbonddim)
    Jcombined = sample_ordered_pivots(Jpivots, maxbonddim)

    Iset = kronecker(Ipivots, Isite_index, tci.localdims[vp])
    Jset = kronecker(Jpivots, Jsite_index, tci.localdims[vq])

    extraIJset = if length(tci.IJset_history) > 0
        tci.IJset_history[end]
    else
        Dict(key => MultiIndex[] for key in keys(tci.IJset))
    end

    Icombined = union(Iset, extraIJset[Ikey])
    Jcombined = union(Jset, extraIJset[Jkey])
    return Dict(Ikey => Icombined, Jkey => Jcombined)
end

function _prepare_keys_and_pivots(
    tci::SimpleTCI{ValueType},
    edge::NamedEdge
) where {ValueType}
    vp, vq = separatevertices(tci.g, edge)
    Ikey = subtreevertices(tci.g, vq => vp)
    Jkey = subtreevertices(tci.g, vp => vq)

    # Prepare I-side
    adjacent_edges_vp = adjacentedges(tci.g, vp; combinededges = edge)
    InIkeys = edgeInIJkeys(tci.g, vp, adjacent_edges_vp)
    Ipivots = pivotset(tci.IJset, InIkeys, Ikey, tci.localdims[vp])
    Isite_index = findfirst(==(vp), Ikey)

    # Prepare J-side
    adjacent_edges_vq = adjacentedges(tci.g, vq; combinededges = edge)
    InJkeys = edgeInIJkeys(tci.g, vq, adjacent_edges_vq)
    Jpivots = pivotset(tci.IJset, InJkeys, Jkey, tci.localdims[vq])
    Jsite_index = findfirst(==(vq), Jkey)

    return (Ikey, Jkey), (Ipivots, Jpivots), (Isite_index, Jsite_index), (vp, vq)
end

function pivotset(
    IJset::Dict{SubTreeVertex,Vector{MultiIndex}},
    Inkeys::Vector{SubTreeVertex},
    Outkey::SubTreeVertex,  # original subregions order
    localdim::Int,
)
    pivotset = MultiIndex[]
    for indices in Iterators.product((IJset[inkey] for inkey in Inkeys)...)
        indexset = zeros(Int, length(Outkey))
        for (inkey, index) in zip(Inkeys, indices)
            for (idx, key) in enumerate(inkey)
                id = findfirst(==(key), Outkey)
                indexset[id] = index[idx]
            end
        end
        push!(pivotset, indexset)
    end
    return pivotset
end

function sample_ordered_pivots(pivots::Vector{MultiIndex}, maxsize::Int)
    n = length(pivots)
    if n ≤ maxsize
        return pivots
    end
    selected_indices = shuffle(1:n)[1:maxsize]
    return pivots[sort(selected_indices)]
end

function kronecker(pivotset::Vector{MultiIndex}, site_index::Union{Int,Nothing}, localdims::Int)
    isnothing(site_index) && return MultiIndex[]
    return MultiIndex[
        [is[1:site_index-1]..., j, is[site_index+1:end]...] for is in pivotset,
        j = 1:localdims
    ][:]
end
