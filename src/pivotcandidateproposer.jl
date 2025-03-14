"""
Abstract type for pivot candidate generation strategies
"""
abstract type AbstractPivotCandidateProposer end

"""
Default strategy that uses kronecker product and union with extra indices
"""
struct DefaultPivotCandidateProposer <: AbstractPivotCandidateProposer end

"""
Default strategy that runs through within all indices of site tensor according to the bond and connect them with IJSet from neighbors
"""
function generate_pivot_candidates(
    ::DefaultPivotCandidateProposer,
    tci::SimpleTCI{ValueType},
    edge::NamedEdge,
    extraIJset::Dict{SubTreeVertex,Vector{MultiIndex}},
) where {ValueType}

    vp, vq = separatevertices(tci.g, edge)
    Ikey, subIkey = subtreevertices(tci.g, vq => vp), vp
    Jkey, subJkey = subtreevertices(tci.g, vp => vq), vq

    adjacent_edges_vp = adjacentedges(tci.g, vp; combinededges = edge)
    InIkeys = edgeInIJkeys(tci.g, vp, adjacent_edges_vp)

    adjacent_edges_vq = adjacentedges(tci.g, vq; combinededges = edge)
    InJkeys = edgeInIJkeys(tci.g, vq, adjacent_edges_vq)

    # Generate base index sets for both sides
    Iset = kronecker(tci.IJset, Ikey, InIkeys, vp, tci.localdims[vp])
    Jset = kronecker(tci.IJset, Jkey, InJkeys, vq, tci.localdims[vq])

    # Combine with extra indices if available
    Icombined = union(Iset, extraIJset[Ikey])
    Jcombined = union(Jset, extraIJset[Jkey])
    return (Ikey => Jkey), Dict(Ikey => Icombined, Jkey => Jcombined)
end


"""
    kronecker(IJset, target_subtree, source_subtrees, target_site, local_dim)

Generates an index set for a tensor network subspace.

# Arguments
- `IJset::Dict{SubTreeVertex,Vector{MultiIndex}}`: Dictionary containing existing index sets
- `target_subtree::SubTreeVertex`: Subtree vertex to which the resulting indices will be applied
- `source_subtrees::Vector{SubTreeVertex}`: List of subtree vertices used as input sources
- `target_site::Int`: The specific site where local dimension values will be inserted
- `local_dim::Int`: Local dimension of the site (number of possible states)

# Returns
- `Vector{MultiIndex}`: List of generated indices
"""
function kronecker(
    IJset::Dict{SubTreeVertex,Vector{MultiIndex}},
    target_subtree::SubTreeVertex,
    source_subtrees::Vector{SubTreeVertex},
    target_site::Int,
    local_dim::Int,
)::Vector{MultiIndex}
    pivotset = MultiIndex[]
    for indices in Iterators.product((IJset[inkey] for inkey in source_subtrees)...)
        indexset = zeros(Int, length(target_subtree))
        for (inkey, index) in zip(source_subtrees, indices)
            for (idx, key) in enumerate(inkey)
                id = findfirst(==(key), target_subtree)
                indexset[id] = index[idx]
            end
        end
        push!(pivotset, indexset)
    end

    site_index = findfirst(==(target_site), target_subtree)
    filtered_subregions = filter(x -> x ≠ Set([target_site]), target_subtree)

    if site_index === nothing
        return MultiIndex[]
    end

    return MultiIndex[
        [is[1:site_index-1]..., j, is[site_index+1:end]...] for is in pivotset,
        j = 1:local_dim
    ][:]
end
