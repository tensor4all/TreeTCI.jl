"""
Abstract type for pivot candidate generation strategies
"""
abstract type PivotCandidateProper end

"""
Default strategy that uses kronecker product and union with extra indices
"""
struct DefaultPivotCandidateProper <: PivotCandidateProper end

"""
Default strategy that runs through within all indices of site tensor according to the bond and connect them with IJSet from neighbors
"""
function generate_pivot_candidates(
    ::DefaultPivotCandidateProper,
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

function kronecker(
    IJset::Dict{SubTreeVertex,Vector{MultiIndex}},
    Outkey::SubTreeVertex,  # original subregions order
    Inkeys::Vector{SubTreeVertex},  # original subregions order
    site::Int,          # direct connected site
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

    site_index = findfirst(==(site), Outkey)
    filtered_subregions = filter(x -> x ≠ Set([site]), Outkey)

    return MultiIndex[
        [is[1:site_index-1]..., j, is[site_index+1:end]...] for is in pivotset,
        j = 1:localdim
    ][:]
end
