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
    bond::Pair{SubTreeVertex,SubTreeVertex},
    extraIJset::Dict{SubTreeVertex,Vector{MultiIndex}},
) where {ValueType}

    vp, vq = separatevertices(tci.g, tci.regionbonds[bond])
    Ikey, subIkey = subtreevertices(tci.g, vp => vq), vq
    Jkey, subJkey = subtreevertices(tci.g, vq => vp), vp

    distances = bonddistances(tci.g, tci.regionbonds, bond)

    adjacent_bonds_vp = adjacentbonds(tci.g, vp, tci.regionbonds)
    InOutbondsJ, InOutkeysJ =
        inoutbondskeys(tci.g, tci.regionbonds, distances, 0, vp, adjacent_bonds_vp)

    adjacent_bonds_vq = adjacentbonds(tci.g, vq, tci.regionbonds)
    InOutbondsI, InOutkeysI =
        inoutbondskeys(tci.g, tci.regionbonds, distances, 0, vq, adjacent_bonds_vq)

    InIkey, InJkey = first(InOutkeysI), first(InOutkeysJ)

    # Generate base index sets for both sides
    Iset = kronecker(tci.IJset, Ikey, InIkey, subIkey, tci.localdims[subIkey])
    Jset = kronecker(tci.IJset, Jkey, InJkey, subJkey, tci.localdims[subJkey])

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
