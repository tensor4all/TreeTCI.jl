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
    g::NamedGraph,
    IJset::Dict{SubTreeVertex,Vector{MultiIndex}},
    IJset_history::Dict{SubTreeVertex,Vector{Vector{MultiIndex}}},
    extraIJset::Dict{SubTreeVertex,Vector{MultiIndex}},
    regionbonds::Dict{Pair{SubTreeVertex,SubTreeVertex},NamedEdge},
    localdims::Vector{Int},
    bond::Pair{SubTreeVertex,SubTreeVertex},
    IJkey::Pair{SubTreeVertex,SubTreeVertex},
    subloclkey::Pair{Int,Int},
)
    Ikey, Jkey = IJkey
    subIkey, subJkey = subloclkey
    # Generate base index sets for both sides
    Iset = kronecker(IJset, Ikey, subIkey, localdims[subIkey])
    Jset = kronecker(IJset, Jkey, subJkey, localdims[subJkey])

    # Combine with extra indices if available
    Icombined = union(Iset, get(extraIJset, Ikey, MultiIndex[]))
    Jcombined = union(Jset, get(extraIJset, Jkey, MultiIndex[]))

    return (Icombined, Jcombined)
end

function kronecker(
    IJset::Dict{SubTreeVertex,Vector{MultiIndex}},
    subregions::SubTreeVertex,  # original subregions order
    site::Int,          # direct connected site
    localdim::Int,
)
    site_index = findfirst(==(site), subregions)
    filtered_subregions = filter(x -> x ≠ Set([site]), subregions)
    pivotset = IJset[subregions]
    return MultiIndex[
        [is[1:site_index-1]..., j, is[site_index+1:end]...] for is in pivotset,
        j = 1:localdim
    ][:]
end
