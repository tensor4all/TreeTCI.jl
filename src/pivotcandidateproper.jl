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
    InIJkeys::Pair{Vector{SubTreeVertex},Vector{SubTreeVertex}},
    extraIJset::Dict{SubTreeVertex,Vector{MultiIndex}},
    regionbonds::Dict{Pair{SubTreeVertex,SubTreeVertex},NamedEdge},
    localdims::Vector{Int},
    bond::Pair{SubTreeVertex,SubTreeVertex},
    IJkey::Pair{SubTreeVertex,SubTreeVertex},
    subloclkey::Pair{Int,Int},
)
    Ikey, Jkey = IJkey
    subIkey, subJkey = subloclkey
    InIkey, InJkey = InIJkeys

    # Generate base index sets for both sides
    Iset = kronecker(IJset, Ikey, InIkey, subIkey, localdims[subIkey])
    Jset = kronecker(IJset, Jkey, InJkey, subJkey, localdims[subJkey])

    # Combine with extra indices if available
    Icombined = union(Iset, extraIJset[Ikey])
    Jcombined = union(Jset, extraIJset[Jkey])
    return (Icombined, Jcombined)
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
