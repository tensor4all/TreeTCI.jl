"""
Abstract type for pivot candidate generation strategies
"""
abstract type Sweep2sitePathProper end

"""
Default strategy that uses kronecker product and union with extra indices
"""
struct DefaultSweep2sitePathProper <: Sweep2sitePathProper end

"""
Default strategy that runs through within all indices of site tensor according to the bond and connect them with IJSet from neighbors
"""
function generate_sweep2site_path(
    ::DefaultSweep2sitePathProper,
    tci::SimpleTCI{ValueType};
    origin_bond = undef,

) where {ValueType}
    bond_path = Vector{Pair{SubTreeVertex, SubTreeVertex}}()

    n = length(tci.localdims) # TODO: Implement for AbstractTreeTensorNetwork

    # assigne the uuid for each bond
    invariantbondids = Dict([i => key for (i, key) in enumerate(keys(tci.regionbonds))])
    reverse_invariantbondids =
        Dict([key => i for (i, key) in enumerate(keys(tci.regionbonds))])

    # choose the center bond id.
    d = n
    origin_id = undef
    for (id, key) in invariantbondids
        e = tci.regionbonds[key]
        p, q = separatevertices(tci.g, e)
        Iset = length(subtreevertices(tci.g, p => q))
        Jset = length(subtreevertices(tci.g, q => p))
        d_tmp = abs(Iset - Jset)
        if d_tmp < d
            d = d_tmp
            origin_id = id
        end
    end
    center_id = origin_id

    if origin_bond != undef
        origin_id = reverse_invariantbondids[origin_bond]
        center_id = origin_id
    end

    # Init flags
    flags = Dict(keys(invariantbondids) .=> 0)

    while true
        distances = bonddistances(tci.g, tci.regionbonds, invariantbondids[origin_id])

        candidates =
            bondinfocandidates(tci.g, tci.regionbonds, invariantbondids[center_id])
        candidates = filter(
            ((c, parent_child),) -> flags[reverse_invariantbondids[c]] == 0,
            candidates,
        )

        # If candidates is empty, exit while loop
        if isempty(candidates)
            break
        end

        max_distance = maximum(distances[c] for (c, parent_child) in candidates)
        candidates =
            filter(((c, parent_child),) -> distances[c] == max_distance, candidates)

        center_info = first(candidates)

        incomings = bondcandidates(tci.g, last(center_info), tci.regionbonds)
        # Update flags. However, the center bond is not applied.
        if all(flags[reverse_invariantbondids[c]] == 1 for c in incomings) &&
            center_id != origin_id
            flags[center_id] = 1
        end

        # pivot candidates
        center_bond = first(center_info)
        center_id = reverse_invariantbondids[center_bond]

        push!(bond_path, center_bond)
    end

    return bond_path, invariantbondids[origin_id]
end
