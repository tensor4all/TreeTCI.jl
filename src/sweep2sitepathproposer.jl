"""
Abstract type for pivot candidate generation strategies
"""
abstract type AbstractSweep2sitePathProposer end

"""
Default strategy
"""
struct DefaultSweep2sitePathProposer <: AbstractSweep2sitePathProposer end

"""
Random strategy
"""
struct RandomSweep2sitePathProposer <: AbstractSweep2sitePathProposer end

"""
LocalAdjacent strategy
"""
struct LocalAdjacentSweep2sitePathProposer <: AbstractSweep2sitePathProposer end

"""
Default strategy that return the sequence path defined by the edges(g)
"""
function generate_sweep2site_path(
    ::DefaultSweep2sitePathProposer,
    tci::SimpleTCI{ValueType},
) where {ValueType}
    return collect(edges(tci.g))
end

"""
Random strategy that returns a random sequence of edges
"""
function generate_sweep2site_path(
    ::RandomSweep2sitePathProposer,
    tci::SimpleTCI{ValueType},
) where {ValueType}
    return shuffle(collect(edges(tci.g)))
end

"""
LocalAdjacent strategy that runs through within all indices of site tensor according to the bond and connect them with IJSet from neighbors
"""
function generate_sweep2site_path(
    ::LocalAdjacentSweep2sitePathProposer,
    tci::SimpleTCI{ValueType};
    origin_edge = undef,
) where {ValueType}
    edge_path = Vector{NamedEdge}()

    n = length(vertices(tci.g))

    # choose the center bond id.
    if origin_edge == undef
        d = n
        for e in edges(tci.g)
            p, q = separatevertices(tci.g, e)
            Iset = length(subtreevertices(tci.g, p => q))
            Jset = length(subtreevertices(tci.g, q => p))
            d_tmp = abs(Iset - Jset)
            if d_tmp < d
                d = d_tmp
                origin_edge = e
            end
        end
    end
    center_edge = origin_edge

    # Init flags
    flags = Dict(e => 0 for e in edges(tci.g))

    while true

        candidates = candidateedges(tci.g, center_edge)
        candidates = [e for e in candidates if flags[e] == 0]


        # If candidates is empty, exit while loop
        if isempty(candidates)
            break
        end

        distances = distanceedges(tci.g, origin_edge)
        max_distance = maximum(distances[e] for e in candidates)
        candidates = filter(e -> distances[e] == max_distance, candidates)

        center_edge_ = first(candidates)

        p, q = separatevertices(tci.g, center_edge)
        v = center_edge_ in adjacentedges(tci.g, p) ? q : p #
        incomings = [edge for edge in adjacentedges(tci.g, v) if edge != center_edge]

        # Update flags. However, the center bond is not applied.
        if all(flags[e] == 1 for e in incomings) && center_edge != origin_edge
            flags[center_edge] = 1
        end

        # pivot candidates
        center_edge = center_edge_

        push!(edge_path, center_edge)
    end

    return edge_path
end
