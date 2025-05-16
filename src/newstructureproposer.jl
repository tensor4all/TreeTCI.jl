"""
Abstract type for structure proposal methods
"""
abstract type AbstractNewStructureProposer end

struct NewStructureLocalSwap <: AbstractNewStructureProposer end

struct NewStructureGlobalSwap <: AbstractNewStructureProposer end

function generate_new_structure(
    ::NewStructureLocalSwap,
    tci::SimpleTCI{ValueType},
) where {ValueType}
    es = collect(edges(tci.g))
    edge = rand(es)
    vs = src(edge) => dst(edge)
    return swap_2site(tci.g, vs)
end

function generate_new_structure(
    ::NewStructureGlobalSwap,
    tci::SimpleTCI{ValueType},
) where {ValueType}
    vs = collect(vertices(tci.g))
    es = Set(edges(tci.g))

    all_pairs = Set((v1 => v2) for v1 in vs for v2 in vs if v1 < v2)

    es_symmetric = Set(min(src(e), dst(e)) => max(src(e), dst(e)) for e in es)

    candidate_pairs = setdiff(all_pairs, es_symmetric)

    e = rand(candidate_pairs)
    @show e
    return swap_2site(tci.g, first(e) => last(e))
end
