"""
Abstract type for structure proposal methods
"""
abstract type AbstractNewStructureProposer end

struct NewStructureLocalSwap <: AbstractNewStructureProposer end

struct NewStructureGlobalSwap <: AbstractNewStructureProposer end

function generate_new_structure(
    ::NewStructureLocalSwap,
    tci::SimpleTCI{ValueType},
    edge::NamedEdge,
) where {ValueType}
    edge in edges(tci.g) || error("Edge $edge not in graph")
    vs = src(edge) => dst(edge)
    return swap_2site(tci.g, vs)
end

function generate_new_structure(
    ::NewStructureGlobalSwap,
    tci::SimpleTCI{ValueType},
    vs::Pair{Int, Int},
) where {ValueType}
    first(vs) in vertices(tci.g) || error("Vertex $first(vs) not in graph")
    last(vs) in vertices(tci.g) || error("Vertex $last(vs) not in graph")
    return swap_2site(tci.g, vs)
end
