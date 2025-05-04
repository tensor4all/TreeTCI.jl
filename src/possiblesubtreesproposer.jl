using Combinatorics

abstract type AbstractPossibleSubtreesProposer end

"""
Default strategy that returns the all possible structures such that maximum degree is below a given integer threshold.
"""
struct DefaultPossibleSubtreesProposer <: AbstractPossibleSubtreesProposer end

"""
Structural search strategy that returns the all possible structures such that maximum degree is below a given integer threshold.
"""
struct StructuralSearchPossibleSubtreesProposer <: AbstractPossibleSubtreesProposer end


function generate_possible_subtrees(
    ::DefaultPossibleSubtreesProposer,
    tci::SimpleTCI{ValueType},
    sites::Vector{Int},
) where {ValueType}
    vp, vq = sites
    Ikey = subtreevertices(tci.g, vq => vp)
    Jkey = subtreevertices(tci.g, vp => vq)

    # the original structure
    neighborsI = filter(!=(vq), neighbors(tci.g, vp))
    neighborsJ = filter(!=(vp), neighbors(tci.g, vq))
    I_subtree = map(v -> subtreevertices(tci.g, vp => v), neighborsI)
    J_subtree = map(v -> subtreevertices(tci.g, vq => v), neighborsJ)

    return [I_subtree => J_subtree]
end

function generate_possible_subtrees(
    ::StructuralSearchPossibleSubtreesProposer,
    tci::SimpleTCI{ValueType},
    sites::Vector{Int},
    max_degree::Int,
) where {ValueType}
    vp, vq = sites
    subI = neighbors(tci.g, vp)
    subJ = neighbors(tci.g, vq)
    subI = filter(!=(vq), subI)
    subJ = filter(!=(vp), subJ)

    children_all = vcat(subI, subJ)
    n            = length(children_all)
    child_limit  = max_degree - 1

    splits = Vector{Pair{Vector{SubTreeVertex},Vector{SubTreeVertex}}}()

    for k in 0:n
        if k ≤ child_limit && n - k ≤ child_limit
            for left in combinations(children_all, k)
                left_vec = collect(left)
                right_vec = sort!(setdiff(children_all, left_vec))
                vertex_parents = Dict(v => vp in neighbors(tci.g, v) ? vp : vq for v in children_all)

                left_subtree = map(v -> subtreevertices(tci.g, vertex_parents[v] => v), left_vec)
                right_subtree = map(v -> subtreevertices(tci.g, vertex_parents[v] => v), right_vec)
                push!(splits, left_subtree => right_subtree)
            end
        end
    end
    return splits
end




