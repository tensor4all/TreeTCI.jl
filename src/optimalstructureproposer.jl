abstract type AbstractOptimalStructureProposer end

"""
Default strategy that dose not perform any structural search.
"""
struct DefaultOptimalStructureProposer <: AbstractOptimalStructureProposer end

"""
Entanglement strategy that returns the structure with minimum entanglement.
"""
struct EntanglementOptimalStructureProposer <: AbstractOptimalStructureProposer end

function generate_optimal_structure(
    ::EntanglementOptimalStructureProposer,
    tci::SimpleTCI{ValueType},
    singularvalues_vector::Vector{Vector{ValueType}},
) where {ValueType}
    minEE::Float64 = Inf
    index::Int = 0
    for (i, S) in enumerate(singularvalues_vector)
        EE = sum(S .^ 2 .* log.(S .^ 2))
        if EE < minEE
            minEE = EE
            index = i
        end
    end
    return index
end
