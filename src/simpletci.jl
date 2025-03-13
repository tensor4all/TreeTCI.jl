MultiIndex = Vector{Int}
SubTreeVertex = Vector{Int}

mutable struct SimpleTCI{ValueType}
    IJset::Dict{SubTreeVertex,Vector{MultiIndex}}
    localdims::Vector{Int}
    g::NamedGraph
    #"Error estimate per bond by 2site sweep."
    bonderrors::Dict{NamedEdge,Float64} # key is the bond id
    # "Error estimate for backtruncation of bonds."
    pivoterrors::Vector{Float64} # key is the bond id
    #"Maximum sample for error normalization."
    maxsamplevalue::Float64
    IJset_history::Vector{Dict{SubTreeVertex,Vector{MultiIndex}}}

    function SimpleTCI{ValueType}(localdims::Vector{Int}, g::NamedGraph) where {ValueType}
        length(localdims) > 1 || error("localdims should have at least 2 elements!")
        n = length(localdims)

        # assign the key for each bond
        bonderrors = Dict(e => 0.0 for e in edges(g))

        !Graphs.is_cyclic(g) ||
            error("TreeTensorNetwork is not supported for loopy tensor network.")

        new{ValueType}(
            Dict{SubTreeVertex,Vector{MultiIndex}}(),               # IJset
            localdims,
            g,
            bonderrors,
            Float64[],
            0.0,                                                   # maxsamplevalue
            Vector{Dict{SubTreeVertex,Vector{MultiIndex}}}(),       # IJset_history
        )
    end
end

function SimpleTCI{ValueType}(
    func::F,
    localdims::Vector{Int},
    g::NamedGraph,
    initialpivots::Vector{MultiIndex} = [ones(Int, length(localdims))],
) where {F,ValueType}
    tci = SimpleTCI{ValueType}(localdims, g)
    addglobalpivots!(tci, initialpivots)
    tci.maxsamplevalue = maximum(abs, (func(x) for x in initialpivots))
    abs(tci.maxsamplevalue) > 0 ||
        error("The function should not be zero at the initial pivot.")
    return tci
end

@doc """
 Add global pivots to index sets
 """
function addglobalpivots!(
    tci::SimpleTCI{ValueType},
    pivots::Vector{MultiIndex},
) where {ValueType}
    if any(length(tci.localdims) .!= length.(pivots)) # AbstructTreeTensorNetworkをから引き継ぎlength(tci)ができると良い
        throw(DimensionMismatch("Please specify a pivot as one index per leg of the TTN."))
    end
    for pivot in pivots
        for e in edges(tci.g)
            p, q = separatevertices(tci.g, e)
            Iset_key = subtreevertices(tci.g, p => q)
            Jset_key = subtreevertices(tci.g, q => p)

            if !haskey(tci.IJset, Iset_key)
                tci.IJset[Iset_key] = Vector{MultiIndex}()
            end
            if !haskey(tci.IJset, Jset_key)
                tci.IJset[Jset_key] = Vector{MultiIndex}()
            end
            pushunique!(tci.IJset[Iset_key], [pivot[i] for i in Iset_key])
            pushunique!(tci.IJset[Jset_key], [pivot[j] for j in Jset_key])
        end
    end

    tci.IJset[[i for i = 1:length(tci.localdims)]] = Int[]

    nothing
end
