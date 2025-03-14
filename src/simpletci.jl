MultiIndex = Vector{Int}
SubTreeVertex = Vector{Int}

@doc """
    SimpleTCI{ValueType}

Tree tensor cross interpolation (TCI) for tree tensor networks.

# Fields
- `IJset::Dict{SubTreeVertex,Vector{MultiIndex}}`: Pivots sets for each subtrees
- `localdims::Vector{Int}`: Local dimensions for each vertex tensor
- `g::NamedGraph`: Tree graph structure
- `bonderrors::Dict{NamedEdge,Float64}`: Error estimate per bond by 2-site sweep
- `pivoterrors::Vector{Float64}`: Error estimate for backtruncation of bonds
- `maxsamplevalue::Float64`: Maximum sample value for error normalization
- `IJset_history::Vector{Dict{SubTreeVertex,Vector{MultiIndex}}}`: History of pivots sets for each sweep

# Example
```julia
# Create a simple tree graph
g = NamedGraph([1, 2, 3])
add_edge!(g, 1 => 2)
add_edge!(g, 2 => 3)

# Define local dimensions
localdims = [2, 2, 2]

# Create a SimpleTCI instance
tci = SimpleTCI{Float64}(localdims, g)

# Add initial pivots
addglobalpivots!(tci, [[1,1,1], [2,1,1]])
```
"""
mutable struct SimpleTCI{ValueType}
    IJset::Dict{SubTreeVertex,Vector{MultiIndex}}
    localdims::Vector{Int}
    g::NamedGraph
    bonderrors::Dict{NamedEdge,Float64}
    pivoterrors::Vector{Float64}
    maxsamplevalue::Float64
    IJset_history::Vector{Dict{SubTreeVertex,Vector{MultiIndex}}}

    function SimpleTCI{ValueType}(localdims::Vector{Int}, g::NamedGraph) where {ValueType}
        n = length(localdims)
        n > 1 || error("localdims should have at least 2 elements!")
        n == length(vertices(g)) || error(
            "The number of vertices in the graph must be equal to the length of localdims.",
        )
        !Graphs.is_cyclic(g) ||
            error("SimpleTCI is not supported for loopy tensor network.")

        # assign the key for each bond
        bonderrors = Dict(e => 0.0 for e in edges(g))

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

"""
 Initialize a SimpleTCI instance with a function, local dimensions, and graph.
 The initial grobal pivots are set to ones(Int, length(localdims)).
"""
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

"""
 Add global pivots to IJset.
"""
function addglobalpivots!(
    tci::SimpleTCI{ValueType},
    pivots::Vector{MultiIndex},
) where {ValueType}
    if any(length(tci.localdims) .!= length.(pivots))
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
            pushunique!(tci.IJset[Iset_key], MultiIndex([pivot[i] for i in Iset_key]))
            pushunique!(tci.IJset[Jset_key], MultiIndex([pivot[j] for j in Jset_key]))
        end
    end

    tci.IJset[[i for i = 1:length(tci.localdims)]] = MultiIndex[]

    nothing
end

function pushunique!(collection, item)
    if !(item in collection)
        push!(collection, item)
    end
end

function pushunique!(collection, items...)
    for item in items
        pushunique!(collection, item)
    end
end
