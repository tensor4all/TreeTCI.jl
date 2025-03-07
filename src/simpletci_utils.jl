MultiIndex = Vector{Int}
SubTreeVertex = Vector{Int}

function separatevertices(g::NamedGraph, edge::NamedEdge)
    has_edge(g, edge) || error("The edge is not in the graph.")
    p, q = src(edge), dst(edge)
    p, q = p < q ? (p, q) : (q, p)
    return p, q
end

function adjacentbonds(
    g::NamedGraph,
    vertex::Int,
    regionbonds::Dict{Pair{SubTreeVertex,SubTreeVertex},NamedEdge},
)
    bonds = []
    for (key, edge) in regionbonds
        if src(edge) == vertex || dst(edge) == vertex
            push!(bonds, key)
        end
    end
    return bonds
end

function subtreevertices(
    g::NamedGraph,
    parent_children::Pair{Int,<:Union{Int,Vector{Int}}},
)::Vector{Int}
    parent, children = first(parent_children), last(parent_children)
    if children isa Int
        children = [children]
    end
    grandchildren = []
    for child in children
        candidates = outneighbors(g, child)
        candidates = [cand for cand in candidates if cand != parent]
        append!(grandchildren, subtreevertices(g, child => candidates))
        append!(grandchildren, [child])
    end
    sort!(grandchildren)
    return grandchildren
end

function subregionvertices(g::NamedGraph, edge::NamedEdge)
    p, q = separatevertices(g, edge)
    Iregions = subtreevertices(g, q => p)
    Jregions = subtreevertices(g, p => q)
    return Pair(Iregions, Jregions)
end

function bonddistances(
    g::NamedGraph,
    regionbonds::Dict{Pair{SubTreeVertex,SubTreeVertex},NamedEdge},
    origin_bond::Pair{SubTreeVertex,SubTreeVertex},
)::Dict{Pair{SubTreeVertex,SubTreeVertex},Int}
    p, q = separatevertices(g, regionbonds[origin_bond])
    distances = Dict{Pair{SubTreeVertex,SubTreeVertex},Int}()
    distances[origin_bond] = 0
    distances = distanceBFS(g, p => q, distances, regionbonds)
    distances = distanceBFS(g, q => p, distances, regionbonds)
    return distances
end

function distanceBFS(
    g::NamedGraph,
    parent_children::Pair{Int,<:Union{Int,Vector{Int}}},
    distances::Dict{Pair{SubTreeVertex,SubTreeVertex},Int},
    regionbonds::Dict{Pair{SubTreeVertex,SubTreeVertex},NamedEdge},
)::Dict{Pair{SubTreeVertex,SubTreeVertex},Int}
    parent, children = first(parent_children), last(parent_children)
    for child in children
        parent_key = ""
        for (key, item) in regionbonds
            if (src(item) == parent && dst(item) == child) ||
               (src(item) == child && dst(item) == parent)
                parent_key = key
            end
        end

        candidates = outneighbors(g, child)
        candidates = [cand for cand in candidates if cand != parent]
        for cand in candidates
            for (key, item) in regionbonds
                if (src(item) == child && dst(item) == cand) ||
                   (src(item) == cand && dst(item) == child)
                    distances[key] = distances[parent_key] + 1
                    break
                end
            end
        end
        distances =
            merge!(distances, distanceBFS(g, child => candidates, distances, regionbonds))
    end
    return distances
end

function bondinfocandidates(
    g::NamedGraph,
    regionbonds::Dict{Pair{SubTreeVertex,SubTreeVertex},NamedEdge},
    center_bond::Pair{SubTreeVertex,SubTreeVertex},
)
    p, q = separatevertices(g, regionbonds[center_bond])
    candidates = vcat(
        [(c, q => p) for c in bondcandidates(g, p => q, regionbonds)],
        [(c, p => q) for c in bondcandidates(g, q => p, regionbonds)],
    )
    return candidates
end

function bondcandidates(
    g::NamedGraph,
    parent_child::Pair{Int,Int},
    regionbonds::Dict{Pair{SubTreeVertex,SubTreeVertex},NamedEdge},
)::Vector{Pair{SubTreeVertex,SubTreeVertex}}
    parent, child = first(parent_child), last(parent_child)
    candidates = []
    neighbors = outneighbors(g, child)
    neighbors = [cand for cand in neighbors if cand != parent]
    for cand in neighbors
        for (key, item) in regionbonds
            if (src(item) == child && dst(item) == cand) ||
               (src(item) == cand && dst(item) == child)
                push!(candidates, key)
                break
            end
        end
    end
    return candidates
end

function bondtokey(
    g::NamedGraph,
    vf::Int,
    bonds,
    regionbonds::Dict{Pair{SubTreeVertex,SubTreeVertex},NamedEdge},
)
    keys = SubTreeVertex[]
    for bond in bonds
        p, q = separatevertices(g, regionbonds[bond])
        if p == vf
            push!(keys, subtreevertices(g, p => q))
        elseif q == vf
            push!(keys, subtreevertices(g, q => p))
        end
    end
    return keys
end
