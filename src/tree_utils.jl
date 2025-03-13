
function separatevertices(g::NamedGraph, edge::NamedEdge)
    has_edge(g, edge) || error("The edge is not in the graph.")
    p, q = src(edge), dst(edge)
    p, q = p < q ? (p, q) : (q, p)
    return p, q
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

function adjacentedges(
    g::NamedGraph,
    vertex::Int;
    combinededges::Union{NamedEdge,Vector{NamedEdge}} = Vector{NamedEdge}(),
)::Vector{NamedEdge}
    if combinededges isa NamedEdge
        combinededges = [combinededges]
    end

    adjedges = NamedEdge[]
    for e in edges(g)
        if (src(e) == vertex || dst(e) == vertex) && e ∉ combinededges
            push!(adjedges, e)
        end
    end
    return adjedges
end

function candidateedges(g::NamedGraph, edge::NamedEdge)::Vector{NamedEdge}
    p, q = separatevertices(g, edge)
    candidates =
        adjacentedges(g, p; combinededges = edge) ∪
        adjacentedges(g, q; combinededges = edge)
    return candidates
end

function distanceedges(g::NamedGraph, edge::NamedEdge)::Dict{NamedEdge,Int}
    p, q = separatevertices(g, edge)
    distances = Dict{NamedEdge,Int}()
    distances[edge] = 0
    distances = distanceBFSedge(g, edge, distances)
    return distances
end

function distanceBFSedge(
    g::NamedGraph,
    edge::NamedEdge,
    distances::Dict{NamedEdge,Int},
)::Dict{NamedEdge,Int}

    candidates = candidateedges(g, edge)
    candidates = filter(cand -> cand ∉ keys(distances), candidates)
    for cand in candidates
        distances[cand] = distances[edge] + 1
        distances = merge!(distances, distanceBFSedge(g, cand, distances))
    end
    return distances
end
