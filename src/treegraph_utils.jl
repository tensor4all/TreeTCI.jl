
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
    grandchildren = Int[]
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
    candidates = unique(
        vcat(
            adjacentedges(g, p; combinededges = edge),
            adjacentedges(g, q; combinededges = edge),
        ),
    )
    return candidates
end

function distanceedges(g::NamedGraph, edge::NamedEdge)::Dict{NamedEdge,Int}
    p, q = separatevertices(g, edge)
    distances = Dict{NamedEdge,Int}()

    function compute_distances(root, opposite, state)
        for subvertex in filter(v -> v != root, subtreevertices(g, opposite => root))
            parent = state.parents[subvertex]
            e = NamedEdge(parent, subvertex)
            distances[e ∈ edges(g) ? e : reverse(e)] = state.dists[subvertex]
        end
    end

    state_p = namedgraph_dijkstra_shortest_paths(g, p)
    compute_distances(p, q, state_p)

    state_q = namedgraph_dijkstra_shortest_paths(g, q)
    compute_distances(q, p, state_q)

    distances[edge] = 0
    return distances
end

function swap_2site!(
    g::NamedGraph,
    vs::Pair{Int, Int},
)
    p, q = vs
    p_neighbors = neighbors(g, p)
    q_neighbors = neighbors(g, q)

    rem_vertices!(g, [p, q])

    add_vertex!(g, p)
    add_vertex!(g, q)

    for p_i in p_neighbors
        add_edge!(g, NamedEdge(p_i => q))
    end

    for q_i in q_neighbors
        add_edge!(g,  NamedEdge(q_i => p))
    end

    p_neighbors = neighbors(g, p)
    q_neighbors = neighbors(g, q)
    return g
end

function add_subtree!(
    g::NamedGraph,
    v::Int,
    edge::NamedEdge,
)
    p, q = separatevertices(g, edge)
    p_regions = subtreevertices(g, q => p)
    q_regions = subtreevertices(g, p => q)
    parent = v in p_regions ? q : p
    rem_edge!(g, edge)
    add_edge!(g, NamedEdge(parent => v))
end