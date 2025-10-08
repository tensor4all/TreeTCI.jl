using ITensors
using ITensorNetworks
const ITN = ITensorNetworks

function entanglements_entropy(s)
    s = diag(s)
    s2 = s.^2
    s2 = s2 / sum(s2)
    s2 = s2[s2 .> 0.0]
    return -sum(s2 .* log.(s2))
end

function ttnopt(
    ttn::TreeTensorNetwork,
    nsweeps::Int=50;
    ortho_vertex::Int=1,
    max_degree::Int = 1,
    T0::Float64 = 0.0,
)
    ttn = convert_ITensorNetwork(ttn, ortho_vertex)
    normalize!(ttn)
    neighbor_vertices = neighbors(ttn, ortho_vertex)
    next_vertex = first(neighbor_vertices)
    origin_edge = NamedEdge(min(ortho_vertex, next_vertex) => max(ortho_vertex, next_vertex))
    center_edge = origin_edge

    flag_indices = [tags(first(ITN.linkinds(ttn, e))) for e in edges(ttn)]
    origin_flag_index = tags(first(ITN.linkinds(ttn, origin_edge)))

    edge_list = [[src(e), dst(e)] for e in edges(ttn)]

    original_entanglements = Dict()
    final_entanglements = Dict()

    for sweep = 0:nsweeps
        flags = Dict(flag_indices[i] => 0 for i in 1:length(flag_indices))
        final_entanglements = Dict()
        while true
            g = ttn.tensornetwork.data_graph.underlying_graph
            p, q = separatevertices(g, center_edge)
            linkind = first(ITN.linkinds(ttn, center_edge))
            maxbonddim = dim(linkind)
            tag = tags(linkind)
            siteindices = [ITN.siteinds(ttn, p); ITN.siteinds(ttn, q)]

            ψ = ITensors.contract(ttn[p], ttn[q])
            if sweep == 0
                leftinds = filter(ind -> ind != ITN.siteinds(ttn, p), inds(ttn[p]))
                _, s, _ = ITensors.svd(ψ, leftinds; cutoff = 0.0)
                ee = entanglements_entropy(s)
                original_entanglements[src(center_edge), dst(center_edge)] = ee
            else
                entanglements, leftinds_list = propose_structure(ψ, siteindices, max_degree)
                index = decide_structure(entanglements, sweep, nsweeps; T0 = T0)
                leftinds = leftinds_list[index]
                ee = entanglements[index]
                final_entanglements[src(center_edge), dst(center_edge)] = ee
            end

            u, s, v = ITensors.svd(ψ, leftinds; maxdim = maxbonddim, lefttags = tag, righttags = tag)
            ttn[p] = u
            ttn[q] = v * s

            # update next center edge
            g = ttn.tensornetwork.data_graph.underlying_graph

            candidates = candidateedges(g, center_edge)
            candidates = [e for e in candidates if flags[tags(first(ITN.linkinds(ttn, e)))] == 0]

            # If candidates is empty, exit while loop
            if isempty(candidates)
                break
            end

            same_index_edge = first([e for e in edges(ttn) if tags(first(ITN.linkinds(ttn, e))) == origin_flag_index])
            distances = distanceedges(g, same_index_edge)
            max_distance = maximum(distances[e] for e in candidates)
            candidates = filter(e -> distances[e] == max_distance, candidates)
            center_edge_ = first(candidates)
            prev_vertex, next_vertex = center_edge_ in adjacentedges(g, p) ? (q, p) : (p, q) #
            incomings = [first(ITN.linkinds(ttn, edge)) for edge in adjacentedges(g, prev_vertex) if edge != center_edge]
            center_flag_index = tags(first(ITN.linkinds(ttn, center_edge)))

            if all(flags[tags(e)] == 1 for e in incomings) && center_flag_index != origin_flag_index
                flags[center_flag_index] = 1
            end

            if next_vertex == p
                ttn[next_vertex] = u * s
                ttn[prev_vertex] = v
            end
            center_edge = center_edge_

        end

        new_edge_list = [[src(e), dst(e)] for e in edges(ttn)]
        if sweep > 1 && Set(edge_list) == Set(new_edge_list)
            break
        end
        edge_list = new_edge_list
    end
    return ttn.tensornetwork.data_graph.underlying_graph, original_entanglements, final_entanglements
end

function propose_structure(ψ, siteindices, max_degree::Int)
    s1_ind = siteindices[1]
    s2_ind = siteindices[2]

    remain_inds = filter(ind -> ind != s1_ind && ind != s2_ind, inds(ψ))
    n = length(remain_inds)

    entanglements = Float64[]
    leftinds_list = []
    for k = 0:n
        if k <= max_degree && (n-k) <= max_degree
            for left in combinations(remain_inds, k)
                leftinds = [s1_ind; left]
                _, s, _ = svd(ψ, leftinds, cutoff = 0.0)
                ee = entanglements_entropy(s)
                push!(entanglements, ee)
                push!(leftinds_list, leftinds)
            end
        end
    end
    return entanglements, leftinds_list
end

function decide_structure(entanglements, nowsweep, nsweeps; T0::Float64 = 0.0)
    if T0 > 0.0
        T = T0 * nowsweep / nsweeps
        p = exp.(-entanglements / T)
        p = p / sum(p)
        index = sample(1:length(entanglements), Weights(p))
    else
        index = argmin(entanglements)
    end
    return index
end

function convert_ITensorNetwork(ttn::TreeTensorNetwork, ortho_vertex::Int=1)
    g = ttn.tensornetwork.data_graph.underlying_graph
    siteinds = []
    edge_inds = Dict{NamedEdge, ITensors.Index}()
    itensors = ITensors.ITensor[]
    for v in vertices(g)
        tns = ttn.tensornetwork.data_graph[v]
        tns_data = tns.data
        tns_inds = tns.indices
        # Add site index
        site = ITensors.Index(tns_inds[1].dim; tags="s$v")
        push!(siteinds, site)
        # Add edge indices
        inds = []
        for ind in tns_inds[2:end]
            tag = ind.name
            parts = split(tag, "=>")
            src, dst = parse(Int, parts[1]), parse(Int, parts[2])
            e = NamedEdge(src, dst)
            # Get edge ID from mapping
            edge_id = findfirst(edge -> edge == e, collect(edges(g)))
            ind = get!(edge_inds, NamedEdge(src, dst), ITensors.Index(ind.dim; tags="e$edge_id"))
            push!(inds, ind)
        end
        # Add itensor
        itensor = ITensors.ITensor(tns_data, [site; inds])
        push!(itensors, itensor)
    end

    # ortho normalize
    state = namedgraph_dijkstra_shortest_paths(g, ortho_vertex)
    distances = state.dists
    max_distance = maximum(distances[v] for v in vertices(g))
    for d = max_distance:-1:1
        children = filter(v -> distances[v] == d, vertices(g))
        for child in children
            parent = state.parents[child]
            ψ = ITensors.contract(itensors[parent], itensors[child])
            virtualindex = commonind(itensors[parent], itensors[child])
            child_inds = inds(itensors[child])
            left_inds = filter(i -> i != virtualindex, child_inds)
            u, s, v = svd(ψ, left_inds, maxdim = dim(virtualindex); lefttags=tags(virtualindex), righttags=tags(virtualindex))
            v = v * s
            itensors[child] = u
            itensors[parent] = v
        end
    end

    ttn = ITN.ITensorNetwork(itensors)
    ttn = ITN.TreeTensorNetwork(ttn, ortho_region=vertices(ttn)[ortho_vertex])
    return ttn
end