using TreeTCI: TreeTensorNetwork
using SimpleTensorNetworks: permute, IndexedArray, hascommondindices
using LinearAlgebra
using ITensors

function ttnopt(
    ttn::TreeTensorNetwork{V};
    maxbonddim::Int = typemax(Int),
    tolerance::Float64 = 0.0,
    max_degree::Int = 2,
    nsweeps::Int = 10,
    T0::Float64 = 0.0,
    
) where {V}

    tn = ttn.tensornetwork.data_graph
    tn_g = tn.underlying_graph
    center_vertex = ttn.center_vertex
    neighbor_vertices = neighbors(tn_g, center_vertex)
    origin_edge = NamedEdge(min(center_vertex, first(neighbor_vertices)) => max(center_vertex, first(neighbor_vertices)))
    
    id2edge = collect(edges(tn_g))
    edge2id = Dict{NamedEdge,Int}(e => i for (i,e) in enumerate(id2edge))
    center_edge = origin_edge
    origin_edge_id = edge2id[origin_edge]

    for i = 1:nsweeps
        @show i
        # Init flags
        flags = Dict(k => 0 for k in 1:length(id2edge))
        while true
            ttn, center_edge, id2edge, edge2id, flags, finish = optimize_structure(ttn, center_edge, id2edge, edge2id, flags, origin_edge_id, i, nsweeps; max_degree = max_degree, T0 = T0, maxbonddim = maxbonddim, tolerance = tolerance)
            @show center_edge
            if finish
                @show "converged"
                break
            end
        end
    end


    return 0
end

function optimize_structure(
    ttn,
    center_edge,
    id2edge,
    edge2id,
    flags,
    origin_edge_id,
    nowstep::Int,
    nsweeps::Int;
    max_degree::Int = 1,
    T0::Float64 = 0.0,
    maxbonddim::Int = typemax(Int),
    tolerance::Float64 = 0.0,
)
    tn = ttn.tensornetwork.data_graph
    tn_g = tn.underlying_graph
    p, q = separatevertices(tn_g, center_edge)
    subI = filter(x->x!=q, neighbors(tn_g, p))
    subJ = filter(x->x!=p, neighbors(tn_g, q))
    children = vcat(subI, subJ)
    ψ = contract(tn[p], tn[q])

    n = length(children)
    limit = max_degree

    ψ_inds = ψ.indices
    s1 = ψ_inds[1]
    s2 = ψ_inds[2]
    other_inds = ψ_inds[3:end]

    entanglements = Float64[]
    virtual_indices_pairs = []
    gs = []
    id2edges = []

    for k in 0:n
        if k <= limit && (n-k) <= limit
            for left in combinations(children, k)
                leftset = Set(left)

                # split virtual bonds into left and right
                left_bonds = other_inds[1:k]
                right_bonds = other_inds[(k+1):end]

                # left: s1 + left_bonds, right: s2 + right_bonds
                left_indices = [s1; left_bonds]
                right_indices = [s2; right_bonds]

                ψ_permuted = permute(ψ, [left_indices; right_indices])

                # size of left and right
                left_size = prod([ind.dim for ind in left_indices])
                right_size = prod([ind.dim for ind in right_indices])

                # reshape and SVD
                reshaped = reshape(ψ_permuted.data, left_size, right_size)
                _, S, _ = svd(reshaped)

                S2 = S.^2
                S2 = S2 / sum(S2)
                S2 = S2[S2 .> 0.0]
                ee = -sum(S2 .* log.(S2))

                # graphs
                push!(entanglements, ee)
                push!(virtual_indices_pairs, (left_indices, right_indices))

                g_new = deepcopy(tn_g)
                id2edge_new = deepcopy(id2edge)
                rem_edge!(g_new, center_edge)

                for v in children
                    # remove the old parent
                    old_parent = (v in subI ? p : q)

                    e_old = old_parent < v ? NamedEdge(old_parent=>v) : NamedEdge(v=>old_parent)
                    e_old_id = edge2id[e_old]

                    rem_edge!(g_new, e_old)
                    # connect the new parent
                    new_parent = (v in leftset ? p : q)
                    e_new = new_parent < v ? NamedEdge(new_parent=>v) : NamedEdge(v=>new_parent)

                    id2edge_new[e_old_id] = e_new
                    add_edge!(g_new, e_new)
                end
                add_edge!(g_new, center_edge)

                push!(gs, g_new)
                push!(id2edges, id2edge_new)
            end
        end
    end

    index = propose_structure(entanglements, nowstep, nsweeps; T0 = T0)

    edge2id = Dict(e => i for (i, e) in enumerate(id2edge))

    tn_g = gs[index]
    id2edge = id2edges[index]
    edge2id = Dict(e => i for (i, e) in enumerate(id2edge))

    # update next center edge
    candidates = candidateedges(tn_g, center_edge)
    candidates = [e for e in candidates if flags[edge2id[e]] == 0]

    # If candidates is empty, exit while loop
    if isempty(candidates)
        return tn, tn_g, center_edge, id2edge, edge2id, flags, true
    end

    distances = distanceedges(tn_g, id2edge[origin_edge_id])
    max_distance = maximum(distances[e] for e in candidates)
    candidates = filter(e -> distances[e] == max_distance, candidates)

    center_edge_ = first(candidates)
    center_edge_id = edge2id[center_edge_]

    p, q = separatevertices(tn_g, center_edge)
    v = center_edge_ in adjacentedges(tn_g, p) ? q : p #
    incomings = [edge for edge in adjacentedges(tn_g, v) if edge != center_edge]

    # Update flags - ID management
    if !isempty(incomings) && all(flags[edge2id[e]] == 1 for e in incomings) && center_edge_id != origin_edge_id
        flags[center_edge_id] = 1
    end

    U, V = update_tn(ψ, v, first(virtual_indices_pairs[index]), last(virtual_indices_pairs[index]); maxbonddim = maxbonddim, tolerance = tolerance)
    ttn = TreeTensorNetwork(tn_g, sitetensors, ttn.center_vertex)

    # update center edge
    center_edge = center_edge_
    
    return ttn, center_edge, id2edge, edge2id, flags, false
end

function propose_structure(entanglements::Vector{Float64}, nowstep::Int, nsteps::Int; T0::Float64 = 0.0)
    index = 0
    if T0 > 0.0
        T = T0 * (nsteps - nowstep) / nsteps
        p = exp.(-entanglements / T)
        p = p / sum(p)
        index = sample(1:length(entanglements), Weights(p))
    else
        index = argmin(entanglements)
    end
    return index
end

function update_tn(
    ψ,
    next_vertex,
    left_indices,
    right_indices;
    maxbonddim::Int = typemax(Int),
    tolerance::Float64 = 0.0,
)
    ψ_permuted = permute(ψ, [left_indices; right_indices])

    # TODO: SimpleTensorNetworks SVD
    # ここで left/right 側の結合次元を出してSVD → u,s,v を返す処理へ発展
    left_inds = [ITensors.Index(left_indices[i].dim) for i in 1:length(left_indices)]
    right_inds = [ITensors.Index(right_indices[i].dim) for i in 1:length(right_indices)]
    ψ_permuted = ITensors.ITensor(ψ_permuted.data, vcat(left_inds, right_inds)...)
    u_resolve = left_indices[1].name == "s$(next_vertex)"
    v_resolve = right_indices[1].name == "s$(next_vertex)"
    u, s, v = svd(ψ_permuted, left_inds; maxdim = maxbonddim, cutoff = tolerance)
    d = first(size(s))
    if u_resolve
        u = u * s
    elseif v_resolve
        v = v * s
    end

    u = Array(u, ITensors.inds(u)...)
    v = Array(v, ITensors.inds(v)...)

    return u, v
end
