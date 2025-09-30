function crossinterpolate_with_3site_swapping(
    ::Type{ValueType},
    f,
    localdims::Union{Vector{Int},NTuple{N,Int}},
    g::NamedGraph,
    nsweeps:: Int = 100,
    origin_edge = nothing,
    initialpivots::Vector{MultiIndex} = [ones(Int, length(localdims))];
    kwargs...,
) where {ValueType,N}

    tci = SimpleTCI{ValueType}(f, localdims, g, initialpivots)
    ranks, errors = optimize!(tci, f; kwargs...)
    n = length(vertices(tci.g))


    g_tmp = deepcopy(tci.g)

    if origin_edge == nothing
        d = n
        for e in edges(tci.g)
            p, q = separatevertices(tci.g, e)
            Iset = length(subtreevertices(tci.g, p => q))
            Jset = length(subtreevertices(tci.g, q => p))
            d_tmp = abs(Iset - Jset)
            if d_tmp < d
                d = d_tmp
                origin_edge = e
            end
        end
    end

    id2edge = collect(edges(tci.g))
    edge2id = Dict{NamedEdge,Int}(e => i for (i,e) in enumerate(id2edge))
    center_edge = origin_edge
    center_edge_id = edge2id[center_edge]
    origin_edge_id = edge2id[origin_edge]
    previous_center_edge_id = center_edge_id

    for i = 1:nsweeps
        # Init flags
        flags = Dict(k => 0 for k in 1:length(id2edge))
        while true
            # update next center edge
            candidates = candidateedges(tci.g, id2edge[previous_center_edge_id])
            candidates = [e for e in candidates if flags[edge2id[e]] == 0]

            # If candidates is empty, exit while loop
            if isempty(candidates)
                break
            end

            distances = distanceedges(tci.g, id2edge[origin_edge_id])
            max_distance = maximum(distances[e] for e in candidates)
            candidates = filter(e -> distances[e] == max_distance, candidates)

            center_edge_ = first(candidates)
            center_edge_id = edge2id[center_edge_]

            p, q = separatevertices(tci.g, id2edge[previous_center_edge_id])
            v = center_edge_ in adjacentedges(tci.g, p) ? q : p #
            incomings = [edge for edge in adjacentedges(tci.g, v) if edge != id2edge[previous_center_edge_id]]

            # Update flags - ID management
            if all(flags[edge2id[e]] == 1 for e in incomings) && center_edge_id != origin_edge_id
                flags[center_edge_id] = 1
            end

            # update center edge
            center_edge = center_edge_

            # Structural search
            tci, id2edge = optimize_with_3site_swapping(tci, f, initialpivots, id2edge[center_edge_id], id2edge[previous_center_edge_id], id2edge, edge2id; kwargs...)
            edge2id = Dict(e => i for (i, e) in enumerate(id2edge))

            previous_center_edge_id = center_edge_id
        end
        if tci.g == g_tmp
            @show "converged"
            break
        end
        g_tmp = deepcopy(tci.g) # update g_tmp
    end
    sitetensors = fillsitetensors(tci, f)

    ranks, errors = optimize!(tci, f; kwargs...)
    return TreeTensorNetwork(tci.g, sitetensors), ranks, errors
end

function crossinterpolate_with_structuralsearch(
    ::Type{ValueType},
    f,
    localdims::Union{Vector{Int},NTuple{N,Int}},
    g::NamedGraph,
    max_degree::Int = 1,
    nsweeps:: Int = 100,
    origin_edge = nothing,
    initialpivots::Vector{MultiIndex} = [ones(Int, length(localdims))];
    kwargs...,
) where {ValueType,N}

    tci = SimpleTCI{ValueType}(f, localdims, g, initialpivots)
    ranks, errors = optimize!(tci, f; kwargs...)
    n = length(vertices(tci.g))

    g_tmp = deepcopy(tci.g)

    if origin_edge == nothing
        d = n
        for e in edges(tci.g)
            p, q = separatevertices(tci.g, e)
            Iset = length(subtreevertices(tci.g, p => q))
            Jset = length(subtreevertices(tci.g, q => p))
            d_tmp = abs(Iset - Jset)
            if d_tmp < d
                d = d_tmp
                origin_edge = e
            end
        end
    end

    id2edge = collect(edges(tci.g))
    edge2id = Dict{NamedEdge,Int}(e => i for (i,e) in enumerate(id2edge))
    center_edge = origin_edge
    origin_edge_id = edge2id[origin_edge]

    for i = 1:nsweeps
        # Init flags
        flags = Dict(k => 0 for k in 1:length(id2edge))
        while true

            # Structural search
            tci, id2edge = optimize_with_localmanipulation(tci, f, initialpivots, center_edge, max_degree, id2edge, edge2id; kwargs...)
            edge2id = Dict(e => i for (i, e) in enumerate(id2edge))

            # update next center edge
            candidates = candidateedges(tci.g, center_edge)
            candidates = [e for e in candidates if flags[edge2id[e]] == 0]

            # If candidates is empty, exit while loop
            if isempty(candidates)
                break
            end

            distances = distanceedges(tci.g, id2edge[origin_edge_id])
            max_distance = maximum(distances[e] for e in candidates)
            candidates = filter(e -> distances[e] == max_distance, candidates)

            center_edge_ = first(candidates)
            center_edge_id = edge2id[center_edge_]

            p, q = separatevertices(tci.g, center_edge)
            v = center_edge_ in adjacentedges(tci.g, p) ? q : p #
            incomings = [edge for edge in adjacentedges(tci.g, v) if edge != center_edge]

            # Update flags - ID management
            if all(flags[edge2id[e]] == 1 for e in incomings) && center_edge_id != origin_edge_id
                flags[center_edge_id] = 1
            end

            # update center edge
            center_edge = center_edge_
        end
        if tci.g == g_tmp
            @show "converged"
            break
        end
        g_tmp = deepcopy(tci.g) # update g_tmp
    end
    sitetensors = fillsitetensors(tci, f)

    ranks, errors = optimize!(tci, f; kwargs...)
    return TreeTensorNetwork(tci.g, sitetensors), ranks, errors
end

# Optimization with local 3-site tensor swapping
function optimize_with_3site_swapping(
    tci::SimpleTCI{ValueType},
    f,
    initialpivots::Vector{MultiIndex},
    center_edge::NamedEdge,
    adjacent_edge::NamedEdge,
    id2edge,
    edge2id;
    kwargs...
) where {ValueType}
    best_err = maximum(values(tci.bonderrors))

    tci_best = tci
    id2edge_best = id2edge


    # center_edgeとadjacent_edgeからp,q,rを抽出（qが両方に接続）
    tmp1, tmp2 = src(center_edge), dst(center_edge)
    tmp3, tmp4 = src(adjacent_edge), dst(adjacent_edge)

    # qが両方のエッジに接続するようにp, q, rを決定
    if tmp1 in (tmp3, tmp4)
        q = tmp1
        p = tmp2
        r = tmp1 == tmp3 ? tmp4 : tmp3
    elseif tmp2 in (tmp3, tmp4)
        q = tmp2
        p = tmp1
        r = tmp2 == tmp3 ? tmp4 : tmp3
    else
        @warn "No common node between center_edge and adjacent_edge"
        return tci, id2edge
    end

    @assert r != p && r != q

    # get the subtree sets that are connected to each node
    p_sub = filter(x -> x ∉ (q, r), neighbors(tci.g, p))
    q_sub = filter(x -> x ∉ (p, r), neighbors(tci.g, q))
    r_sub = filter(x -> x ∉ (p, q), neighbors(tci.g, r))

    for (a, b, c) in permutations((p, q, r))
        g_new = deepcopy(tci.g)
        id2edge_new = deepcopy(id2edge)

        for (v1, v2) in ((p, q), (q, r))
            e_old = NamedEdge(min(v1, v2) => max(v1, v2))
            if has_edge(g_new, e_old)
                e_old_id = edge2id[e_old]
                rem_edge!(g_new, e_old)
            end
        end

        e_ab = NamedEdge(min(a, b) => max(a, b))
        e_bc = NamedEdge(min(b, c) => max(b, c))
        add_edge!(g_new, e_ab)
        add_edge!(g_new, e_bc)

        # 3ノード間エッジのID更新（optimize_with_localmanipulationと同じパターン）
        if haskey(edge2id, NamedEdge(min(p, q) => max(p, q)))
            e_pq_id = edge2id[NamedEdge(min(p, q) => max(p, q))]
            id2edge_new[e_pq_id] = e_ab
        end
        if haskey(edge2id, NamedEdge(min(q, r) => max(q, r)))
            e_qr_id = edge2id[NamedEdge(min(q, r) => max(q, r))]
            id2edge_new[e_qr_id] = e_bc
        end

        # サブグラフの再接続（重複エッジを避ける）
        # 各ノードのサブグラフを新しい親に再接続
        for (old_node, new_node) in [(p, a), (q, b), (r, c)]
            if old_node != new_node  # ノードが変わった場合のみ再接続
                # 元のノードからサブグラフを取得
                old_sub = filter(x -> x ∉ (p, q, r), neighbors(tci.g, old_node))

                # 新しいノードにサブグラフを再接続
                for sub_node in old_sub
                    e_old = NamedEdge(min(old_node, sub_node) => max(old_node, sub_node))
                    if has_edge(g_new, e_old) && haskey(edge2id, e_old)
                        e_old_id = edge2id[e_old]
                        rem_edge!(g_new, e_old)

                        # 新しいエッジを追加
                        e_new = NamedEdge(min(new_node, sub_node) => max(new_node, sub_node))
                        id2edge_new[e_old_id] = e_new
                        add_edge!(g_new, e_new)
                    end
                end
            end
        end

        tci_tmp = SimpleTCI{ValueType}(f, tci.localdims, g_new, initialpivots)
        # tci_tmp.converged_IJset = tci.converged_IJset
        _, _ = optimize!(tci_tmp, f; kwargs...)
        err = maximum(values(tci_tmp.bonderrors))

        # 7. update best
        if err < best_err
            best_err = err
            tci_best = deepcopy(tci_tmp)
            id2edge_best = deepcopy(id2edge_new)
        end
    end

    return tci_best, id2edge_best
end

# Optimization with local 2-site tensor manipulation
function optimize_with_localmanipulation(
    tci::SimpleTCI{ValueType},
    f,
    initialpivots::Vector{MultiIndex},
    center_edge::NamedEdge,
    max_degree::Int,
    id2edge,
    edge2id;
    kwargs...
) where {ValueType}

    best_err = maximum(values(tci.bonderrors))
    tci_best = tci
    id2edge_best = id2edge

    # calculate the two ends of the split object and its child node list
    p, q = src(center_edge), dst(center_edge)
    subI = filter(x->x!=q, neighbors(tci.g,p))
    subJ = filter(x->x!=p, neighbors(tci.g,q))
    children = vcat(subI, subJ)
    n = length(children)
    limit = max_degree

    # k nodes are left (p side), the rest are right (q side)
    for k in 0:n
        if k ≤ limit && (n-k) ≤ limit
            for left in combinations(children, k)
                leftset = Set(left)

                # copy the graph and replace one by one
                g_new = deepcopy(tci.g)
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

                # optimize
                tci_tmp = SimpleTCI{ValueType}(f, tci.localdims, g_new, initialpivots)
                # tci_tmp.converged_IJset = tci.converged_IJset
                _, _ = optimize!(tci_tmp, f; kwargs...)
                err = maximum(values(tci_tmp.bonderrors))

                # update best
                if err < best_err && !(err ≈ best_err)
                    best_err = err
                    tci_best = deepcopy(tci_tmp)
                    id2edge_best = deepcopy(id2edge_new)
                end

            end
        end
    end

    return tci_best, id2edge_best
end