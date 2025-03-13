function fillsitetensors(
    tci::SimpleTCI{ValueType},
    f;
    center_vertex::Int = 0,
) where {ValueType}

    sitetensors =
        Vector{Pair{Array{ValueType},Vector{NamedEdge}}}(undef, length(vertices(tci.g)))

    if center_vertex ∉ vertices(tci.g)
        center_vertex = first(vertices(tci.g))
    end
    state = namedgraph_dijkstra_shortest_paths(tci.g, center_vertex)

    distances = state.dists
    max_distance = maximum(distances[v] for v in vertices(tci.g))
    for d = max_distance:-1:0
        children = filter(v -> distances[v] == d, vertices(tci.g))
        for child in children
            # adjacent_edges = adjacentedges(tci.g, child)
            parent = state.parents[child]
            edge = filter(
                e ->
                    src(e) == parent && dst(e) == child ||
                        dst(e) == parent && src(e) == child,
                edges(tci.g),
            )
            edge = isempty(edge) ? nothing : only(edge)
            incomingedges = setdiff(adjacentedges(tci.g, child), Set([edge]))
            InKeys =
                !isempty(incomingedges) ? edgeInIJkeys(tci.g, child, incomingedges) :
                SubTreeVertex[]
            OutKeys = edge != nothing ? edgeInIJkeys(tci.g, child, edge) : SubTreeVertex[]
            if d != 0
                T = sitetensor(tci, child, edge, InKeys => OutKeys, f)
                sitetensors[child] = T => vcat(incomingedges, [edge])
            else
                T = sitetensor(tci, child, edge, InKeys => OutKeys, f, core = true)
                sitetensors[child] = T => incomingedges
            end
        end
    end
    return sitetensors
end

function sitetensor(
    tci::SimpleTCI{ValueType},
    site::Int,
    InOutkeys,
    T::AbstractArray{ValueType,N},
) where {ValueType,N}
    Inkeys, Outkeys = InOutkeys
    return reshape(
        T,
        tci.localdims[site],
        [length(tci.IJset[key]) for key in Inkeys]...,
        [length(tci.IJset[key]) for key in Outkeys]...,
    )
end

function sitetensor(
    tci::SimpleTCI{ValueType},
    site::Int,
    edge,
    InOutkeys,
    f;
    core = false,
) where {ValueType}
    Inkeys, Outkeys = InOutkeys
    L = length(tci.localdims)
    Pi1 = filltensor(ValueType, f, tci.localdims, tci.IJset, Inkeys, Outkeys, Val(1))
    Pi1 = reshape(
        Pi1,
        prod(vcat([tci.localdims[site]], [length(tci.IJset[key]) for key in Inkeys])),
        prod([length(tci.IJset[key]) for key in Outkeys]),
    )
    updatemaxsample!(tci, Pi1)

    if core
        T = sitetensor(tci, site, InOutkeys, Pi1)
        return T
    end

    p, q = separatevertices(tci.g, edge)
    if p == site
        I1key = subtreevertices(tci.g, q => p)
    elseif q == site
        I1key = subtreevertices(tci.g, p => q)
    end

    P = reshape(
        filltensor(ValueType, f, tci.localdims, tci.IJset, [I1key], Outkeys, Val(0)),
        length(tci.IJset[I1key]),
        prod([length(tci.IJset[key]) for key in Outkeys]),
    )
    length(tci.IJset[I1key]) == sum([length(tci.IJset[key]) for key in Outkeys]) || error("Pivot matrix at bond $(site) is not square!")
    Tmat = transpose(transpose(P) \ transpose(Pi1))
    T = reshape(
        Tmat,
        tci.localdims[site],
        [length(tci.IJset[key]) for key in Inkeys]...,
        [length(tci.IJset[key]) for key in Outkeys]...,
    )
    return T
end


function filltensor(
    ::Type{ValueType},
    f,
    localdims::Vector{Int},
    IJset::Dict{SubTreeVertex,Vector{MultiIndex}},
    Inkeys::Vector{SubTreeVertex},
    Outkeys::Vector{SubTreeVertex},
    ::Val{M},
)::Array{ValueType} where {ValueType,M}
    N = length(localdims)
    nin = sum([length(first(IJset[key])) for key in Inkeys])
    nout = sum([length(first(IJset[key])) for key in Outkeys])
    ncent = N - nin - nout
    M == ncent || error("Invalid number of central indices")
    Inlocaldims = [[localdims[i] for i in IJkey] for IJkey in Inkeys]
    Outlocaldims = [[localdims[i] for i in IJkey] for IJkey in Outkeys]
    Clocaldims = [
        localdims[i] for i = 1:N if
        all(i ∉ IJkey for IJkey in Inkeys) && all(i ∉ IJkey for IJkey in Outkeys)
    ]
    expected_size = (
        Clocaldims...,
        prod([length(IJset[key]) for key in Inkeys]),
        prod([length(IJset[key]) for key in Outkeys]),
    )
    return reshape(
        _call(ValueType, f, localdims, IJset, Inkeys, Outkeys, Val(ncent)),
        expected_size...,
    )
end

function _call(
    ::Type{V},
    f,
    localdims::Vector{Int},
    IJset::Dict{SubTreeVertex,Vector{MultiIndex}},
    Inkeys::Vector{SubTreeVertex},
    Outkeys::Vector{SubTreeVertex},
    ::Val{M},
)::Array{V} where {V,M}
    N = length(localdims)
    nin = prod([length(first(IJset[key])) for key in Inkeys])
    nout = prod([length(first(IJset[key])) for key in Outkeys])
    L = M + nin + nout

    Ckey = [
        i for i = 1:N if
        all(i ∉ IJkey for IJkey in Inkeys) && all(i ∉ IJkey for IJkey in Outkeys)
    ]
    Clocaldims = [localdims[i] for i in Ckey]

    indexset = MultiIndex(undef, L)
    result = Array{V,3}(
        undef,
        prod(Clocaldims),
        prod([length(IJset[key]) for key in Inkeys]),
        prod([length(IJset[key]) for key in Outkeys]),
    )

    for (c, cindex) in enumerate(Iterators.product(ntuple(x -> 1:Clocaldims[x], M)...))
        indexset = zeros(Int, N)
        for (idx, key) in enumerate(Ckey)
            indexset[key] = cindex[idx]
        end
        for (i, lindices) in
            enumerate(Iterators.product((IJset[inkey] for inkey in Inkeys)...))
            for (inkey, index) in zip(Inkeys, lindices)
                for (idx, key) in enumerate(inkey)
                    indexset[key] = index[idx]
                end
            end

            for (j, rindices) in
                enumerate(Iterators.product((IJset[outkey] for outkey in Outkeys)...))
                for (outkey, index) in zip(Outkeys, rindices)
                    for (idx, key) in enumerate(outkey)
                        indexset[key] = index[idx]
                    end
                end
                result[c, i, j] = f(indexset)
            end
        end
    end
    return result
end

function edgeInIJkeys(g::NamedGraph, v::Int, combinededges)
    if combinededges isa NamedEdge
        combinededges = [combinededges]
    end
    keys = SubTreeVertex[]
    for edge in combinededges
        p, q = separatevertices(g, edge)
        if p == v
            push!(keys, subtreevertices(g, p => q))
        elseif q == v
            push!(keys, subtreevertices(g, q => p))
        end
    end
    return keys
end
