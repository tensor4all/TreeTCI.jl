"""
Invalidate the site tensor at bond `b`.
"""
function invalidatesitetensors!(tci::SimpleTCI{T}) where {T}
    for v in vertices(tci.g)
        tci.sitetensors[v] =
            zeros(T, [0 for _ in size(first(tci.sitetensors[v]))]...) => [e for e in edges(tci.g) if src(e) == v || dst(e) == v]
    end
    nothing
end

"""
See also: [`optimize!`](@ref), [`optfirstpivot`](@ref), [`CachedFunction`](@ref), [`crossinterpolate1`](@ref)
"""
function crossinterpolate(
    ::Type{ValueType},
    f,
    localdims::Union{Vector{Int},NTuple{N,Int}},
    g::NamedGraph,
    initialpivots::Vector{MultiIndex} = [ones(Int, length(localdims))];
    kwargs...,
) where {ValueType,N}
    tci = SimpleTCI{ValueType}(f, localdims, g, initialpivots)
    ranks, errors = optimize!(tci, f; kwargs...)
    return tci, ranks, errors
end

function optimize!(
    tci::SimpleTCI{ValueType},
    f;
    tolerance::Union{Float64,Nothing} = nothing,
    pivottolerance::Union{Float64,Nothing} = nothing,
    maxbonddim::Int = typemax(Int),
    maxiter::Int = 20,
    sweepstrategy::Symbol = :backandforth, # TODO: Implement for Tree structure
    pivotsearch::Symbol = :full,
    verbosity::Int = 0,
    loginterval::Int = 10,
    normalizeerror::Bool = true,
    ncheckhistory::Int = 3,
    maxnglobalpivot::Int = 5,
    nsearchglobalpivot::Int = 5,
    tolmarginglobalsearch::Float64 = 10.0,
    strictlynested::Bool = false,
    checkbatchevaluatable::Bool = false,
) where {ValueType}
    errors = Float64[]
    ranks = Int[]
    nglobalpivots = Int[]
    local tol::Float64

    if checkbatchevaluatable && !(f isa BatchEvaluator)
        error("Function `f` is not batch evaluatable")
    end

    if nsearchglobalpivot > 0 && nsearchglobalpivot < maxnglobalpivot
        error("nsearchglobalpivot < maxnglobalpivot!")
    end

    # Deprecate the pivottolerance option
    if !isnothing(pivottolerance)
        if !isnothing(tolerance) && (tolerance != pivottolerance)
            throw(
                ArgumentError(
                    "Got different values for pivottolerance and tolerance in optimize!(TCI2). For TCI2, both of these options have the same meaning. Please assign only `tolerance`.",
                ),
            )
        else
            @warn "The option `pivottolerance` of `optimize!(tci::TensorCI2, f)` is deprecated. Please update your code to use `tolerance`, as `pivottolerance` will be removed in the future."
            tol = pivottolerance
        end
    elseif !isnothing(tolerance)
        tol = tolerance
    else # pivottolerance == tolerance == nothing, therefore set tol to default value
        tol = 1e-8
    end

    tstart = time_ns()

    if maxbonddim >= typemax(Int) && tol <= 0
        throw(
            ArgumentError(
                "Specify either tolerance > 0 or some maxbonddim; otherwise, the convergence criterion is not reachable!",
            ),
        )
    end

    globalpivots = MultiIndex[]
    for iter = 1:maxiter
        errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
        abstol = tol * errornormalization

        if verbosity > 1
            println("  Walltime $(1e-9*(time_ns() - tstart)) sec: starting 2site sweep")
            flush(stdout)
        end

        sweep2site!(
            tci,
            f,
            2;
            iter1 = 1,
            abstol = abstol,
            maxbonddim = maxbonddim,
            pivotsearch = pivotsearch,
            verbosity = verbosity,
            sweepstrategy = sweepstrategy,
            fillsitetensors = true,
        )
        if verbosity > 0 && length(globalpivots) > 0 && mod(iter, loginterval) == 0
            abserr = [abs(evaluate(tci, p) - f(p)) for p in globalpivots]
            nrejections = length(abserr .> abstol)
            if nrejections > 0
                println(
                    "  Rejected $(nrejections) global pivots added in the previous iteration, errors are $(abserr)",
                )
                flush(stdout)
            end
        end
        push!(errors, last(pivoterror(tci)))

        if verbosity > 1
            println(
                "  Walltime $(1e-9*(time_ns() - tstart)) sec: start searching global pivots",
            )
            flush(stdout)
        end
    end

    errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
    return ranks, errors ./ errornormalization
end

"""
Perform 2site sweeps on a SimpleTCI.
!TODO: Implement for Tree structure

"""
function sweep2site!(
    tci::SimpleTCI{ValueType},
    f,
    niter::Int;
    iter1::Int = 1,
    abstol::Float64 = 1e-8,
    maxbonddim::Int = typemax(Int),
    sweepstrategy::Symbol = :backandforth,
    pivotsearch::Symbol = :full,
    verbosity::Int = 0,
    fillsitetensors::Bool = false,
) where {ValueType}
    invalidatesitetensors!(tci)

    bond_path, origin_bond = generate_sweep2site_path(DefaultSweep2sitePathProper(), tci)

    for iter = iter1:iter1+niter-1
        extraIJset = Dict(key => MultiIndex[] for key in keys(tci.IJset))
        if length(tci.IJset_history) > 0
            extraIJset = tci.IJset_history[end]
        end

        push!(tci.IJset_history, deepcopy(tci.IJset))

        flushpivoterror!(tci)

        for bond in bond_path
            updatepivots!(
                tci,
                bond,
                f;
                abstol = abstol,
                maxbonddim = maxbonddim,
                verbosity = verbosity,
                extraIJset = extraIJset,
            )
        end
    end

    if fillsitetensors
        fillsitetensors!(tci, f, origin_bond)
    end
    nothing
end

# WIP
function fillsitetensors!(
    tci::SimpleTCI{ValueType},
    f,
    center_bond::Pair{SubTreeVertex,SubTreeVertex},
) where {ValueType}

    distances = bonddistances(tci.g, tci.regionbonds, center_bond)
    max_distance = maximum(distances[c] for c in keys(tci.regionbonds))
    for d = max_distance:-1:0
        candidates = filter(b -> distances[b] == d, keys(tci.regionbonds))
        for bond in candidates
            vp, vq = separatevertices(tci.g, tci.regionbonds[bond])

            adjacent_bonds_vp = adjacentbonds(tci.g, vp, tci.regionbonds)
            adjacent_bonds_vq = adjacentbonds(tci.g, vq, tci.regionbonds)

            if d > 0
                bonds_ = filter(b -> distances[b] == d - 1, keys(tci.regionbonds))

                common_elements_vp = intersect(bonds_, adjacent_bonds_vp)
                common_elements_vq = intersect(bonds_, adjacent_bonds_vq)

                if isempty(common_elements_vp)
                    vf = vp
                    adjacent_bonds = adjacent_bonds_vp
                elseif isempty(common_elements_vq)
                    vf = vq
                    adjacent_bonds = adjacent_bonds_vq
                end
                InOutbonds, InOutkeys =
                    inoutbondskeys(tci.g, tci.regionbonds, distances, d, vf, adjacent_bonds)
                setsitetensor!(tci, vf, bond, InOutbonds, InOutkeys, f)
            else
                vf = vp
                adjacent_bonds = adjacent_bonds_vp
                InOutbonds, InOutkeys =
                    inoutbondskeys(tci.g, tci.regionbonds, distances, d, vf, adjacent_bonds)
                setsitetensor!(tci, vf, bond, InOutbonds, InOutkeys, f)

                vf = vq
                adjacent_bonds = adjacent_bonds_vq
                InOutbonds, InOutkeys =
                    inoutbondskeys(tci.g, tci.regionbonds, distances, d, vf, adjacent_bonds)
                setsitetensor!(tci, vf, bond, InOutbonds, InOutkeys, f; core = true)
            end
        end
    end
    nothing
end

function flushpivoterror!(tci::SimpleTCI{ValueType}) where {ValueType}
    tci.pivoterrors = Float64[]
    nothing
end


"""
Update pivots at bond `b` of `tci` using the TCI2 algorithm.
Site tensors will be invalidated.
"""
function updatepivots!(
    tci::SimpleTCI{ValueType},
    bond::Pair{SubTreeVertex,SubTreeVertex},
    f::F;
    reltol::Float64 = 1e-14,
    abstol::Float64 = 0.0,
    maxbonddim::Int = typemax(Int),
    verbosity::Int = 0,
    extraIJset::Dict{SubTreeVertex,Vector{MultiIndex}} = Dict{
        SubTreeVertex,
        Vector{MultiIndex},
    }(),
) where {F,ValueType}
    invalidatesitetensors!(tci)
    N = length(tci.localdims)

    (IJkey, combinedIJset) = generate_pivot_candidates(
        DefaultPivotCandidateProper(),
        tci,
        bond,
        extraIJset,
    )
    Ikey, Jkey = first(IJkey), last(IJkey)

    t1 = time_ns()
    Pi = reshape(
        filltensor(
            ValueType,
            f,
            tci.localdims,
            combinedIJset,
            [Ikey],
            [Jkey],
            Val(0),
        ),
        length(combinedIJset[Ikey]),
        length(combinedIJset[Jkey]),
    )
    t2 = time_ns()

    updatemaxsample!(tci, Pi)

    luci = TCI.MatrixLUCI(Pi, reltol = reltol, abstol = abstol, maxrank = maxbonddim)
    # TODO: we will implement luci according to optimal index subsets by following step
    # 1. Compute the optimal index subsets (We also need the indices to set new pivots)
    # 2. Reshape the Pi matrix by the optimal index subsets
    # 3. Compute the LUCI by the reshaped Pi matrix

    t3 = time_ns()
    if verbosity > 2
        x, y = length(combinedIJset[Ikey]), length(combinedIJset[Jkey]),
        println(
            "    Computing Pi ($x x $y) at bond $b: $(1e-9*(t2-t1)) sec, LU: $(1e-9*(t3-t2)) sec",
        )
    end

    tci.IJset[Ikey] = combinedIJset[Ikey][TCI.rowindices(luci)]
    tci.IJset[Jkey] = combinedIJset[Jkey][TCI.colindices(luci)]

    updateerrors!(tci, bond, TCI.pivoterrors(luci))
    nothing
end

function setsitetensor!(
    tci::SimpleTCI{ValueType},
    site::Int,
    InOutbonds,
    InOutkeys,
    T::AbstractArray{ValueType,N},
) where {ValueType,N}
    Inkeys, Outkeys = InOutkeys
    Inbonds, Outbonds = InOutbonds
    tci.sitetensors[site] = (
        reshape(
            T,
            tci.localdims[site],
            [length(tci.IJset[key]) for key in Inkeys]...,
            [length(tci.IJset[key]) for key in Outkeys]...,
        ) => vcat(
            [tci.regionbonds[bond] for bond in Inbonds],
            [tci.regionbonds[bond] for bond in Outbonds],
        )
    )
    nothing
end

function setsitetensor!(
    tci::SimpleTCI{ValueType},
    site::Int,
    bond::Pair{SubTreeVertex,SubTreeVertex},
    InOutbonds,
    InOutkeys,
    f;
    core = false,
) where {ValueType}
    Inkeys, Outkeys = InOutkeys
    Inbonds, Outbonds = InOutbonds
    L = length(tci.localdims)
    Pi1 = filltensor(ValueType, f, tci.localdims, tci.IJset, Inkeys, Outkeys, Val(1))
    Pi1 = reshape(
        Pi1,
        prod(vcat([tci.localdims[site]], [length(tci.IJset[key]) for key in Inkeys])),
        prod([length(tci.IJset[key]) for key in Outkeys]),
    )
    updatemaxsample!(tci, Pi1)

    if core
        setsitetensor!(tci, site, InOutbonds, InOutkeys, Pi1)
        return tci.sitetensors[site]
    end

    p, q = separatevertices(tci.g, tci.regionbonds[bond])
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
    tci.sitetensors[site] = (
        reshape(
            Tmat,
            tci.localdims[site],
            [length(tci.IJset[key]) for key in Inkeys]...,
            [length(tci.IJset[key]) for key in Outkeys]...,
        ) => vcat(
            [tci.regionbonds[bond] for bond in Inbonds],
            [tci.regionbonds[bond] for bond in Outbonds],
        )
    )
    return tci.sitetensors[site]
end

function updatemaxsample!(tci::SimpleTCI{V}, samples::Array{V}) where {V}
    tci.maxsamplevalue = TCI.maxabs(tci.maxsamplevalue, samples)
end

function updateerrors!(
    tci::SimpleTCI{T},
    bond::Pair{SubTreeVertex,SubTreeVertex},
    errors::AbstractVector{Float64},
) where {T}
    updatebonderror!(tci, bond, last(errors))
    updatepivoterror!(tci, errors)
    nothing
end

function updatebonderror!(
    tci::SimpleTCI{T},
    bond::Pair{SubTreeVertex,SubTreeVertex},
    error::Float64,
) where {T}
    tci.bonderrors[bond] = error
    nothing
end

function updatepivoterror!(tci::SimpleTCI{T}, errors::AbstractVector{Float64}) where {T}
    erroriter = Iterators.map(max, TCI.padzero(tci.pivoterrors), TCI.padzero(errors))
    tci.pivoterrors =
        Iterators.take(erroriter, max(length(tci.pivoterrors), length(errors))) |> collect
    nothing
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

function pivoterror(tci::SimpleTCI{T}) where {T}
    return maxbonderror(tci)
end

function maxbonderror(tci::SimpleTCI{T}) where {T}
    return maximum(tci.bonderrors)
end


function searchglobalpivots(
    tci::SimpleTCI{ValueType},
    f,
    abstol;
    verbosity::Int = 0,
    nsearch::Int = 100,
    maxnglobalpivot::Int = 5,
)::Vector{MultiIndex} where {ValueType}
    if nsearch == 0 || maxnglobalpivot == 0
        return MultiIndex[]
    end

    if !issitetensorsavailable(tci)
        fillsitetensors!(tci, f)
    end

    pivots = Dict{Float64,MultiIndex}()
    ttcache = TTCache(tci)
    for _ = 1:nsearch
        pivot, error = _floatingzone(ttcache, f; earlystoptol = 10 * abstol, nsweeps = 100)
        if error > abstol
            pivots[error] = pivot
        end
        if length(pivots) == maxnglobalpivot
            break
        end
    end

    if length(pivots) == 0
        if verbosity > 1
            println("  No global pivot found")
        end
        return MultiIndex[]
    end

    if verbosity > 1
        maxerr = maximum(keys(pivots))
        println("  Found $(length(pivots)) global pivots: max error $(maxerr)")
    end

    return [p for (_, p) in pivots]
end

"""
Return if site tensors are available
"""
function issitetensorsavailable(tci::SimpleTCI{T}) where {T}
    return all(length(tci.sitetensors[b]) != 0 for b = 1:length(tci))
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

function inoutbondskeys(g, regionbonds, distances, d, vf, adjacent_bonds)
    Inbonds =
        intersect(adjacent_bonds, filter(b -> distances[b] == d + 1, keys(regionbonds)))
    Outbonds = intersect(adjacent_bonds, filter(b -> distances[b] == d, keys(regionbonds)))
    Inkeys = bondtokey(g, vf, Inbonds, regionbonds)
    Outkeys = bondtokey(g, vf, Outbonds, regionbonds)
    return (Inbonds => Outbonds), (Inkeys => Outkeys)
end
