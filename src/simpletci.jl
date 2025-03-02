using Base: SimpleLogger

mutable struct SimpleTCI{ValueType}
    IJset::Dict{SubTreeVertex, Vector{MultiIndex}}
    localdims::Vector{Int}
    g:: NamedGraph
    sitetensors::Dict{Int, Array{ValueType}}
    regionbonds::Dict{Pair{SubTreeVertex, SubTreeVertex}, NamedEdge}
    # "Error estimate for backtruncation of bonds."
    pivoterrors::Dict{Pair{SubTreeVertex, SubTreeVertex}, Float64} # key is the bond id
    #"Error estimate per bond by 2site sweep."
    bonderrors::Dict{Pair{SubTreeVertex, SubTreeVertex}, Float64} # key is the bond id
    #"Maximum sample for error normalization."
    maxsamplevalue::Float64
    IJset_history::Dict{SubTreeVertex, Vector{Vector{MultiIndex}}}

    function SimpleTCI{ValueType}(
        localdims::Vector{Int},
        g::NamedGraph,
    ) where {ValueType}
        length(localdims) > 1 || error("localdims should have at least 2 elements!")
        n = length(localdims)

        # assign the key for each bond
        regionbonds = Dict{Pair{SubTreeVertex, SubTreeVertex}, NamedEdge}()
        pivoterrors = Dict{Pair{SubTreeVertex, SubTreeVertex}, Float64}()
        bonderrors = Dict{Pair{SubTreeVertex, SubTreeVertex}, Float64}()
        for e in edges(g)
            subregions_pair = subregion_vertices(g, e)
            regionbonds[subregions_pair] = e
            pivoterrors[subregions_pair] = 0.0
            bonderrors[subregions_pair] = 0.0
        end

        # make an IndexedArray for each verticec
        sitetensors = Dict{Int, Array{ValueType}}()
        for v in vertices(g)
            dims = [localdims[v]; [0 for e in edges(g) if src(e) == v || dst(e) == v]...]
            sitetensors[v] = zeros(dims...)
        end

        !Graphs.is_cyclic(g) || error("TreeTensorNetwork is not supported for loopy tensor network.")

        new{ValueType}(
            Dict{SubTreeVertex, Vector{MultiIndex}}(),               # IJset
            localdims,
            g,
            sitetensors,
            regionbonds,
            pivoterrors,
            bonderrors,
            0.0,                                                   # maxsamplevalue
            Dict{SubTreeVertex, Vector{Vector{MultiIndex}}}(),       # IJset_history
        )
    end
end

function SimpleTCI{ValueType}(
    func::F,
    localdims::Vector{Int},
    g::NamedGraph,
    initialpivots::Vector{MultiIndex}=[ones(Int, length(localdims))],
) where {F, ValueType}
    tci = SimpleTCI{ValueType}(localdims, g)
    addglobalpivots!(tci, initialpivots)
    tci.maxsamplevalue = maximum(abs, (func(x) for x in initialpivots))
    abs(tci.maxsamplevalue) > 0 || error("The function should not be zero at the initial pivot.")
    invalidatesitetensors!(tci)
    return tci
end

"""
Add global pivots to index sets
"""
function addglobalpivots!(
    tci::SimpleTCI{ValueType},
    pivots::Vector{MultiIndex}
) where {ValueType}
    if any(length(tci.localdims) .!= length.(pivots)) # AbstructTreeTensorNetworkをから引き継ぎlength(tci)ができると良い
        throw(DimensionMismatch("Please specify a pivot as one index per leg of the TTN."))
    end
    for pivot in pivots
        for e in edges(tci.g)
            p, q = separate_vertices(tci.g, e)
            Iset_key = subtree_vertices(tci.g, p => q)
            Jset_key = subtree_vertices(tci.g, q => p)

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

    if length(pivots) > 0
        invalidatesitetensors!(tci)
    end

    nothing
end

"""
Invalidate the site tensor at bond `b`.
"""
function invalidatesitetensors!(tci::SimpleTCI{T}) where {T}
    for v in vertices(tci.g)
        tci.sitetensors[v] = zeros(T, [0 for _ in size(tci.sitetensors[v])]...)
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
    initialpivots::Vector{MultiIndex}=[ones(Int, length(localdims))];
    kwargs...
) where {ValueType,N}
    tci = SimpleTCI{ValueType}(f, localdims, g, initialpivots)
    ranks, errors = optimize!(tci, f; kwargs...)
    return tci, ranks, errors
end

function optimize!(
    tci::SimpleTCI{ValueType},
    f;
    tolerance::Union{Float64, Nothing}=nothing,
    pivottolerance::Union{Float64, Nothing}=nothing,
    maxbonddim::Int=typemax(Int),
    maxiter::Int=20,
    sweepstrategy::Symbol=:backandforth, # TODO: Implement for Tree structure
    pivotsearch::Symbol=:full,
    verbosity::Int=0,
    loginterval::Int=10,
    normalizeerror::Bool=true,
    ncheckhistory::Int=3,
    maxnglobalpivot::Int=5,
    nsearchglobalpivot::Int=5,
    tolmarginglobalsearch::Float64=10.0,
    strictlynested::Bool=false,
    checkbatchevaluatable::Bool=false
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
            throw(ArgumentError("Got different values for pivottolerance and tolerance in optimize!(TCI2). For TCI2, both of these options have the same meaning. Please assign only `tolerance`."))
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
        throw(ArgumentError(
            "Specify either tolerance > 0 or some maxbonddim; otherwise, the convergence criterion is not reachable!"
        ))
    end

    globalpivots = MultiIndex[]
    for iter in 1:maxiter
        errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
        abstol = tol * errornormalization;

        if verbosity > 1
            println("  Walltime $(1e-9*(time_ns() - tstart)) sec: starting 2site sweep")
            flush(stdout)
        end

        sweep2site!(
            tci, f, 2;
            iter1 = 1,
            abstol=abstol,
            maxbonddim=maxbonddim,
            pivotsearch=pivotsearch,
            strictlynested=strictlynested,
            verbosity=verbosity,
            sweepstrategy=sweepstrategy,
            fillsitetensors=true
            )
        if verbosity > 0 && length(globalpivots) > 0 && mod(iter, loginterval) == 0
            abserr = [abs(evaluate(tci, p) - f(p)) for p in globalpivots]
            nrejections = length(abserr .> abstol)
            if nrejections > 0
                println("  Rejected $(nrejections) global pivots added in the previous iteration, errors are $(abserr)")
                flush(stdout)
            end
        end
        push!(errors, pivoterror(tci))

        if verbosity > 1
            println("  Walltime $(1e-9*(time_ns() - tstart)) sec: start searching global pivots")
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
    tci::SimpleTCI{ValueType}, f, niter::Int;
    iter1::Int=1,
    abstol::Float64=1e-8,
    maxbonddim::Int=typemax(Int),
    sweepstrategy::Symbol=:backandforth,
    pivotsearch::Symbol=:full,
    verbosity::Int=0,
    strictlynested::Bool=false,
    fillsitetensors::Bool=true
) where {ValueType}
    invalidatesitetensors!(tci)

    n = length(tci.localdims) # TODO: Implement for AbstractTreeTensorNetwork

    # assigne the uuid for each bond
    invariantbondids = Dict([i => key for (i, key) in enumerate(keys(tci.regionbonds))])
    reverse_invariantbondids = Dict([key => i for (i, key) in enumerate(keys(tci.regionbonds))])

    # choose the center bond id.
    d = n
    origin_id = undef
    for (id, key) in invariantbondids
        e = tci.regionbonds[key]
        p, q = separate_vertices(tci.g, e)
        Iset = length(subtree_vertices(tci.g, p => q))
        Jset = length(subtree_vertices(tci.g, q => p))
        d_tmp = abs(Iset - Jset)
        if d_tmp < d
            d = d_tmp
            origin_id = id
        end
    end
    center_id = origin_id

    for iter in iter1:iter1+niter-1
        extraIJset = Dict(key => MultiIndex[] for key in keys(tci.IJset))
        if !strictlynested && length(tci.IJset_history) > 0
            extraIJset = Dict(key => tci.IJset_history[key][end] for key in keys(tci.IJset_history))
        end

        for key in keys(tci.IJset)
            if !haskey(tci.IJset_history, key)
                tci.IJset_history[key] = [deepcopy(tci.IJset[key])]
            else
                push!(tci.IJset_history[key], deepcopy(tci.IJset[key]))
            end
        end
        flushpivoterror!(tci)

        # Init flags
        flags = Dict{Int, Bool}()
        for key in keys(invariantbondids)
            flags[key] = 0
        end

        # Sweep until all bonds are applied.
        while true
            distances = bonddistances(tci.g, tci.regionbonds, invariantbondids[origin_id])

            candidates = bondinfocandidates(tci.g, tci.regionbonds, invariantbondids[center_id])
            candidates = filter(((c, parent_child), ) -> flags[reverse_invariantbondids[c]] == 0, candidates)

            # If candidates is empty, exit while loop
            if isempty(candidates)
                break
            end

            max_distance = maximum(distances[c] for (c, parent_child) in candidates)
            candidates = filter(((c, parent_child), ) -> distances[c] == max_distance, candidates)

            center_info = first(candidates)

            incomings = bondcandidates(tci.g, last(center_info), tci.regionbonds)
            # Update flags. However, the center bond is not applied.
            if all(flags[reverse_invariantbondids[c]] == 1 for c in incomings) && center_id != origin_id
                flags[center_id] = 1
            end

            # pivot candidates
            center_bond = first(center_info)
            # Update the pivot at the center bond.
            updatepivots!(
                    tci, center_bond, f;
                    abstol=abstol,
                    maxbonddim=maxbonddim,
                    verbosity=verbosity,
                )
        end
    end

    if fillsitetensors
        fillsitetensors!(tci, f)
    end
    nothing
end

function fillsitetensors!(
    tci::SimpleTCI{ValueType}, f) where {ValueType}
    for b in 1:length(tci)
       setsitetensor!(tci, f, b)
    end
    nothing
end

function setsitetensor!(
    tci::SimpleTCI{ValueType}, b::Int, T::AbstractArray{ValueType,N}
) where {ValueType,N}
    L = length(tci.localdims)
    I_key = collect(1:b-1)
    J_key = collect(b+1:L)
    tci.sitetensors[b] = reshape(
        T,
        length(tci.Iset[I_key]),
        tci.localdims[b],
        length(tci.Jset[J_key])
    )
end

function flushpivoterror!(tci::SimpleTCI{T}) where {T}
    for key in keys(tci.regionbonds)
        tci.pivoterrors[key] = 0.0
    end
    nothing
end

function forwardsweep(sweepstrategy::Symbol, iteration::Int)
    return (
        (sweepstrategy == :forward) ||
        (sweepstrategy == :backandforth && isodd(iteration))
    )
end


"""
Update pivots at bond `b` of `tci` using the TCI2 algorithm.
Site tensors will be invalidated.
"""
function updatepivots!(
    tci::SimpleTCI{ValueType},
    bond::Pair{SubTreeVertex,SubTreeVertex},
    f::F;
    reltol::Float64=1e-14,
    abstol::Float64=0.0,
    maxbonddim::Int=typemax(Int),
    verbosity::Int=0,
) where {F,ValueType}
    invalidatesitetensors!(tci)
    N = length(tci.localdims)

    vp, vq = separate_vertices(tci.g, tci.regionbonds[bond])
    Ikey, subIkey = subtree_vertices(tci.g, vp => vq), vq
    Jkey, subJkey = subtree_vertices(tci.g, vq => vp), vp

    # extract last histroy if not strictly nested
    extraIJset = Dict(key => MultiIndex[] for key in keys(tci.IJset))
    if length(tci.IJset_history) > 0
        extraIJset = Dict(key => tci.IJset_history[key][end] for key in keys(tci.IJset_history))
    end

    Icombined, Jcombined = generate_pivot_candidates(
        DefaultPivotCandidateProper(),
        tci.g,
        tci.IJset,
        tci.IJset_history,
        extraIJset,
        tci.regionbonds,
        tci.localdims,
        bond,
        (Ikey => Jkey),
        (subIkey => subJkey),
    )

    t1 = time_ns()
    Pi = reshape(
        filltensor(ValueType, f, tci.localdims,
        Icombined, Jcombined, Val(0)),
        length(Icombined), length(Jcombined)
    )
    t2 = time_ns()

    updatemaxsample!(tci, Pi)
    luci = TCI.MatrixLUCI(
        Pi,
        reltol=reltol,
        abstol=abstol,
        maxrank=maxbonddim,
    )
    t3 = time_ns()
    if verbosity > 2
        x, y = length(Icombined), length(Jcombined)
        println("    Computing Pi ($x x $y) at bond $b: $(1e-9*(t2-t1)) sec, LU: $(1e-9*(t3-t2)) sec")
    end

    tci.IJset[Ikey] = Icombined[TCI.rowindices(luci)]
    tci.IJset[Jkey] = Jcombined[TCI.colindices(luci)]
    if length(first(extraIJset)) == 0 && length(last(extraIJset)) == 0
        setsitetensor!(tci, b, TCI.left(luci))
        setsitetensor!(tci, b + 1, TCI.right(luci))
    end
    updateerrors!(tci, b, TCI.pivoterrors(luci))
    nothing
end

function setsitetensor!(
    tci::SimpleTCI{ValueType}, b::Int, T::AbstractArray{ValueType,N}
) where {ValueType,N}
    L = length(tci.localdims)
    I_key = collect(1:b-1) # !TODO: It is only for TT structure.
    J_key = collect(b+1:L) # !TODO: It is only for TT structure.
    tci.sitetensors[b] = reshape(
        T,
        length(tci.Iset[I_key]),
        tci.localdims[b],
        length(tci.Jset[J_key])
    )
end

function setsitetensor!(
    tci::SimpleTCI{ValueType}, f, b::Int; leftorthogonal=true
) where {ValueType}
    leftorthogonal || error("leftorthogonal==false is not supported!")
    N = length(tci.localdims)
    I_key = collect(1:b-1) # !TODO: It is only for TT structure.
    J_key = collect(b+1:N) # !TODO: It is only for TT structure.
    Is = leftorthogonal ? kronecker(tci.Iset[I_key], tci.localdims[b]) : tci.Iset[I_key]
    Js = leftorthogonal ? tci.Jset[J_key] : kronecker(tci.localdims[b], tci.Jset[J_key])
    Pi1 = reshape(
        filltensor(ValueType, f, tci.localdims, tci.Iset[I_key], tci.Jset[J_key], Val(1)),
        length(Is), length(Js))
    updatemaxsample!(tci, Pi1)

    if (leftorthogonal && b == length(tci)) ||
        (!leftorthogonal && b == 1)
        setsitetensor!(tci, b, Pi1)
        return tci.sitetensors[b]
    end

    I1_key = collect(1:b) # !TODO: It is only for TT structure.
    P = reshape(
        filltensor(ValueType, f, tci.localdims, tci.Iset[I1_key], tci.Jset[J_key], Val(0)),
        length(tci.Iset[I1_key]), length(tci.Jset[J_key]))
    length(tci.Iset[I1_key]) == length(tci.Jset[J_key]) || error("Pivot matrix at bond $(b) is not square!")

    #Tmat = transpose(transpose(rrlu(P)) \ transpose(Pi1))
    Tmat = transpose(transpose(P) \ transpose(Pi1))
    tci.sitetensors[b] = reshape(Tmat, length(tci.Iset[I_key]), tci.localdims[b], length(tci.Iset[I1_key]))
    return tci.sitetensors[b]
end


function updatemaxsample!(tci::SimpleTCI{V}, samples::Array{V}) where {V}
    tci.maxsamplevalue = TCI.maxabs(tci.maxsamplevalue, samples)
end

function updateerrors!(
    tci::SimpleTCI{T},
    b::Int,
    errors::AbstractVector{Float64}
) where {T}
    updatebonderror!(tci, b, last(errors))
    updatepivoterror!(tci, errors)
    nothing
end

function updatebonderror!(
    tci::SimpleTCI{T}, b::Int, error::Float64
) where {T}
    tci.bonderrors[b] = error
    nothing
end

function updatepivoterror!(tci::SimpleTCI{T}, errors::AbstractVector{Float64}) where {T}
    erroriter = Iterators.map(max, TCI.padzero(tci.pivoterrors), TCI.padzero(errors))
    tci.pivoterrors = Iterators.take(
        erroriter,
        max(length(tci.pivoterrors), length(errors))
    ) |> collect
    nothing
end


function filltensor(
    ::Type{ValueType},
    f,
    localdims::Vector{Int},
    Iset::Vector{MultiIndex},
    Jset::Vector{MultiIndex},
    ::Val{M}
)::Array{ValueType,M+2} where {ValueType,M}
    if length(Iset) * length(Jset) == 0
        return Array{ValueType,M+2}(undef, ntuple(i->0, M+2)...)
    end

    N = length(localdims)
    nl = length(first(Iset))
    nr = length(first(Jset))
    ncent = N - nl - nr
    expected_size = (length(Iset), localdims[nl+1:nl+ncent]..., length(Jset))
    M == ncent || error("Invalid number of central indices")
    return reshape(
        _call(ValueType, f, localdims, Iset, Jset, Val(ncent)),
        expected_size...
    )
end

function _call(
    ::Type{V},
    f,
    localdims::Vector{Int},
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M}
)::Array{V,M + 2} where {V,M}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{V,M+2}(undef, ntuple(i->0, M+2)...)
    end

    nl = length(first(leftindexset))
    nr = length(first(rightindexset))
    L = M + nl + nr
    indexset = MultiIndex(undef, L)
    result = Array{V, 3}(undef, length(leftindexset), prod(localdims[nl+1:L-nr]), length(rightindexset))
    for (i, lindex) in enumerate(leftindexset)
        for (c, cindex) in enumerate(Iterators.product(ntuple(x -> 1:localdims[nl+x], M)...))
            for (j, rindex) in enumerate(rightindexset)
                indexset[1:nl] .= lindex
                indexset[nl+1:L-nr] .= cindex
                indexset[L-nr+1:L] .= rindex
                result[i, c, j] = f(indexset)
            end
        end
    end

    return reshape(result, length(leftindexset), localdims[nl+1:L-nr]..., length(rightindexset))

end

function pivoterror(tci::SimpleTCI{T}) where {T}
    return maxbonderror(tci)
end

function maxbonderror(tci::SimpleTCI{T}) where {T}
    return maximum(tci.bonderrors)
end


function searchglobalpivots(
    tci::SimpleTCI{ValueType}, f, abstol;
    verbosity::Int=0,
    nsearch::Int = 100,
    maxnglobalpivot::Int = 5
)::Vector{MultiIndex} where {ValueType}
    if nsearch == 0 || maxnglobalpivot == 0
        return MultiIndex[]
    end

    if !issitetensorsavailable(tci)
        fillsitetensors!(tci, f)
    end

    pivots = Dict{Float64,MultiIndex}()
    ttcache = TTCache(tci)
    for _ in 1:nsearch
        pivot, error = _floatingzone(ttcache, f; earlystoptol = 10 * abstol, nsweeps=100)
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

    return [p for (_,p) in pivots]
end

"""
Return if site tensors are available
"""
function issitetensorsavailable(tci::SimpleTCI{T}) where {T}
    return all(length(tci.sitetensors[b]) != 0 for b in 1:length(tci))
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
