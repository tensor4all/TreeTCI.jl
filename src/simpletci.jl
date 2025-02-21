using Base: SimpleLogger
MultiIndex = Vector{Int}

mutable struct SimpleTCI{ValueType}
    IJset::Dict{Vector{Int}, Vector{MultiIndex}}
    localdims::Vector{Int}
    g:: NamedGraph
    sitetensors::Dict{Int, Array{ValueType}}
    idbonds::Dict{String, NamedEdge}
    # "Error estimate for backtruncation of bonds."
    pivoterrors::Dict{String, Float64} # key is the bond id
    #"Error estimate per bond by 2site sweep."
    bonderrors::Dict{String, Float64} # key is the bond id
    #"Maximum sample for error normalization."
    maxsamplevalue::Float64
    IJset_history::Dict{Vector{Int}, Vector{Vector{MultiIndex}}}

    function SimpleTCI{ValueType}(
        localdims::Vector{Int},
        g::NamedGraph,
    ) where {ValueType}
        length(localdims) > 1 || error("localdims should have at least 2 elements!")
        n = length(localdims)

        # make a unique bond index for each bond
        idbonds = Dict{String, NamedEdge}()
        pivoterrors = Dict{String, Float64}()
        bonderrors = Dict{String, Float64}()
        for e in edges(g)
            u = string(uuid4())
            idbonds[u] = e
            pivoterrors[u] = 0.0
            bonderrors[u] = 0.0
        end

        # make an IndexedArray for each verticec
        sitetensors = Dict{Int, Array{ValueType}}()
        for v in vertices(g)
            dims = [localdims[v]; [0 for e in edges(g) if src(e) == v || dst(e) == v]...]
            sitetensors[v] = zeros(dims...)
        end

        !Graphs.is_cyclic(g) || error("TreeTensorNetwork is not supported for loopy tensor network.")

        new{ValueType}(
            Dict{Vector{Int}, Vector{MultiIndex}}(),               # IJset
            localdims,
            g,
            sitetensors,
            idbonds,
            pivoterrors,
            bonderrors,
            0.0,                                                   # maxsamplevalue
            Dict{Vector{Int}, Vector{Vector{MultiIndex}}}(),       # IJset_history
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
            Iset_key = subregions(tci.g, p, q)
            Jset_key = subregions(tci.g, q, p)

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

    #if maxnglobalpivot > 0 && nsearchglobalpivot > 0
        #!strictlynested || error("nglobalpivots > 0 requires strictlynested=false!")
    #end
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

    # choose the center bond id.
    d = n
    center_id = undef
    for key in keys(tci.idbonds)
        e = tci.idbonds[key]
        p, q = separate_vertices(tci.g, e)
        Iset = length(subregions(tci.g, p, q))
        Jset = length(subregions(tci.g, q, p))
        d_tmp = abs(Iset - Jset)
        if d_tmp < d
            d = d_tmp
            center_id = key
        end
    end
    center_id_ = center_id

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
        flags = Dict{String, Bool}()
        for key in keys(tci.idbonds)
            flags[key] = 0
        end

        # Sweep until all bonds are applied.
        while true
            p, q = separate_vertices(tci.g, tci.idbonds[center_id])
            distances = Dict{String, Int}()
            distances[center_id] = 0
            distances = distanceBFS(tci.g, p, q, distances, tci.idbonds)
            distances = distanceBFS(tci.g, q, p, distances, tci.idbonds)

            p_, q_ = separate_vertices(tci.g, tci.idbonds[center_id_])
            candidates = vcat([(c, q_, p_) for c in candidate_bondids(tci.g, p_, q_, tci.idbonds)],
                            [(c, p_, q_) for c in candidate_bondids(tci.g, q_, p_, tci.idbonds)])
            candidates = filter(((c, vt, vf), ) -> flags[c] == 0, candidates)

            # If candidates is empty, exit while loop
            if isempty(candidates)
                break
            end

            max_distance = maximum(distances[c] for (c, vt, vf) in candidates)
            candidates = filter(((c, vt, vf), ) -> distances[c] == max_distance, candidates)

            center_info = first(candidates)

            incomings = candidate_bondids(tci.g, center_info[2], center_info[3], tci.idbonds)

            # Update flags. However, the center bond is not applied.
            if all(flags[c] == 1 for c in incomings) && center_id_ != center_id
                flags[center_id_] = 1
            end

            # Next center bond is the candidate with the maximum distance.
            center_id_ = center_info[1]
            vp, vq = separate_vertices(tci.g, tci.idbonds[center_id_])
            Ikeys, subIkey = subregions(tci.g, vp, vq), vq
            Jkeys, subJkey = subregions(tci.g, vq, vp), vp
            subIJkey = [subIkey, subJkey]
            IJkeys = [Ikeys, Jkeys]

            # Update the pivot at the center bond.
            updatepivots!(
                    tci, center_id_, f, true;
                    abstol=abstol,
                    maxbonddim=maxbonddim,
                    pivotsearch=pivotsearch,
                    verbosity=verbosity,
                    IJkeys=IJkeys,
                    subIJkey=subIJkey,
                    extraIJset=Dict(first(IJkeys) => extraIJset[first(IJkeys)], last(IJkeys) => extraIJset[last(IJkeys)])
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
    for key in keys(tci.idbonds)
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
    bondid::String,
    f::F,
    leftorthogonal::Bool; # 多分いらないけど、何に使ってるかに依存する。
    reltol::Float64=1e-14,
    abstol::Float64=0.0,
    maxbonddim::Int=typemax(Int),
    pivotsearch::Symbol=:full,
    verbosity::Int=0,
    IJkeys::Vector{Vector{Int}}=Vector{Vector{Int}}(),
    subIJkey::Vector{Int}=Vector{Int}(),
    extraIJset::Dict{Vector{Int}, Vector{MultiIndex}}=Dict{Vector{Int}, Vector{MultiIndex}}(),
) where {F,ValueType}
    invalidatesitetensors!(tci)
    N = length(tci.localdims)
    Ikey, Jkey = first(IJkeys), last(IJkeys)
    Iset = kronecker(tci.IJset, first(IJkeys), first(subIJkey), tci.localdims[first(subIJkey)])
    Jset = kronecker(tci.IJset, last(IJkeys), last(subIJkey), tci.localdims[last(subIJkey)])
    Icombined = union(Iset, extraIJset[Ikey])
    Jcombined = union(Jset, extraIJset[Jkey])

    luci = if pivotsearch === :full
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
            leftorthogonal=leftorthogonal
        )
        t3 = time_ns()
        if verbosity > 2
            x, y = length(Icombined), length(Jcombined)
            println("    Computing Pi ($x x $y) at bond $b: $(1e-9*(t2-t1)) sec, LU: $(1e-9*(t3-t2)) sec")
        end
        luci
    elseif pivotsearch === :rook
        t1 = time_ns()
        I0 = Int.(Iterators.filter(!isnothing, findfirst(isequal(i), Icombined) for i in tci.Iset[b+1]))::Vector{Int}
        J0 = Int.(Iterators.filter(!isnothing, findfirst(isequal(j), Jcombined) for j in tci.Jset[b]))::Vector{Int}
        Pif = SubMatrix{ValueType}(f, Icombined, Jcombined)
        t2 = time_ns()
        res = MatrixLUCI(
            ValueType,
            Pif,
            (length(Icombined), length(Jcombined)),
            I0, J0;
            reltol=reltol, abstol=abstol,
            maxrank=maxbonddim,
            leftorthogonal=leftorthogonal,
            pivotsearch=:rook,
            usebatcheval=true
        )
        updatemaxsample!(tci, [ValueType(Pif.maxsamplevalue)])

        t3 = time_ns()

        # Fall back to full search if rook search fails
        if npivots(res) == 0
            Pi = reshape(
                filltensor(ValueType, f, tci.localdims,
                Icombined, Jcombined, Val(0)),
                length(Icombined), length(Jcombined)
            )
            updatemaxsample!(tci, Pi)
            res = MatrixLUCI(
                Pi,
                reltol=reltol,
                abstol=abstol,
                maxrank=maxbonddim,
                leftorthogonal=leftorthogonal
            )
        end

        t4 = time_ns()
        if verbosity > 2
            x, y = length(Icombined), length(Jcombined)
            println("    Computing Pi ($x x $y) at bond $b: $(1e-9*(t2-t1)) sec, LU: $(1e-9*(t3-t2)) sec, fall back to full: $(1e-9*(t4-t3)) sec")
        end
        res
    else
        throw(ArgumentError("Unknown pivot search strategy $pivotsearch. Choose from :rook, :full."))
    end
    I_key = collect(1:b)
    J_key = collect(b+1:N)
    tci.Iset[I_key] = Icombined[TCI.rowindices(luci)]
    tci.Jset[J_key] = Jcombined[TCI.colindices(luci)]
    if length(extraIset) == 0 && length(extraJset) == 0
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

function kronecker(
    IJset::Dict{Vector{Int}, Vector{MultiIndex}},
    subregions::Vector{Int},  # original subregions order
    site::Int,          # direct connected site
    localdim::Int,
)
    site_index = findfirst(==(site), subregions)
    filtered_subregions = filter(x -> x ≠ Set([site]), subregions)
    pivotset = IJset[subregions]
    return MultiIndex[[is[1:site_index-1]..., j, is[site_index:end]...] for is in pivotset, j in 1:localdim][:]
end

function kronecker(
    Iset::Union{Vector{MultiIndex},TCI.IndexSet{MultiIndex}},
    localdim::Int
)
    return MultiIndex[[is..., j] for is in Iset, j in 1:localdim][:]
end

function kronecker(
    localdim::Int,
    Jset::Union{Vector{MultiIndex},TCI.IndexSet{MultiIndex}}
)
    return MultiIndex[[i, js...] for i in 1:localdim, js in Jset][:]
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

function separate_vertices(g::NamedGraph, edge::NamedEdge)
    has_edge(g, edge) || error("The edge is not in the graph.")
    return src(edge), dst(edge)
end

# いらないかも
function separate_vertices(g::NamedGraph, bondid::String, idbonds::Dict{String, NamedEdge})
    edge = idbonds[bondid]
    has_edge(g, edge) || error("The edge is not in the graph.")
    return src(edge), dst(edge)
end

function subregions(g::NamedGraph, parent::Int, children::Union{Int, Vector{Int}}) :: Vector{Int}
    if children isa Int
        children = [children]
    end
    grandchildren = []
    for child in children
        candidates = outneighbors(g, child)
        candidates = [cand for cand in candidates if cand != parent]
        append!(grandchildren, subregions(g, child, candidates))
        append!(grandchildren, [child])
    end
    sort!(grandchildren)
    return grandchildren
end

function distanceBFS(g::NamedGraph, parent::Int, children::Union{Int, Vector{Int}}, distances::Dict{String, Int}, idbonds::Dict{String, NamedEdge}) :: Dict{String, Int}
    for child in children
        parent_key = ""
        for (key, item) in idbonds
            if (src(item) == parent && dst(item) == child) || (src(item) == child && dst(item) == parent)
                parent_key = key
            end
        end

        candidates = outneighbors(g, child)
        candidates = [cand for cand in candidates if cand != parent]
        for cand in candidates
            for (key, item) in idbonds
                if (src(item) == child && dst(item) == cand) || (src(item) == cand && dst(item) == child)
                    distances[key] = distances[parent_key] + 1
                    break
                end
            end
        end
        distances = merge!(distances, distanceBFS(g, child, candidates, distances, idbonds))
    end
    return distances
end

function candidate_bondids(g::NamedGraph, parent::Int, child::Int, idbonds::Dict{String, NamedEdge}) :: Vector{String}
    candidates = []
    neighbors = outneighbors(g, child)
    neighbors = [cand for cand in neighbors if cand != parent]
    for cand in neighbors
        for (key, item) in idbonds
            if (src(item) == child && dst(item) == cand) || (src(item) == cand && dst(item) == child)
                push!(candidates, key)
                break
            end
        end
    end
    return candidates
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
