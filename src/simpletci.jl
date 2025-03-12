MultiIndex = Vector{Int}
SubTreeVertex = Vector{Int}

using Base: SimpleLogger

mutable struct SimpleTCI{ValueType}
    IJset::Dict{SubTreeVertex,Vector{MultiIndex}}
    localdims::Vector{Int}
    g::NamedGraph
    #"Error estimate per bond by 2site sweep."
    bonderrors::Dict{NamedEdge,Float64} # key is the bond id
    # "Error estimate for backtruncation of bonds."
    pivoterrors::Vector{Float64} # key is the bond id
    #"Maximum sample for error normalization."
    maxsamplevalue::Float64
    IJset_history::Vector{Dict{SubTreeVertex,Vector{MultiIndex}}}

    function SimpleTCI{ValueType}(localdims::Vector{Int}, g::NamedGraph) where {ValueType}
        length(localdims) > 1 || error("localdims should have at least 2 elements!")
        n = length(localdims)

        # assign the key for each bond
        bonderrors = Dict(e => 0.0 for e in edges(g))

        !Graphs.is_cyclic(g) ||
            error("TreeTensorNetwork is not supported for loopy tensor network.")

        new{ValueType}(
            Dict{SubTreeVertex,Vector{MultiIndex}}(),               # IJset
            localdims,
            g,
            bonderrors,
            Float64[],
            0.0,                                                   # maxsamplevalue
            Vector{Dict{SubTreeVertex,Vector{MultiIndex}}}(),       # IJset_history
        )
    end
end

function SimpleTCI{ValueType}(
    func::F,
    localdims::Vector{Int},
    g::NamedGraph,
    initialpivots::Vector{MultiIndex} = [ones(Int, length(localdims))],
) where {F,ValueType}
    tci = SimpleTCI{ValueType}(localdims, g)
    addglobalpivots!(tci, initialpivots)
    tci.maxsamplevalue = maximum(abs, (func(x) for x in initialpivots))
    abs(tci.maxsamplevalue) > 0 ||
        error("The function should not be zero at the initial pivot.")
    return tci
end

@doc"""
Add global pivots to index sets
"""
function addglobalpivots!(
    tci::SimpleTCI{ValueType},
    pivots::Vector{MultiIndex},
) where {ValueType}
    if any(length(tci.localdims) .!= length.(pivots)) # AbstructTreeTensorNetworkをから引き継ぎlength(tci)ができると良い
        throw(DimensionMismatch("Please specify a pivot as one index per leg of the TTN."))
    end
    for pivot in pivots
        for e in edges(tci.g)
            p, q = separatevertices(tci.g, e)
            Iset_key = subtreevertices(tci.g, p => q)
            Jset_key = subtreevertices(tci.g, q => p)

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

    tci.IJset[[i for i = 1:length(tci.localdims)]] = Int[]

    nothing
end

@doc"""
    optimize!(tci::SimpleTCI{ValueType}, f; kwargs...)

Optimize the tensor cross interpolation (TCI) by iteratively updating pivots.

# Arguments
- `tci`: The SimpleTCI object to optimize
- `f`: The function to interpolate

# Keywords
- `tolerance::Union{Float64,Nothing} = nothing`: Error tolerance for convergence
- `pivottolerance::Union{Float64,Nothing} = nothing`: Deprecated, use tolerance instead
- `maxbonddim::Int = typemax(Int)`: Maximum bond dimension
- `maxiter::Int = 20`: Maximum number of iterations
- `sweepstrategy::Symbol = :backandforth`: Strategy for sweeping
- `pivotsearch::Symbol = :full`: Strategy for pivot search
- `verbosity::Int = 0`: Verbosity level
- `loginterval::Int = 10`: Interval for logging
- `normalizeerror::Bool = true`: Whether to normalize errors
- `ncheckhistory::Int = 3`: Number of history steps to check
- `maxnglobalpivot::Int = 5`: Maximum number of global pivots
- `nsearchglobalpivot::Int = 5`: Number of global pivots to search
- `tolmarginglobalsearch::Float64 = 10.0`: Tolerance margin for global search
- `strictlynested::Bool = false`: Whether to enforce strict nesting
- `checkbatchevaluatable::Bool = false`: Whether to check if function is batch evaluatable

# Returns
- `ranks`: Vector of ranks at each iteration
- `errors`: Vector of normalized errors at each iteration
"""
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

@doc"""
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
) where {ValueType}

    edge_path = generate_sweep2site_path(DefaultSweep2sitePathProper(), tci)


    for iter = iter1:iter1+niter-1
        extraIJset = Dict(key => MultiIndex[] for key in keys(tci.IJset))
        if length(tci.IJset_history) > 0
            extraIJset = tci.IJset_history[end]
        end

        push!(tci.IJset_history, deepcopy(tci.IJset))

        flushpivoterror!(tci)

        for edge in edge_path
            updatepivots!(
                tci,
                edge,
                f;
                abstol = abstol,
                maxbonddim = maxbonddim,
                verbosity = verbosity,
                extraIJset = extraIJset,
            )
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
    edge::NamedEdge,
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

        N = length(tci.localdims)

        (IJkey, combinedIJset) = generate_pivot_candidates(
            DefaultPivotCandidateProper(),
            tci,
            edge,
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

        updateerrors!(tci, edge, TCI.pivoterrors(luci))
        nothing
    end


function updatemaxsample!(tci::SimpleTCI{V}, samples::Array{V}) where {V}
    tci.maxsamplevalue = TCI.maxabs(tci.maxsamplevalue, samples)
end

function updateerrors!(
    tci::SimpleTCI{T},
    edge::NamedEdge,
    errors::AbstractVector{Float64},
) where {T}
    updateedgeerror!(tci, edge, last(errors))
    updatepivoterror!(tci, errors)
    nothing
end

function updateedgeerror!(
    tci::SimpleTCI{T},
    edge::NamedEdge,
    error::Float64,
) where {T}
    tci.bonderrors[edge] = error
    nothing
end

function updatepivoterror!(tci::SimpleTCI{T}, errors::AbstractVector{Float64}) where {T}
    erroriter = Iterators.map(max, TCI.padzero(tci.pivoterrors), TCI.padzero(errors))
    tci.pivoterrors =
        Iterators.take(erroriter, max(length(tci.pivoterrors), length(errors))) |> collect
    nothing
end

function pivoterror(tci::SimpleTCI{T}) where {T}
    return maxbonderror(tci)
end

function maxbonderror(tci::SimpleTCI{T}) where {T}
    return maximum(values(tci.bonderrors))
end

"""
Return if site tensors are available
"""

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
