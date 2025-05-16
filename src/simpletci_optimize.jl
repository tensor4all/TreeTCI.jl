@doc raw"""
    optimize!(
        tci::SimpleTCI{ValueType}, f;
        tolerance::Union{Float64,Nothing} = nothing,
        maxbonddim::Int = typemax(Int),
        maxiter::Int = 20,
        sweepstrategy::AbstractSweep2sitePathProposer = DefaultSweep2sitePathProposer(),
        pivotstrategy::AbstractPivotCandidateProposer = DefaultPivotCandidateProposer(),
        verbosity::Int = 0,
        loginterval::Int = 10,
        normalizeerror::Bool = true,
        ncheckhistory::Int = 3,
    )

Optimize the SimpleTCI instance by iteratively updating pivots.

# Arguments
- `tci`: The SimpleTCI object to optimize
- `f`: The function to interpolate
- `tolerance::Union{Float64,Nothing} = nothing`: Error tolerance for convergence
- `maxbonddim::Int = typemax(Int)`: Maximum bond dimension
- `maxiter::Int = 20`: Maximum number of iterations
- `sweepstrategy::AbstractSweep2sitePathProposer = DefaultSweep2sitePathProposer()`: Strategy for sweeping
- `pivotstrategy::AbstractPivotCandidateProposer = DefaultPivotCandidateProposer()`: Strategy for proposing pivot candidates
- `verbosity::Int = 0`: Verbosity level
- `loginterval::Int = 10`: Interval for logging
- `normalizeerror::Bool = true`: Whether to normalize errors
- `ncheckhistory::Int = 3`: Number of history steps to check

# Returns
- `ranks`: Vector of ranks at each iteration
- `errors`: Vector of normalized errors at each iteration

# Note
- The SimpleTCI object will be modified in place.
- Set `tolerance` to be > 0 or `maxbonddim` to some reasonable value. Otherwise, convergence is not reachable.

"""
function optimize!(
    tci::SimpleTCI{ValueType},
    f;
    tolerance::Float64 = 1e-8,
    maxbonddim::Int = typemax(Int),
    maxiter::Int = 20,
    sweepstrategy::AbstractSweep2sitePathProposer = DefaultSweep2sitePathProposer(),
    pivotstrategy::AbstractPivotCandidateProposer = DefaultPivotCandidateProposer(),
    verbosity::Int = 0,
    loginterval::Int = 10,
    normalizeerror::Bool = true,
    ncheckhistory::Int = 3,
    ) where {ValueType}

    # Histories of properties for checking convergence.
    errors = Float64[]
    ranks = Int[]

    tstart = time_ns()

    if maxbonddim >= typemax(Int) && tolerance <= 0
        throw(
            ArgumentError(
                "Specify either tolerance > 0 or some maxbonddim; otherwise, the convergence criterion is not reachable!",
            ),
        )
    end

    for iter = 1:maxiter
        errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
        abstol = tolerance * errornormalization

        if verbosity > 1
            println("  Walltime $(1e-9*(time_ns() - tstart)) sec: starting 2site sweep")
            flush(stdout)
        end

        sweep2site!(
            tci,
            f;
            abstol = abstol,
            maxbonddim = maxbonddim,
            verbosity = verbosity,
            sweepstrategy = sweepstrategy,
            pivotstrategy = pivotstrategy,
        )

        push!(ranks, rank(tci))
        push!(errors, pivoterror(tci))

        if convergencecriterion(
            ranks,
            errors,
            maxbonddim,
            tolerance,
            ncheckhistory
        )
            println("Converged at $(iter)th-sweep.")
            tci.converged_IJset = deepcopy(tci.IJset)
            break
        end
    end

    errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
    return ranks, errors ./ errornormalization
end

"""
 Perform 2site sweeps on a SimpleTCI.
"""
function sweep2site!(
    tci::SimpleTCI{ValueType},
    f;
    abstol::Float64 = 1e-8,
    maxbonddim::Int = typemax(Int),
    sweepstrategy::AbstractSweep2sitePathProposer = DefaultSweep2sitePathProposer(),
    pivotstrategy::AbstractPivotCandidateProposer = DefaultPivotCandidateProposer(),
    verbosity::Int = 0,
) where {ValueType}

    edge_path = generate_sweep2site_path(sweepstrategy, tci)

    flushpivoterror!(tci)

    for edge in edge_path
        updatepivots!(
            tci,
            edge,
            f;
            abstol = abstol,
            maxbonddim = maxbonddim,
            pivotstrategy = pivotstrategy,
            verbosity = verbosity,
        )
    end

    nothing
end

"""
 Update pivots at bond of tci object.
"""
function updatepivots!(
    tci::SimpleTCI{ValueType},
    edge::NamedEdge,
    f::F;
    reltol::Float64 = 1e-14,
    abstol::Float64 = 0.0,
    maxbonddim::Int = typemax(Int),
    pivotstrategy::AbstractPivotCandidateProposer = DefaultPivotCandidateProposer(),
    verbosity::Int = 0,
) where {F,ValueType}

    N = length(tci.localdims)

    combinedIJset = generate_pivot_candidates(pivotstrategy, tci, edge)
    keys_array = collect(keys(combinedIJset))
    Ikey, Jkey = first(keys_array), last(keys_array)

    t1 = time_ns()
    Pi = reshape(
        filltensor(ValueType, f, tci.localdims, combinedIJset, [Ikey], [Jkey], Val(0)),
        length(combinedIJset[Ikey]),
        length(combinedIJset[Jkey]),
    )
    t2 = time_ns()

    updatemaxsample!(tci, Pi)

    luci = TCI.MatrixLUCI(Pi, reltol = reltol, abstol = abstol, maxrank = maxbonddim)

    t3 = time_ns()
    if verbosity > 2
        x, y = length(combinedIJset[Ikey]),
        length(combinedIJset[Jkey]),
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

function flushpivoterror!(tci::SimpleTCI{ValueType}) where {ValueType}
    tci.pivoterrors = Float64[]
    nothing
end

function updateedgeerror!(tci::SimpleTCI{T}, edge::NamedEdge, error::Float64) where {T}
    tci.bonderrors[edge] = error
    nothing
end

function updatepivoterror!(tci::SimpleTCI{T}, errors::AbstractVector{Float64}) where {T}
    erroriter = Iterators.map(max, TCI.padzero(tci.pivoterrors), TCI.padzero(errors))
    tci.pivoterrors =
        Iterators.take(erroriter, max(length(tci.pivoterrors), length(errors))) |> collect
    nothing
end

function rank(tci::SimpleTCI{ValueType}) where {ValueType}
    return maximum(length(IJset) for IJset in values(tci.IJset))
end

function pivoterror(tci::SimpleTCI{T}) where {T}
    return maxbonderror(tci)
end

function maxbonderror(tci::SimpleTCI{T}) where {T}
    return maximum(values(tci.bonderrors))
end

function convergencecriterion(
    ranks::AbstractVector{Int},
    errors::AbstractVector{Float64},
    maxbonddim::Int,
    tolerance::Float64,
    ncheckhistory::Int,
)::Bool
    if length(errors) < ncheckhistory
        return false
    end
    lastranks = last(ranks, ncheckhistory)
    return (
        all(last(errors, ncheckhistory) .< tolerance) &&
        minimum(lastranks) == lastranks[end]
    ) || all(lastranks .>= maxbonddim)
end