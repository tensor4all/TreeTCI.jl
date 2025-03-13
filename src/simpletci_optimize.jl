@doc """
     optimize!(tci::SimpleTCI{ValueType}, f; kwargs...)

 Optimize the tensor cross interpolation (TCI) by iteratively updating pivots.

 # Arguments
 - `tci`: The SimpleTCI object to optimize
 - `f`: The function to interpolate

 # Keywords
 - `tolerance::Union{Float64,Nothing} = nothing`: Error tolerance for convergence
 - `maxbonddim::Int = typemax(Int)`: Maximum bond dimension
 - `maxiter::Int = 20`: Maximum number of iterations
 - `sweepstrategy::AbstractSweep2sitePathProper = DefaultSweep2sitePathProper()`: Strategy for sweeping
 - `verbosity::Int = 0`: Verbosity level
 - `loginterval::Int = 10`: Interval for logging
 - `normalizeerror::Bool = true`: Whether to normalize errors
 - `ncheckhistory::Int = 3`: Number of history steps to check

 # Returns
 - `ranks`: Vector of ranks at each iteration
 - `errors`: Vector of normalized errors at each iteration
 """
function optimize!(
    tci::SimpleTCI{ValueType},
    f;
    tolerance::Float64 = 1e-8,
    maxbonddim::Int = typemax(Int),
    maxiter::Int = 20,
    sweepstrategy::AbstractSweep2sitePathProper = DefaultSweep2sitePathProper(),
    verbosity::Int = 0,
    loginterval::Int = 10,
    normalizeerror::Bool = true,
    ncheckhistory::Int = 3,
) where {ValueType}
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

    globalpivots = MultiIndex[]
    for iter = 1:maxiter
        errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
        abstol = tolerance * errornormalization

        if verbosity > 1
            println("  Walltime $(1e-9*(time_ns() - tstart)) sec: starting 2site sweep")
            flush(stdout)
        end

        sweep2site!(
            tci,
            f,
            2;
            abstol = abstol,
            maxbonddim = maxbonddim,
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

@doc """
 Perform 2site sweeps on a SimpleTCI.
 !TODO: Implement for Tree structure

 """
function sweep2site!(
    tci::SimpleTCI{ValueType},
    f,
    niter::Int;
    abstol::Float64 = 1e-8,
    maxbonddim::Int = typemax(Int),
    sweepstrategy::AbstractSweep2sitePathProper = DefaultSweep2sitePathProper(),
    verbosity::Int = 0,
) where {ValueType}

    edge_path = generate_sweep2site_path(sweepstrategy, tci)

    for _ in 1:niter
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

    (IJkey, combinedIJset) =
        generate_pivot_candidates(DefaultPivotCandidateProper(), tci, edge, extraIJset)
    Ikey, Jkey = first(IJkey), last(IJkey)

    t1 = time_ns()
    Pi = reshape(
        filltensor(ValueType, f, tci.localdims, combinedIJset, [Ikey], [Jkey], Val(0)),
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

function pivoterror(tci::SimpleTCI{T}) where {T}
    return maxbonderror(tci)
end

function maxbonderror(tci::SimpleTCI{T}) where {T}
    return maximum(values(tci.bonderrors))
end
