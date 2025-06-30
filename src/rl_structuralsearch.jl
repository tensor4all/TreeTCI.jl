using ReinforcementLearning
const RLBase = ReinforcementLearningBase

abstract type AbstractAction end

struct LocalSwapAction  <: AbstractAction
    edge::NamedEdge
end

struct GlobalSwapAction <: AbstractAction
    pair::Pair{Int,Int}
end

mutable struct TCIEnv{ValueType} <: AbstractEnv
    f
    localdims::Vector{Int}
    g::NamedGraph
    initialpivots::Vector{MultiIndex}
    maxstep::Int
    tol::Float64
    kwargs

    # runtime variables
    tci::SimpleTCI{ValueType}
    step::Int
    reward::Float64
end

# コンストラクタを修正
function TCIEnv{ValueType}(
    f,
    localdims::Vector{Int},
    g::NamedGraph;
    initialpivots::Vector{MultiIndex} = [ones(Int, length(localdims))],
    maxstep::Int = 20,
    tol::Float64 = 1e-8,
    kwargs = (maxbonddim=typemax(Int), tolerance=nothing, maxiter=20)
) where {ValueType}
    # SimpleTCIの初期化を修正
    tci0 = SimpleTCI{ValueType}(f, localdims, g, initialpivots)
    return TCIEnv{ValueType}(f, localdims, g, initialpivots, maxstep, tol, kwargs,
                            tci0, 0, -Inf)
end

function RLBase.state_space(::TCIEnv{ValueType}) where {ValueType}
    return nothing
end

function RLBase.state(env::TCIEnv{ValueType}, _obs, _player) where {ValueType}
    errors = collect(values(env.tci.bonderrors))

    max_err = maximum(errors)
    mean_err = mean(errors)

    return [max_err, mean_err]
end

RLBase.is_terminated(env::TCIEnv{ValueType}) where {ValueType} =
    env.step ≥ env.maxstep || maximum(values(env.tci.bonderrors)) < env.tol

RLBase.reward(env::TCIEnv{ValueType}) where {ValueType} = env.reward

function RLBase.reset!(env::TCIEnv{ValueType}) where {ValueType}
    env.step = 0
    env.tci = SimpleTCI{ValueType}(env.f, env.localdims, env.g, env.initialpivots)
    _, errs = TreeTCI.optimize!(env.tci, env.f; env.kwargs...)
    env.reward = -maximum(errs)
    return RLBase.state(env, nothing, nothing)
end


function classify_pattern(tci::SimpleTCI, tol::Float64)
    err_max, e_max = findmax(tci.bonderrors)
    err_max < tol && return :none, nothing

    p, q = src(e_max), dst(e_max)
    C = Set{Int}()
    stack = [p, q]
    while !isempty(stack)
        v = pop!(stack)
        if v in C
            continue
        end
        push!(C, v)
        for nb in outneighbors(tci.g, v)
            e = NamedEdge(v=>nb)
            if get(tci.bonderrors, e, 0.0) ≥ tol
                push!(stack, nb)
            end
            e = NamedEdge(nb=>v)
            if get(tci.bonderrors, e, 0.0) ≥ tol
                push!(stack, nb)
            end
        end
    end
    length(C) > 1 ? (:cluster, collect(C)) : (:peak, e_max)
end

function RLBase.action_space(env::TCIEnv{ValueType}) where {ValueType}
    pat, info = classify_pattern(env.tci, env.tol)
    acts = Vector{AbstractAction}()

    if pat == :cluster
        C = Set(info)
        outer = [nb for v in C for nb in outneighbors(env.tci.g,v) if nb ∉ C]

        for v in C
            for nb in outneighbors(env.tci.g,v)
                if nb ∉ C
                    if NamedEdge(v=>nb) in edges(env.tci.g)
                        push!(acts, LocalSwapAction(NamedEdge(v=>nb)))
                    elseif NamedEdge(nb=>v) in edges(env.tci.g)
                        push!(acts, LocalSwapAction(NamedEdge(nb=>v)))
                    else
                        error("Edge between $v and $nb not found in graph")
                    end
                end
            end
        end
        for v in C, leaf in outer
            v ≠ leaf && push!(acts, GlobalSwapAction(v=>leaf))
        end

    elseif pat == :peak
        push!(acts, LocalSwapAction(info))
    end
    acts
end

function RLBase.act!(env::TCIEnv{ValueType}, act::AbstractAction) where {ValueType}
    current_err = maximum(values(env.tci.bonderrors))

    if act isa LocalSwapAction
        g_new = generate_new_structure(NewStructureLocalSwap(), env.tci, act.edge)
    elseif act isa GlobalSwapAction
        g_new = generate_new_structure(NewStructureGlobalSwap(), env.tci, act.pair)
    else
        error("Unknown action")
    end

    tci_new = SimpleTCI{ValueType}(env.f, env.localdims, g_new, env.initialpivots)
    tci_new.converged_IJset = env.tci.converged_IJset
    _, errs = TreeTCI.optimize!(tci_new, env.f; env.kwargs...)
    new_err = maximum(errs)

    env.reward = current_err - new_err
    env.tci, env.step = tci_new, env.step + 1
    return RLBase.state(env, nothing, nothing), env.reward, RLBase.is_terminated(env), nothing
end