using Random
using LinearAlgebra
using TreeTCI: TCIEnv, SimpleTCI
using Statistics
using Distributions
using NamedGraphs: NamedGraph, add_edge!, edges
using QuanticsGrids
const QG = QuanticsGrids
using ReinforcementLearning


function build_problem()
    Random.seed!(1234)
    R = 4
    μ = zeros(3)
    Σ = rand(LKJ(3, 50.0))
    dist = MvNormal(μ, Hermitian(Σ))
    f(x,y,z) = pdf(dist, [x,y,z])

    grid = QG.DiscretizedGrid{3}(R, (-5,-5,-5), (5,5,5); unfoldingscheme=:interleaved)
    fq = QG.quanticsfunction(Float64, grid, f)

    nsites = 3R
    g = NamedGraph(nsites)
    for i in 1:(nsites-1)
        add_edge!(g, i, i+1)
    end
    localdims = fill(grid.base, nsites)

    return fq, localdims, g
end

function main()

    fq, localdims, g = build_problem()

    env = TCIEnv{Float64}(
        fq,
        localdims,
        g;
        maxstep = 100,
        tol = 1e-8,
        kwargs = (maxbonddim=10, maxiter=10, tolerance=1e-12)
    )

    RLBase.reset!(env)

    println("学習前 最大誤差: ", maximum(RLBase.state(env, nothing, nothing)))

    println("学習開始...")
    history = run(
        RandomPolicy(),
        env,
        StopAfterNEpisodes(10)
    )

    # 結果確認
    println("学習後 最大誤差: ", maximum(RLBase.state(env, nothing, nothing)))
end

main()