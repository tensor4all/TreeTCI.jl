using Random
using LinearAlgebra
using TreeTCI: crossinterpolate_with_3site_swapping
using Statistics
using Distributions
using NamedGraphs: NamedGraph, add_edge!, edges
using QuanticsGrids
const QG = QuanticsGrids


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

    kwargs = (
            maxbonddim = 10,
            tolerance = 1e-10,
            maxiter = 100,
        )

    tci = crossinterpolate_with_3site_swapping(Float64, fq, localdims, g; kwargs...)

    return tci
end

main()