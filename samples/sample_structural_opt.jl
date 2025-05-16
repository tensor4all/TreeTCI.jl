using Test
using TreeTCI: crossinterpolate_adaptivetree, crossinterpolate
using NamedGraphs: NamedGraph, add_edge!, vertices
using Random

function evaluate_error_sampled(f, ttn1, ttn2, localdims::Vector{Int}, nsamples::Int; rng=Random.default_rng())
    total_error1 = 0.0
    total_error2 = 0.0

    for _ in 1:nsamples
        xvec = [rand(rng, 1:d) for d in localdims]
        fx = f(xvec)
        total_error1 += abs(fx - ttn1(xvec))
        total_error2 += abs(fx - ttn2(xvec))
    end

    mean_error1 = total_error1 / nsamples
    mean_error2 = total_error2 / nsamples

    println("Sampled mean |f(x) - original(x)| = ", mean_error1)
    println("Sampled mean |f(x) - optimized(x)| = ", mean_error2)

    return mean_error1, mean_error2
end


function main()
    # make graph
    g = NamedGraph(8)
    add_edge!(g, 1, 2)
    add_edge!(g, 2, 3)
    add_edge!(g, 3, 4)
    add_edge!(g, 4, 5)
    add_edge!(g, 5, 6)
    add_edge!(g, 6, 7)
    add_edge!(g, 7, 8)

    localdims = fill(10, length(vertices(g)))
    f(v) = 1 / (1 + v' * v)
    kwargs = (maxbonddim = 20, maxiter = 10)
    tt, ranks, errors = crossinterpolate(Float64, f, localdims, g; kwargs...)
    optimized_tt, ranks, errors = crossinterpolate_adaptivetree(Float64, f, localdims, g, 20; kwargs...)

    evaluate_error_sampled(f, tt, optimized_tt, localdims, 1000)
    return 0
end

main()
