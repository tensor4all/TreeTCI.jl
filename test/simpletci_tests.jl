@testitem "SimpleTCI" begin
    using Test
    using TreeTCI
    import NamedGraphs: NamedGraph, NamedEdge, add_edge!, vertices, edges, has_edge

    # make graph
    g = NamedGraph(7)
    add_edge!(g, 1, 2)
    add_edge!(g, 2, 3)
    add_edge!(g, 2, 4)
    add_edge!(g, 4, 5)
    add_edge!(g, 5, 6)
    add_edge!(g, 5, 7)

    localdims = fill(2, length(vertices(g)))
    f(v) = 1 / (1 + v' * v)
    kwargs = (
        maxbonddim = 5,
        maxiter = 10,
        pivotstrategy = TreeTCI.SimplePivotCandidateProposer(),
    )
    ttn, ranks, errors = TreeTCI.crossinterpolate(Float64, f, localdims, g; kwargs...)
    @test ttn([1, 1, 1, 1, 1, 1, 1]) ≈ f([1, 1, 1, 1, 1, 1, 1])
    @test ttn([1, 1, 1, 1, 2, 2, 2]) ≈ f([1, 1, 1, 1, 2, 2, 2])
    @test ttn([2, 2, 2, 2, 1, 1, 1]) ≈ f([2, 2, 2, 2, 1, 1, 1])
    @test ttn([1, 2, 1, 2, 1, 2, 1]) ≈ f([1, 2, 1, 2, 1, 2, 1])
    @test ttn([2, 1, 2, 1, 2, 1, 2]) ≈ f([2, 1, 2, 1, 2, 1, 2])
    @test ttn([2, 2, 2, 2, 2, 2, 2]) ≈ f([2, 2, 2, 2, 2, 2, 2])
end

@testitem "optimize! stops on convergence and records the rank history (#17)" begin
    using Test
    using TreeTCI
    import NamedGraphs: NamedGraph, NamedEdge, add_edge!, vertices, edges, has_edge

    g = NamedGraph(7)
    for (a, b) in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
        add_edge!(g, a => b)
    end
    localdims = fill(2, 7)
    f(v) = 1 / (1 + v' * v)

    # `ncheckhistory` is the earliest sweep the criterion can fire on: a tolerance every error
    # satisfies stops the loop after exactly that many sweeps, and each sweep contributes one
    # entry to both returned vectors (the rank vector used to stay empty).
    for ncheckhistory in (3, 5)
        _, ranks, errors = TreeTCI.crossinterpolate(
            Float64,
            f,
            localdims,
            g;
            maxbonddim = 5,
            maxiter = 50,
            tolerance = 1e9,
            ncheckhistory,
        )
        @test length(ranks) == ncheckhistory
        @test length(errors) == ncheckhistory
        @test all(ranks .== 1)
        @test all(isapprox.(errors, 0.04250295159386064; rtol = 1e-12))
    end

    # With a real tolerance the loop runs until the errors and the ranks have settled instead of
    # always running `maxiter` sweeps.
    maxiter = 50
    _, ranks, errors = TreeTCI.crossinterpolate(
        Float64,
        f,
        localdims,
        g;
        maxbonddim = 5,
        maxiter,
        tolerance = 1e-8,
    )
    @test length(ranks) == length(errors)
    @test 0 < length(ranks) < maxiter
    @test maximum(ranks) <= 5
    @test last(errors) < 1e-8
end
