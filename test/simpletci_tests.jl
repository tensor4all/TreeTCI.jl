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
    kwargs = (maxbonddim = 5, maxiter = 10, pivotstrategy = SimplePivotCandidateProposer())
    ttn, ranks, errors = TreeTCI.crossinterpolate(Float64, f, localdims, g; kwargs...)
    @test ttn([1, 1, 1, 1, 1, 1, 1]) ≈ f([1, 1, 1, 1, 1, 1, 1])
    @test ttn([1, 1, 1, 1, 2, 2, 2]) ≈ f([1, 1, 1, 1, 2, 2, 2])
    @test ttn([2, 2, 2, 2, 1, 1, 1]) ≈ f([2, 2, 2, 2, 1, 1, 1])
    @test ttn([1, 2, 1, 2, 1, 2, 1]) ≈ f([1, 2, 1, 2, 1, 2, 1])
    @test ttn([2, 1, 2, 1, 2, 1, 2]) ≈ f([2, 1, 2, 1, 2, 1, 2])
    @test ttn([2, 2, 2, 2, 2, 2, 2]) ≈ f([2, 2, 2, 2, 2, 2, 2])
end
