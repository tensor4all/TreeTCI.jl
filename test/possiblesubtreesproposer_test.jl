
@testitem "PossibleSubtreesProposer" begin
    using Test
    using TreeTCI: SimpleTCI, optimize!, generate_possible_subtrees
    import NamedGraphs: NamedGraph, NamedEdge, add_edge!, vertices, edges, has_edge

    # make graph
    g = NamedGraph(5)
    add_edge!(g, 1, 2)
    add_edge!(g, 2, 3)
    add_edge!(g, 3, 4)
    add_edge!(g, 4, 5)
    max_degree = 3

    localdims = fill(2, length(vertices(g)))
    f(v) = 1 / (1 + v' * v)
    kwargs = (
        maxbonddim = 5,
        maxiter = 10,
        pivotstrategy = TreeTCI.SimplePivotCandidateProposer(),
    )
    tci = SimpleTCI{Float64}(f, localdims, g)
    ranks, errors = optimize!(tci, f; kwargs...)
    proposer = TreeTCI.DefaultPossibleSubtreesProposer()
    edge = NamedEdge(2, 3)
    splits = TreeTCI.generate_possible_subtrees(proposer, tci, edge, max_degree)
    @test splits == [
        [] => [[1], [4, 5]],
        [[1]] => [[4, 5]],
        [[4, 5]] => [[1]],
        [[1], [4, 5]] => [],
    ]
end
