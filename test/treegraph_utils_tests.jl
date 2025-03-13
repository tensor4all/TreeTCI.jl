@testitem "TreeGraphUtils" begin
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

    e = NamedEdge(2 => 4)
    v1, v2 = TreeTCI.separatevertices(g, e)
    @test v1 == 2
    @test v2 == 4

    Ivertices = TreeTCI.subtreevertices(g, v2 => v1) # 4 -> 2
    Jvertices = TreeTCI.subtreevertices(g, v1 => v2) # 2 -> 4

    @test Ivertices == [1, 2, 3]
    @test Jvertices == [4, 5, 6, 7]

    subregions = TreeTCI.subregionvertices(g, e)
    @test first(subregions) == [1, 2, 3]
    @test last(subregions) == [4, 5, 6, 7]

    @test Set(TreeTCI.adjacentedges(g, 4)) == Set([NamedEdge(2 => 4), NamedEdge(4 => 5)])

    @test Set(TreeTCI.candidateedges(g, NamedEdge(2 => 4))) ==
          Set([NamedEdge(1 => 2), NamedEdge(2 => 3), NamedEdge(4 => 5)])

    @test TreeTCI.distanceedges(g, NamedEdge(2 => 4)) == Dict(
        NamedEdge(2 => 4) => 0,
        NamedEdge(1 => 2) => 1,
        NamedEdge(2 => 3) => 1,
        NamedEdge(4 => 5) => 1,
        NamedEdge(5 => 6) => 2,
        NamedEdge(5 => 7) => 2,
    )
end
