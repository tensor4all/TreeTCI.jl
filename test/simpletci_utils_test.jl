using Test
using TreeTCI
import NamedGraphs: NamedGraph, NamedEdge, add_edge!, edges, has_edge

@testset "simpletci.jl" begin
    # make graph
    g = NamedGraph(7)
    add_edge!(g, 1, 2)
    add_edge!(g, 2, 3)
    add_edge!(g, 2, 4)
    add_edge!(g, 4, 5)
    add_edge!(g, 5, 6)
    add_edge!(g, 5, 7)


    @testset "SubTreeVertex" begin
        e = NamedEdge(2 => 4)
        v1, v2 = TreeTCI.separate_vertices(g, e)
        @test v1 == 2
        @test v2 == 4

        Ivertices = TreeTCI.subtree_vertices(g, v2, v1) # 4 -> 2
        Jvertices = TreeTCI.subtree_vertices(g, v1, v2) # 2 -> 4

        @test Ivertices == [1, 2, 3]
        @test Jvertices == [4, 5, 6, 7]

        subregions = TreeTCI.subregion_vertices(g, e)
        @test first(subregions) == [1, 2, 3]
        @test last(subregions) == [4, 5, 6, 7]
    end

end




