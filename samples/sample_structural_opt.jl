using Test
using TreeTCI: SimpleTCI, optimize!, swap_2site!, add_subtree!
import NamedGraphs: NamedGraph, NamedEdge, add_edge!, vertices, edges, has_edge

function main()
    # make graph
    g = NamedGraph(10)
    add_edge!(g, 1, 2)
    add_edge!(g, 2, 3)
    add_edge!(g, 3, 4)
    add_edge!(g, 4, 5)
    add_edge!(g, 5, 6)
    add_edge!(g, 6, 7)
    add_edge!(g, 7, 8)
    add_edge!(g, 8, 9)
    add_edge!(g, 9, 10)

    localdims = fill(2, length(vertices(g)))
    f(v) = 1 / (1 + v' * v)
    kwargs = (maxbonddim = 20, maxiter = 10)
    tci = SimpleTCI{Float64}(f, localdims, g, [ones(Int, length(localdims))])
    ranks, errors = optimize!(tci, f; kwargs...)

    # swap_2site!(g, 1 => 3)
    add_subtree!(g, 6, NamedEdge(3 => 4))

end

main()
