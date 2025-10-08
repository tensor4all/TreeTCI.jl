using Test
using TreeTCI
import NamedGraphs: NamedGraph, NamedEdge, add_edge!, vertices, edges, has_edge

function main()
    # make graph
    g = NamedGraph(10)
    add_edge!(g, 1, 3)
    add_edge!(g, 2, 3)
    add_edge!(g, 3, 5)

    add_edge!(g, 4, 5)
    add_edge!(g, 5, 7)
    add_edge!(g, 6, 7)
    add_edge!(g, 7, 8)
    add_edge!(g, 8, 9)
    add_edge!(g, 8, 10)

    localdims = fill(2, length(vertices(g)))
    f(v) = 1 / (1 + v' * v)
    kwargs = (maxbonddim = 20, maxiter = 10)
    ttn, ranks, errors = TreeTCI.crossinterpolate(Float64, f, localdims, g; kwargs...)
    @show ttn([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), f([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    @show ttn([1, 2, 1, 2, 1, 2, 1, 2, 1, 2]), f([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
end

main()
