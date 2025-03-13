using Revise
using TreeTCI
using NamedGraphs: NamedGraph, add_edge!, edges

function main()
    localdims = fill(2, 7)
    g = NamedGraph(7)
    add_edge!(g, 1, 2)
    add_edge!(g, 2, 3)
    add_edge!(g, 2, 4)
    add_edge!(g, 4, 5)
    add_edge!(g, 5, 6)
    add_edge!(g, 5, 7)

    f(v) = 1 / (1 + v' * v)
    tolerance = 1e-8

    mpn, ranks, errors =
        TreeTCI.TCI.crossinterpolate2(Float64, f, localdims; tolerance = tolerance)
    ttn, ranks, errors = TreeTCI.crossinterpolate(Float64, f, localdims, g)
    @show f([1, 1, 1, 1, 2, 1, 1]), f([1, 2, 1, 2, 2, 1, 1]), f([2, 2, 2, 2, 2, 2, 2])
    @show mpn([1, 1, 1, 1, 2, 1, 1]), mpn([1, 2, 1, 2, 2, 1, 1]), mpn([2, 2, 2, 2, 2, 2, 2])
    @show ttn([1, 1, 1, 1, 2, 1, 1]), ttn([1, 2, 1, 2, 2, 1, 1]), ttn([2, 2, 2, 2, 2, 2, 2])
    nothing
end

main()
