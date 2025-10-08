using Test
using TreeTCI
import NamedGraphs: NamedGraph, NamedEdge, add_edge!, vertices, edges, has_edge

function f_pairwise(v::Vector{Int})
    s = 0.0
    for i in 1:2:length(v)-1
        s += v[i] * v[i+1]
    end
    return 1 / (1+s)
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

    localdims = fill(8, length(vertices(g)))
    f(v) = f_pairwise(v)
    kwargs = (maxbonddim = 64, maxiter = 10)
    ttn, ranks, errors = TreeTCI.crossinterpolate(Float64, f, localdims, g; kwargs...)

end

main()
