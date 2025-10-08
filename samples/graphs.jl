using NamedGraphs: NamedGraph, add_edge!

function graph_TT(R::Int)
    g = NamedGraph(R)
    for i in 1:R-1
        add_edge!(g, i, i+1)
    end
    return g
end