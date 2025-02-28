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


end

main()
