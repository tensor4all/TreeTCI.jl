
module TreeTCI
import Graphs
import NamedGraphs:
    NamedGraph,
    NamedEdge,
    is_directed,
    outneighbors,
    has_edge,
    edges,
    vertices,
    src,
    dst,
    namedgraph_dijkstra_shortest_paths
import TensorCrossInterpolation as TCI
import SimpleTensorNetworks:
    TensorNetwork, IndexedArray, Index, complete_contraction, getindex, contract
import Random: shuffle
include("treegraph_utils.jl")
include("simpletci.jl")
include("simpletci_utils.jl")
include("pivotcandidateproper.jl")
include("sweep2sitepathproper.jl")
include("treetensornetwork.jl")
end
