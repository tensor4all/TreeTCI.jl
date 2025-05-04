
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
    namedgraph_dijkstra_shortest_paths,
    neighbors,
    add_edge!,
    rem_edge!
import TensorCrossInterpolation as TCI
import SimpleTensorNetworks:
    TensorNetwork, IndexedArray, Index, complete_contraction, getindex, contract
import Random: shuffle
import LinearAlgebra: svd
include("treegraph_utils.jl")
include("simpletci.jl")
include("pivotcandidateproposer.jl")
include("sweep2sitepathproposer.jl")
include("possiblesubtreesproposer.jl")
include("optimalstructureproposer.jl")
include("simpletci_optimize.jl")
include("simpletci_tensors.jl")
include("treetensornetwork.jl")
end
