
module TreeTCI
using UUIDs
import Graphs
import NamedGraphs:
    NamedGraph, NamedEdge, is_directed, outneighbors, has_edge, edges, vertices, src, dst
import TensorCrossInterpolation as TCI
import SimpleTensorNetworks: TensorNetwork, IndexedArray, Index
include("treetensornetwork.jl")
include("simpletci_utils.jl")
include("pivotcandidateproper.jl")
include("simpletci.jl")
end
