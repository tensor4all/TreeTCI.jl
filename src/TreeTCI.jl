
module TreeTCI
using UUIDs
import Graphs
import NamedGraphs:
    NamedGraph, NamedEdge, is_directed, outneighbors, has_edge, edges, vertices, src, dst
import TensorCrossInterpolation as TCI
import SimpleTensorNetworks: TensorNetwork, IndexedArray, Index, complete_contraction
include("simpletci_utils.jl")
include("abstracttreetensornetwork.jl")
include("treetensornetwork.jl")
include("simpletci.jl")
include("crossinterpolate.jl")
include("pivotcandidateproper.jl")
end
