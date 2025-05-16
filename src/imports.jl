using Random
using Graphs: simplecycles_limited_length, has_edge, SimpleGraph, center, steiner_tree
using NamedGraphs:
    NamedGraph,
    NamedEdge,
    is_cyclic,
    is_directed,
    neighbors,
    outneighbors,
    has_edge,
    edges,
    vertices,
    namedgraph_dijkstra_shortest_paths
using NamedGraphs.GraphsExtensions:
    src,
    dst,
    is_connected,
    degree,
    add_vertices!, add_vertex!, rem_vertices!, rem_vertex!,
    rem_edge!, add_edge!
import TensorCrossInterpolation as TCI
import SimpleTensorNetworks: TensorNetwork, IndexedArray, Index, complete_contraction, getindex, contract
