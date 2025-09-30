using Random
using LinearAlgebra
using TreeTCI: crossinterpolate, crossinterpolate_with_structuralsearch, crossinterpolate_with_3site_swapping
using NamedGraphs: NamedGraph, add_edge!
using QuanticsGrids
const QG = QuanticsGrids
using SparseIR
include("utils.jl")
include("graphs.jl")

ε(k) = 2*cos(k) + cos(5*k) + 2*cos(20*k)

gk(m::Int, kx::Float64; β::Float64=10.0) = begin
    m = 2 * m + 1
    ν = FermionicFreq(m)
    iν = SparseIR.valueim(ν, β)
    return 1 / (iν - ε(kx))
end

function gkb(b::Vector{Int}, n_m::Int, n_kx::Int; β::Float64=10.0, layout::Symbol=:block)
    @assert length(b) == n_m + n_kx
    parts = split_bits(b; group_bits=[n_m, n_kx], layout=layout)
    mb, kxb = parts
    mbit = length(mb)
    kxbit = length(kxb)
    Nm = 2^mbit
    Nkx = 2^kxbit
    im = frombins(mb)
    ikx = frombins(kxb)
    @assert im ≤ Nm
    @assert ikx ≤ Nkx
    kx = 2π * (ikx - 1)/Nkx
    m = (im - 1)
    return gk(m, kx)
end

function main()
    nkx_bit = 10
    nm_bit = 10
    localdims = fill(2, nkx_bit + nm_bit)
    f(v) = gkb(v, nm_bit, nkx_bit; layout=:block)
    g = graph_TT(nkx_bit + nm_bit)
    maxbonddim = 10
    kwargs = (maxbonddim = maxbonddim, maxiter = 2, tolerance = 1e-10)
    # ttn, ranks, errors = crossinterpolate(ComplexF64, f, localdims, g; kwargs...)
    
    ttn, ranks, errors = crossinterpolate_with_structuralsearch(ComplexF64, f, localdims, g, 2; kwargs...)
    # ttn_tree, ranks, errors = crossinterpolate_with_structuralsearch(ComplexF64, f, localdims, g, 2; kwargs...)
    # ttn_ttnopt, ranks, errors = crossinterpolate_with_3site_swapping(ComplexF64, f, localdims, g; kwargs...)
    
    # @show ttn.data_graph.underlying_graph
    # @show ttn_2site_swapping.data_graph.underlying_graph
    # @show ttn_3site_swapping.data_graph.underlying_graph
    
    # for _ in 1:10
    #     test_input = rand([1,2], nkx_bit + nm_bit)
    #     result = ttn(test_input)
    #     expected = f(test_input)
    # end

    @show ttn.tensornetwork.data_graph.underlying_graph
    @show last(ranks)
    @show last(errors)
    return 0
end

main()