using Random
using LinearAlgebra
using TreeTCI: crossinterpolate, crossinterpolate_with_structuralsearch, crossinterpolate_with_3site_swapping
using NamedGraphs: NamedGraph, add_edge!, edges, src, dst
using QuanticsGrids
const QG = QuanticsGrids
using SparseIR
using ITensorNetworks
using TreeTCI: ttnopt
const ITN = ITensorNetworks
using ITensors
using NPZ
include("utils.jl")
include("graphs.jl")


ε(k) = 2*cos(k) + cos(5*k) + 2*cos(20*k)

gk(m::Int, kx::Float64; β::Float64=10.0) = begin
    m = 2 * m + 1
    ν = FermionicFreq(m)
    iν = SparseIR.valueim(ν, β)
    return 1 / (iν - ε(kx))
end

function gkb(b::Vector{Int}, n_m::Int, n_kx::Int; mu::Float64=0.0, β::Float64=10.0, layout::Symbol=:block)
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
    return gk(m, kx; mu=mu, β=β)
end

function main()
    nkx_bit = 10
    nm_bit = 10
    localdims = fill(2, nkx_bit + nm_bit)
    f(v) = gkb(v, nm_bit, nkx_bit; layout=:interleave)
    g = graph_TT(nkx_bit + nm_bit)
    maxbonddim = 200
    kwargs = (maxbonddim = maxbonddim, maxiter = 100, tolerance = 1e-13)
    center_vertex = (nkx_bit + nm_bit) ÷ 2
    # center_vertex = 1

    ttn, ranks, errors = crossinterpolate(ComplexF64, f, localdims, g; center_vertex = center_vertex, kwargs...)
    @show last(ranks)


    g_tmp, original_entanglements, entanglements = ttnopt(ttn; ortho_vertex = center_vertex, max_degree = 2)
    ttn_, ranks, errors = crossinterpolate(ComplexF64, f, localdims, g_tmp; center_vertex = center_vertex, kwargs...)
    @show last(ranks)

    return 0
end

main()