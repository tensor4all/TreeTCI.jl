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


ε(kx, ky) = -2*cos(kx) - 2*cos(ky)

gk(kx::Float64, ky::Float64; mu::Float64=0.0, β::Float64=10.0) = begin
    m = 1
    ν = FermionicFreq(m)
    iν = SparseIR.valueim(ν, β)
    return 1 / (iν - ε(kx, ky) + mu)
end

function gkb(b::Vector{Int}, n_kx::Int, n_ky; mu::Float64=0.0, β::Float64=10.0, layout::Symbol=:block)
    @assert length(b) == n_kx + n_ky
    parts = split_bits(b; group_bits=[n_kx, n_ky], layout=layout)
    kxb, kyb = parts
    kxbit = length(kxb)
    kybit = length(kyb)
    Nkx = 2^kxbit
    Nky = 2^kybit
    iky = frombins(kyb)
    ikx = frombins(kxb)
    @assert ikx ≤ Nkx
    @assert iky ≤ Nky
    kx = 2π * (ikx - 1)/Nkx
    ky = 2π * (iky - 1)/Nky
    return gk(kx, ky; mu=mu, β=β)
end


function calculate_entanglements(entanglements)
    ee = 0.0
    edges = []
    edge_vals = []
    for (key, value) in entanglements
        ee += value
        push!(edges, key)
        push!(edge_vals, value)
    end
    @show ee / length(edges)
    @show edges
    @show edge_vals
end

function main()
    nkx_bit = 10
    nky_bit = 10
    localdims = fill(2, nkx_bit + nky_bit)
    f(v) = gkb(v, nky_bit, nkx_bit; layout=:interleave, mu=-1.0, β=10.0)
    g = graph_TT(nkx_bit + nky_bit)
    maxbonddim = 200
    kwargs = (maxbonddim = maxbonddim, maxiter = 100, tolerance = 1e-13)
    center_vertex = (nkx_bit + nky_bit) ÷ 2

    ttn, ranks, errors = crossinterpolate(ComplexF64, f, localdims, g; center_vertex = center_vertex, kwargs...)
    
    g_tmp, original_entanglements, entanglements = ttnopt(ttn; ortho_vertex = center_vertex, max_degree = 1)
    ttn_, ranks, errors = crossinterpolate(ComplexF64, f, localdims, g_tmp; center_vertex = center_vertex, kwargs...)
    println("original_entanglements")
    calculate_entanglements(original_entanglements)
    @show last(ranks)

    println("--------------------------------")
    println("ΔG = 2")
    calculate_entanglements(entanglements)
    @show last(ranks)

    g_tmp, original_entanglements, entanglements = ttnopt(ttn; ortho_vertex = center_vertex, max_degree = 2)
    ttn_, ranks, errors = crossinterpolate(ComplexF64, f, localdims, g_tmp; center_vertex = center_vertex, kwargs...)

    println("--------------------------------")
    println("ΔG = 3")
    calculate_entanglements(entanglements)
    @show last(ranks)

    return 0
end

main()