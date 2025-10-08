"""
    split_bits(b; group_bits, layout=:block)

Split a bit sequence b of length `length(b)` (where each element is 1/2) into groups
with bit counts specified by `group_bits`.

Arguments
- b::AbstractVector{<:Integer} : Bit sequence of 1/2 (MSB-first)
- group_bits::AbstractVector{<:Integer} : Number of bits in each group (e.g., [mbit, kxbit, …])
- layout::Symbol = :block | :interleave
    - :block      → Assumes b = [ grp1..., grp2..., … ] concatenation
    - :interleave → Assumes b = [ g1_1, g2_1, …, gG_1, g1_2, g2_2, … ] order
                    Assumes all groups have equal bit counts

Returns
- Vector{Vector{Int}} : Bit sequences for each group (each element is 1/2)
"""
function split_bits(b::AbstractVector{<:Integer}; group_bits::AbstractVector{<:Integer}, layout::Symbol=:block)
    @assert all(x -> x ≥ 0, group_bits) "group_bits must be nonnegative"
    G = length(group_bits)
    total = sum(group_bits)
    @assert length(b) == total "length(b) must equal sum(group_bits)"

    if layout === :block
        parts = Vector{Vector{Int}}(undef, G)
        start = 1
        @inbounds for g in 1:G
            len = group_bits[g]
            parts[g] = collect(b[start:start+len-1])
            start += len
        end
        return parts

    elseif layout === :interleave
        @assert length(unique(group_bits)) == 1 "interleave layout requires all group lengths equal"
        L = group_bits[1]              # 各グループの長さ
        @assert L * G == length(b)
        parts = [Vector{Int}(undef, L) for _ in 1:G]
        # b = [ g1_1, g2_1, …, gG_1, g1_2, g2_2, …, gG_2, … ]
        @inbounds for j in 1:L
            base = (j-1)*G
            for g in 1:G
                parts[g][j] = b[base + g]
            end
        end
        return parts

    else
        error("layout must be :block or :interleave")
    end
end


"""
    join_bits(parts; layout=:block)

Inverse of `split_bits`. Combines bit sequences `parts` from each group (1/2 1-based)
to return a single bit sequence b.

Arguments
- parts::Vector{<:AbstractVector{<:Integer}} : Bit sequences for each group
- layout::Symbol = :block | :interleave

Note
- For :interleave, assumes all groups have equal length.
"""
function join_bits(parts::Vector{<:AbstractVector{<:Integer}}; layout::Symbol=:block)
    G = length(parts)
    if layout === :block
        return vcat(parts...)
    elseif layout === :interleave
        Lset = length.(parts)
        @assert length(unique(Lset)) == 1 ":interleave requires equal group lengths"
        L = Lset[1]
        out = Vector{Int}(undef, L*G)
        # out = [ g1_1, g2_1, …, gG_1, g1_2, g2_2, …, gG_2, … ]
        @inbounds for j in 1:L
            base = (j-1)*G
            for g in 1:G
                out[base + g] = parts[g][j]
            end
        end
        return out
    else
        error("layout must be :block or :interleave")
    end
end


function tobins(i, nbit)
    @assert 1 ≤ i ≤ 2^nbit
    mask = 1 << (nbit-1)
    bin = ones(Int, nbit)
    for n in 1:nbit
        bin[n] = (mask & (i-1)) >> (nbit-n) + 1
        mask = mask >> 1
    end
    return bin
end

function frombins(bin)
    @assert all(1 .≤ bin .≤ 2)
    nbit = length(bin)
    i = 1
    tmp = 2^(nbit-1)
    for n in eachindex(bin)
        i += tmp * (bin[n] -1)
        tmp = tmp >> 1
    end
    return i
end
