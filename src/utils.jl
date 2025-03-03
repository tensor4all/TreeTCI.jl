
function padzero(a::AbstractVector{T}) where {T}
    return Iterators.flatten((a, Iterators.repeated(0)))
end