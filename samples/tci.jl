import TreeTCI

function main()
    f(v) = 1 / (1 + v' * v)
    localdims = fill(10, 5)
    tolerance = 1e-8
    tci, ranks, errors =
        TreeTCI.TCI.crossinterpolate2(Float64, f, localdims; tolerance = tolerance)
    @show tci
end

main()
