import SimpleTensorNetworks: TensorNetwork, IndexedArray, Index
import TensorCrossInterpolation as TCI
include("../src/simpletci.jl")


function main()
    f(v) = 1 / (1 + v' * v)
    localdims = fill(10, 5)
    tolerance = 1e-8
    tci, ranks, errors = TCI.crossinterpolate2(Float64, f, localdims; tolerance = tolerance)
    tci_, ranks_, errors_ = crossinterpolate(Float64, f, localdims; tolerance = tolerance)
    @show f([1, 2, 3, 4, 5])
    @show tci([1, 2, 3, 4, 5])
    @show tci_([1, 2, 3, 4, 5])

    return 0
end

main()
