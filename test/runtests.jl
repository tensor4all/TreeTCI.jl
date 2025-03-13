# using ReTestItems: runtests, @testitem
# using TreeTCI: TreeTCI

# runtests(TreeTCI)

using TreeTCI: TreeTCI
using Test

@testset verbose = true "TreeTCI tests" begin
    # @testset "Code quality (Aqua.jl)" begin
    #     Aqua.test_all(TreeTCI; unbound_args = false, deps_compat = false)
    # end

    @testset verbose = true "Actual tests" begin
        include("treegraph_utils_test.jl")
    end
end
