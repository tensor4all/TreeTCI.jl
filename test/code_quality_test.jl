@testitem "Code quality (Aqua.jl)" begin
    using Test
    using Aqua

    import TreeTCI

    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(TreeTCI; unbound_args = false, deps_compat = false)
    end

end

@testitem "Code linting (JET.jl)" begin
    using Test
    using JET

    import TreeTCI

    if VERSION >= v"1.9"
        @testset "Code linting (JET.jl)" begin
            JET.test_package(TreeTCI; target_defined_modules = true)
        end
    end
end
