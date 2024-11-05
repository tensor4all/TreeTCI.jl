@testitem "Code quality (Aqua.jl)" begin
    using Test
    using Aqua

    import T4ATemplate

    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(T4ATemplate; unbound_args = false, deps_compat = false)
    end

end

@testitem "Code linting (JET.jl)" begin
    using Test
    using JET

    import T4ATemplate

    if VERSION >= v"1.9"
        @testset "Code linting (JET.jl)" begin
            JET.test_package(T4ATemplate; target_defined_modules = true)
        end
    end
end
