using JET
import T4ATemplate

@testset "JET" begin
    if VERSION ≥ v"1.10"
        JET.test_package(T4ATemplate; target_defined_modules=true)
    end
end
