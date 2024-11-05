@testitem begin
    using Aqua
    import T4ATemplate

    @testset "Aqua" begin
        Aqua.test_stale_deps(T4ATemplate)
    end
end
