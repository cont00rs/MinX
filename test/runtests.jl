using Test
using SafeTestsets

@safetestset "Minx.jl" begin
    include("test_assemble.jl")
    include("test_convergence_rate.jl")
    include("test_main.jl")
end
