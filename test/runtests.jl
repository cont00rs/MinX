using Test
using SafeTestsets

@safetestset "Minx.jl" begin
    include("test_main.jl")
end
