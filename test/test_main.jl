using Test
using MinX

@testset "main" begin
    @test last(MinX.convergence_rates(MinX.remainder_test(sin, cos, 1)...)) > 1.99
    @test last(MinX.convergence_rates(MinX.remainder_test(sin, sin, 1)...)) < 1.80
end
