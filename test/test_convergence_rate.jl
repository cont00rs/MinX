using Test
using MinX

@testset "Convergence rate tests" begin
    # One-dimensional heat equation
    # - left, right clamped at zero
    # - unit material properties

    omega = 2pi
    forcing(x) = sin(omega * x) * omega^2
    solution(x) = sin(omega * x)
    dsolution(x) = cos(omega * x) * omega

    delta, l2, energy = MinX.convergence_rate(forcing, solution, dsolution)
    @test 1.99 < last(MinX.convergence_rates(delta, l2)) < 2.01
    # TODO: Fix this upperbound: it convergence with rate 2.0 too?
    @test 0.99 < last(MinX.convergence_rates(delta, energy)) # < 1.01
end
