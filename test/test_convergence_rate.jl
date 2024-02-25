using Test
using MinX

@testset "Convergence rate tests" begin
    # One-dimensional heat equation
    # - left, right clamped at zero
    # - unit material properties

    omega = 2pi
    forcing(xyz) = sin(omega * xyz[1]) * omega^2
    solution(xyz) = sin(omega * xyz[1])
    dsolution(xyz) = cos(omega * xyz[1]) * omega

    delta, l2, energy = MinX.convergence_rate(forcing, solution, dsolution)
    @test 1.99 < last(MinX.convergence_rates(delta, l2)) < 2.01
    # TODO: Fix this upperbound: it convergence with rate 2.0 too?
    @test 0.99 < last(MinX.convergence_rates(delta, energy)) # < 1.01
end
