using Test
using MinX

@testset "Convergence rate tests" begin
    # One-dimensional heat equation
    # - left, right clamped at zero
    # - unit material properties

    omega = 2pi
    forcing(xyz) = sin(omega * xyz[1]) * omega^2
    solution(xyz) = sin(omega * xyz[1])
    dsolution = [xyz -> cos(omega * xyz[1]) * omega]
    boundary = x -> isapprox(x, 0) || isapprox(x, 1)

    delta, l2, energy = MinX.convergence_rate(1, forcing, boundary, solution, dsolution)
    @test 1.99 < last(MinX.convergence_rates(delta, l2)) < 2.01
    # TODO: Fix this upperbound: it convergence with rate 2.0 too?
    @test 0.99 < last(MinX.convergence_rates(delta, energy)) # < 1.01

    forcing2(xyz) = sin(pi * xyz[1]) * pi^2 * sin(pi * xyz[2]) * 2
    solution2(xyz) = sin(pi * xyz[1]) * sin(pi * xyz[2])
    dsolution2 = [
        xyz -> sin(pi * xyz[2]) * pi * cos(pi * xyz[1])
        xyz -> pi * (sin(pi * xyz[1]) * cos(pi * xyz[2]))
    ]
    boundary2 = (x, y) -> x ≈ 0.0 || x ≈ 1.0 || y ≈ 0.0 || y ≈ 1.0

    delta, l2, energy = MinX.convergence_rate(2, forcing2, boundary2, solution2, dsolution2)
    @test 1.99 < last(MinX.convergence_rates(delta, l2)) < 2.01
    @test 0.99 < last(MinX.convergence_rates(delta, energy)) # < 1.01
end
