using Test
using MinX

# TODO: Generalise tests into a table-based test configuration.

@testset "Convergence rate heat" begin
    # One-dimensional heat equation
    # - left, right clamped at zero
    # - unit material properties

    omega = 2pi
    forcing(xyz) = sin(omega * xyz[1]) * omega^2
    solution(xyz) = sin(omega * xyz[1])
    dsolution = [xyz -> cos(omega * xyz[1]) * omega]
    boundary = x -> isapprox(x, 0) || isapprox(x, 1)

    delta, l2, energy =
        MinX.convergence_rate(1, Heat, forcing, boundary, solution, dsolution)
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

    delta, l2, energy =
        MinX.convergence_rate(2, Heat, forcing2, boundary2, solution2, dsolution2)
    @test 1.99 < last(MinX.convergence_rates(delta, l2)) < 2.01
    @test 0.99 < last(MinX.convergence_rates(delta, energy)) # < 1.01

    forcing3(xyz) =
        sin(pi * xyz[1]) * pi^2 * sin(pi * xyz[2]) * 2.0 * sin(pi * xyz[3]) * 1.5
    solution3(xyz) = sin(pi * xyz[1]) * sin(pi * xyz[2]) * sin(pi * xyz[3])
    dsolution3 = [
        xyz -> sin(pi * xyz[3]) * sin(pi * xyz[2]) * pi * cos(pi * xyz[1])
        xyz -> sin(pi * xyz[3]) * pi * (sin(pi * xyz[1]) * cos(pi * xyz[2]))
        xyz -> pi * sin(pi * xyz[1]) * sin(pi * xyz[2]) * cos(pi * xyz[3])
    ]
    boundary3 = (x, y, z) -> x ≈ 0.0 || x ≈ 1.0 || y ≈ 0.0 || y ≈ 1.0 || z ≈ 0.0 || z ≈ 1.0

    delta, l2, energy =
        MinX.convergence_rate(3, Heat, forcing3, boundary3, solution3, dsolution3)
    @test 1.99 < last(MinX.convergence_rates(delta, l2)) < 2.01
    @test 0.99 < last(MinX.convergence_rates(delta, energy)) < 2.01 # expected 1.01
end

@testset "Convergence rate elastic" begin
    boundary = x -> isapprox(x, 0)
    forcing(xyz) = xyz[1]
    solution(xyz) = 0.5 * xyz[1] - (1 / 6) * xyz[1]^3
    dsolution = [xyz -> 0.5 - 0.5 * xyz[1]^2]

    delta, l2, energy =
        MinX.convergence_rate(1, Elastic, forcing, boundary, solution, dsolution)
    @test 1.99 < last(MinX.convergence_rates(delta, l2)) < 2.01
    @test 0.99 < last(MinX.convergence_rates(delta, energy)) < 2.01 # expected 1.01
end
