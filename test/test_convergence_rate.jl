using Test
using MinX


struct ConvergenceTest
    material::Material
    dimension::Integer
    forcing::Function
    fixed_boundary::Function
    solution::Vector{Function}
    derivatives::Vector{Function}
    l2norm::Tuple{Real,Real}
    energy::Tuple{Real,Real}
end

# This does require Lame parameters lambda = mu = 1 yielding E = 2.5, nu = 0.25.
forcing_elastic_2d(lambda, mu) =
    xyz -> [
        -pi^2 * (
            -(lambda + 3mu) * sin(pi * xyz[1]) * sin(pi * xyz[2]) +
            (lambda + mu) * cos(pi * xyz[1]) * cos(pi * xyz[2])
        ),
        -pi^2 * (
            -(lambda + 3mu) * sin(pi * xyz[1]) * sin(pi * xyz[2]) +
            (lambda + mu) * cos(pi * xyz[1]) * cos(pi * xyz[2])
        ),
    ]


convergence_tests = [
    ConvergenceTest(
        heat(1),
        1,
        xyz -> sin(2 * pi * xyz[1]) * (2 * pi)^2,
        x -> isapprox(x, 0) || isapprox(x, 1),
        [xyz -> sin(2 * pi * xyz[1])],
        [xyz -> cos(2 * pi * xyz[1]) * 2 * pi],
        (1.99, 2.01),
        (0.99, 2.01),
    ),
    ConvergenceTest(
        heat(1),
        2,
        xyz -> sin(pi * xyz[1]) * pi^2 * sin(pi * xyz[2]) * 2,
        (x, y) -> x ≈ 0.0 || x ≈ 1.0 || y ≈ 0.0 || y ≈ 1.0,
        [xyz -> sin(pi * xyz[1]) * sin(pi * xyz[2])],
        [
            xyz -> sin(pi * xyz[2]) * pi * cos(pi * xyz[1])
            xyz -> pi * (sin(pi * xyz[1]) * cos(pi * xyz[2]))
        ],
        (1.99, 2.01),
        (0.99, 2.01),
    ),
    ConvergenceTest(
        heat(1),
        3,
        xyz ->
            sin(pi * xyz[1]) * pi^2 * sin(pi * xyz[2]) * 2.0 * sin(pi * xyz[3]) * 1.5,
        (x, y, z) -> x ≈ 0.0 || x ≈ 1.0 || y ≈ 0.0 || y ≈ 1.0 || z ≈ 0.0 || z ≈ 1.0,
        [xyz -> sin(pi * xyz[1]) * sin(pi * xyz[2]) * sin(pi * xyz[3])],
        [
            xyz -> sin(pi * xyz[3]) * sin(pi * xyz[2]) * pi * cos(pi * xyz[1])
            xyz -> sin(pi * xyz[3]) * pi * (sin(pi * xyz[1]) * cos(pi * xyz[2]))
            xyz -> pi * sin(pi * xyz[1]) * sin(pi * xyz[2]) * cos(pi * xyz[3])
        ],
        (1.99, 2.01),
        (0.99, 2.01),
    ),
    ConvergenceTest(
        elastic(1, 0),
        1,
        xyz -> xyz[1],
        x -> isapprox(x, 0),
        [xyz -> 0.5 * xyz[1] - (1 / 6) * xyz[1]^3],
        [xyz -> 0.5 - 0.5 * xyz[1]^2],
        (1.99, 2.01),
        (0.99, 2.01),
    ),
    ConvergenceTest(
        # https://doi.org/10.1007/s00466-023-02282-2
        # TODO: Also implement 'divergent free' example from this paper.
        elastic(2.5, 0.25, true),
        2,
        forcing_elastic_2d(1, 1),
        (x, y) -> x ≈ 0.0 || x ≈ 1.0 || y ≈ 0.0 || y ≈ 1.0,
        [
            xyz -> sin(pi * xyz[1]) * sin(pi * xyz[2]),
            xyz -> sin(pi * xyz[1]) * sin(pi * xyz[2]),
        ],
        [
            xyz -> pi * cos(pi * xyz[1]) * sin(pi * xyz[2]),
            xyz -> pi * sin(pi * xyz[1]) * cos(pi * xyz[2]),
            xyz ->
                pi * cos(pi * xyz[1]) * sin(pi * xyz[2]) +
                pi * sin(pi * xyz[1]) * cos(pi * xyz[2]),
        ],
        (1.99, 2.01),
        (0.99, 2.01),
    ),
]

@testset "Convergence rate tests" begin
    function convergence_test(test::ConvergenceTest)
        delta, l2norm, energy = MinX.convergence_rate(
            test.dimension,
            test.material,
            test.forcing,
            test.fixed_boundary,
            test.solution,
            test.derivatives,
        )
        @test first(test.l2norm) <
              last(MinX.convergence_rates(delta, l2norm)) <
              last(test.l2norm)
        @test first(test.energy) <
              last(MinX.convergence_rates(delta, energy)) <
              last(test.energy)
    end

    map(convergence_test, convergence_tests)
end
