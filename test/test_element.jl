using Test
using MinX
using LinearAlgebra

@testset "Shape func derivative match element size" begin
    for n = 1:4, m = 1:4
        mesh = MinX.Mesh((n, n), (m, m))
        ke = MinX.element_matrix(Elastic, mesh)
        @test isapprox(min(ke.B...), -m / (2 * n))
        @test isapprox(max(ke.B...), +m / (2 * n))
        # Determinant of J(acobian) should be physical area/4
        @test isapprox(det(ke.J), 1 / 4 * (n / m)^2)
    end
end
