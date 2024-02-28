using Test
using MinX

@testset "Mesh construction" begin
    for dim = 1:1
        nel = 10
        mesh = Mesh(Tuple(ones(dim)), Tuple(nel * ones(dim)))
        @test length(mesh.length) == length(mesh.nelems) == length(mesh.dx) == dim
        @test length(elements(mesh)) == nel^dim
        @test length(nodes(mesh)) == (nel + 1)^dim
    end
end

@testset "Mesh utilities" begin
    m1 = Mesh((1,), (10,))
    xyz = zeros(2, 1)
    MinX.coords!(xyz, m1, (1,))
    @test xyz == [0; 1/10;;]
end
