using Test
using Printf
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
    xyz[:, :] = MinX.measure(m1, (1,))
    @test xyz == [0; 1/10;;]

    @test MinX.measure(m1, (1,)) == [0; 1/10;;]

    # Assert measure returns the xyz in expected order
    m2 = Mesh((10, 20), (10, 10))
    @test MinX.measure(m2, (1, 1)) == [[0.0 0.0]; [1.0 0.0]; [0.0 2.0]; [1.0 2.0]]
end

@testset "Mesh boundary 1D" begin
    for nel = 1:3
        mesh = Mesh((1,), (nel,))
        for side in (:left, :right)
            boundary = MinX.boundary(mesh, side)
            @test length(elements(boundary)) == 1
            @test length(nodes(boundary)) == 1

            edge = side == :left ? 1 : nel + 1
            @test CartesianIndex(edge) in elements(boundary)
        end
    end
end

@testset "Mesh boundary 2D" begin
    for nx = 1:3, ny = 1:3
        starts = Dict([
            :left => (1, 1),
            :right => (nx + 1, 1),
            :bottom => (1, 1),
            :top => (1, ny + 1),
        ])

        ends = Dict([
            :left => (1, ny),
            :right => (nx + 1, ny),
            :bottom => (nx, 1),
            :top => (nx, ny + 1),
        ])

        mesh = Mesh((2, 1), (nx, ny))
        for side in (:bottom, :top, :left, :right)
            boundary = MinX.boundary(mesh, side)
            expected = side in (:bottom, :top) ? nx : ny
            @test length(elements(boundary)) == expected
            @test length(nodes(boundary)) == expected + 1
            @test CartesianIndex(starts[side]) in elements(boundary)
            @test CartesianIndex(ends[side]) in elements(boundary)
        end
    end
end

@testset "Mesh boundary 3D" begin
    for nx = 1:3, ny = 1:3, nz = 1:3
        elem_count = Dict([
            :bottom => (nx, ny, 1),
            :top => (nx, ny, 1),
            :left => (1, ny, nz),
            :right => (1, ny, nz),
            :front => (nx, 1, nz),
            :back => (nx, 1, nz),
        ])
        node_count = Dict([
            :bottom => (nx + 1, ny + 1, 1),
            :top => (nx + 1, ny + 1, 1),
            :left => (1, ny + 1, nz + 1),
            :right => (1, ny + 1, nz + 1),
            :front => (nx + 1, 1, nz + 1),
            :back => (nx + 1, 1, nz + 1),
        ])
        starts = Dict([
            :bottom => (1, 1, 1),
            :top => (1, 1, nz + 1),
            :left => (1, 1, 1),
            :right => (nx + 1, 1, 1),
            :front => (1, 1, 1),
            :back => (1, ny + 1, 1),
        ])
        ends = Dict([
            :bottom => (nx, ny, 1),
            :top => (nx, ny, nz + 1),
            :left => (1, ny, nz),
            :right => (nx + 1, ny, nz),
            :front => (nx, 1, nz),
            :back => (nx, ny + 1, nz),
        ])

        mesh = Mesh((1, 2, 3), (nx, ny, nz))
        for side in (:bottom, :top, :left, :right, :front, :back)
            boundary = MinX.boundary(mesh, side)
            @test length(elements(boundary)) == prod(elem_count[side])
            @test length(nodes(boundary)) == prod(node_count[side])
            @test CartesianIndex(starts[side]) in elements(boundary)
            @test CartesianIndex(ends[side]) in elements(boundary)
        end
    end
end
