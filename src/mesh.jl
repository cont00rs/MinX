using StaticArrays

export Mesh, elements, nodes, coords

struct Mesh{Dim}
    length::SVector{Dim,Float64}
    nelems::SVector{Dim,Int}
    dx::SVector{Dim,Float64}
end

# Constructor passing tuples for nelems and length.
function Mesh(length::NTuple{Dim}, nelems::NTuple{Dim}) where {Dim}
    length = SVector{Dim,Float64}(length)
    nelems = SVector{Dim,Int}(nelems)
    dx = SVector{Dim,Float64}(length ./ nelems)
    Mesh(length, nelems, dx)
end

# Formatting for Mesh struct.
function Base.show(io::IO, m::Mesh{Dim}) where {Dim}
    print(io, "Mesh{$(Dim)}: L: $(m.length), nel: $(m.nelems), dx: $(m.dx)")
end

# Iterator over mesh elements using tuple of (i, j, k) indexing.
function elements(mesh::Mesh)
    return Base.product(map(x -> UnitRange(1, x), mesh.nelems)...)
end

# Iterator over mesh nodes using tuple of (i, j, k) indexing.
# TODO: This requires more information for quadratic elements.
function nodes(mesh::Mesh)
    Base.product(map(x -> UnitRange(1, x + 1), mesh.nelems)...)
end

measure(mesh::Mesh{1}) = [0; mesh.dx[1]]
