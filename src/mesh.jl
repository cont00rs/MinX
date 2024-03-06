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
# TODO: Add dispatch on basis type, e.g. P1, P2.
function nodes(mesh::Mesh)
    Base.product(map(x -> UnitRange(1, x + 1), mesh.nelems)...)
end

# TODO: Could `node` and `nodes!` be made more compact?
node(mesh::Mesh{1}, i) = i
node(mesh::Mesh{2}, i, j) = i + (j - 1) * (mesh.nelems[1] + 1)

function nodes!(array::AbstractVector{T}, mesh, ijk::NTuple{1,T}) where {T}
    i, = ijk
    array[1] = node(mesh, i)
    array[2] = node(mesh, i + 1)
end

function nodes!(array::AbstractVector{T}, mesh, ijk::NTuple{2,T}) where {T}
    i, j = ijk
    array[1] = node(mesh, i, j)
    array[2] = node(mesh, i + 1, j)
    array[3] = node(mesh, i, j + 1)
    array[4] = node(mesh, i + 1, j + 1)
end

# Returns the coordinates of the node.
coords(mesh, node) = (node .- 1) .* mesh.dx

# Returns the coordinates of all nodes of the element.
function measure(mesh::Mesh{Dim}, ijk::NTuple{Dim,T}) where {Dim,T}
    # XXX: Assumes linear elements
    dxs = [[(i - 1) * dx i * dx] for (i, dx) in zip(ijk, mesh.dx)]
    measure = reshape(Base.product(dxs...) .|> collect, (2^Dim))
    # Convert from Vector{Vector{...}} to Matrix{...}.
    return permutedims(reduce(hcat, measure))
end
