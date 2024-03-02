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

# TODO: `measure` and `coords` are rather similar. These routines can be
# combined into a single interface. Possibly introduce subtypes to distinguish
# between nodes and elements which are currently both `NTuple{Dim, T}` types,
# preventing dispatch?
# XXX: This assumes an element being two nodes only!
function measure(mesh::Mesh{Dim}, ijk::NTuple{Dim,T}) where {Dim,T}
    dxs = [[(i - 1) * dx i * dx] for (i, dx) in zip(ijk, mesh.dx)]
    measure = reshape(Base.product(dxs...) .|> collect, (2^Dim))
    # Convert from Vector{Vector{...}} to Matrix{...}.
    return permutedims(reduce(hcat, measure))
end

function coords!(xyz::AbstractMatrix, mesh::Mesh{Dim}, ijk::NTuple{Dim,T}) where {Dim,T}
    xyz[:, :] = measure(mesh, ijk)
end

coords(mesh, node) = (node .- 1) .* mesh.dx
coords(mesh) = map(node -> coords(mesh, node), nodes(mesh))
