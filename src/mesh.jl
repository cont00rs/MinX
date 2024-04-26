using StaticArrays

export Mesh, elements, nodes, coords

struct Mesh{Dim}
    length::SVector{Dim,Float64}
    nelems::SVector{Dim,Int}
    dx::SVector{Dim,Float64}
    elements::CartesianIndices{Dim}
    nodes::CartesianIndices{Dim}
    node_offsets::CartesianIndices{Dim}
end

function Mesh(length::NTuple{Dim}, nelems::NTuple{Dim}) where {Dim}
    return Mesh(length, nelems, ntuple(x -> 1, Dim), ntuple(x -> 1, Dim))
end

ranges_from_zip(x...) = tuple(splat(StepRange).(zip(x...))...)

function Mesh(
    length::NTuple{Dim},
    nelems::NTuple{Dim},
    origin::NTuple{Dim},
    offset::NTuple{Dim},
) where {Dim}
    length = SVector{Dim,Float64}(length)
    nelems = SVector{Dim,Int}(nelems)
    dx = SVector{Dim,Float64}(length ./ nelems)

    elements = CartesianIndices(ranges_from_zip(origin, offset, nelems))
    nodes = CartesianIndices(ranges_from_zip(origin, offset, nelems .+ 1))

    node_offsets = CartesianIndices(ntuple(x -> 0:1, Dim))

    Mesh(length, nelems, dx, elements, nodes, node_offsets)
end

# Formatting for Mesh struct.
function Base.show(io::IO, m::Mesh{Dim}) where {Dim}
    print(io, "Mesh{$(Dim)}: L: $(m.length), nel: $(m.nelems), dx: $(m.dx)")
end

# TODO: Add dispatch on basis type, e.g. P1, P2.
elements(mesh::Mesh) = mesh.elements
nodes(mesh::Mesh) = mesh.nodes
node_offsets(mesh::Mesh) = mesh.node_offsets

# Extract all linear node indices for element ijk.
function nodes!(
    array::AbstractVector{CartesianIndex{Dim}},
    mesh::Mesh{Dim},
    ijk,
) where {Dim}
    for (i, offset) in enumerate(node_offsets(mesh))
        array[i] = ijk + offset
    end
end

# Returns the coordinates of the node.
coords(mesh, node::CartesianIndex) = (Tuple(node) .- 1) .* mesh.dx

# Returns the coordinates of all nodes of the element.
function measure(mesh::Mesh{Dim}, ijk) where {Dim}
    # XXX: Assumes linear elements
    dxs = [[(i - 1) * dx i * dx] for (i, dx) in zip(Tuple(ijk), mesh.dx)]
    measure = reshape(Base.product(dxs...) .|> collect, (2^Dim))
    # Convert from Vector{Vector{...}} to Matrix{...}.
    return permutedims(reduce(hcat, measure))
end
