using StaticArrays
using Printf

export Mesh, elements, nodes, coords, boundary

struct Mesh{Dim}
    length::SVector{Dim,Float64}
    nelems::SVector{Dim,Int}
    dx::SVector{Dim,Float64}
    elements::CartesianIndices{Dim}
    nodes::CartesianIndices{Dim}
    offsets::SVector{Dim,Int}
    node_offsets::CartesianIndices{Dim}
end

function Mesh(length::NTuple{Dim}, nelems::NTuple{Dim}) where {Dim}
    return Mesh(
        length,
        nelems,
        ntuple(x -> 1, Dim),
        ntuple(x -> 1, Dim),
        ntuple(x -> 1, Dim),
    )
end

function ranges_from_zip(x...)
    range = length(x) == 2 ? UnitRange : StepRange
    tuple(splat(range).(zip(x...))...)
end

function Mesh(
    length::NTuple{Dim},
    nelems::NTuple{Dim},
    origin::NTuple{Dim},
    offset::NTuple{Dim},
    node_offsets::NTuple{Dim},
) where {Dim}
    length = SVector{Dim,Float64}(length)
    nelems = SVector{Dim,Int}(nelems)
    dx = SVector{Dim,Float64}(length ./ nelems)
    offset = SVector{Dim,Int}(offset)

    elements = CartesianIndices(ranges_from_zip(origin, offset, nelems))
    nodes = CartesianIndices(ranges_from_zip(origin, offset, nelems .+ node_offsets))
    node_offsets = CartesianIndices(ranges_from_zip(ntuple(x -> 0, Dim), node_offsets))

    Mesh(length, nelems, dx, elements, nodes, offset, node_offsets)
end

# Formatting for Mesh struct.
function Base.show(io::IO, m::Mesh{Dim}) where {Dim}
    print(io, "Mesh{$(Dim)}: L: $(m.length), nel: $(m.nelems), dx: $(m.dx)")
end

# TODO: Add dispatch on basis type, e.g. P1, P2.
elements(mesh::Mesh) = mesh.elements
nodes(mesh::Mesh) = mesh.nodes
node_offsets(mesh::Mesh) = mesh.node_offsets

# XXX: Only for P1 elements at the moment.
nodes_per_element(::Mesh{Dim}) where {Dim} = 2^Dim

function nodes_buffer(mesh::Mesh{Dim}) where {Dim}
    return zeros(CartesianIndex{Dim}, nodes_per_element(mesh))
end

function dofs_buffer(mesh::Mesh{Dim}, dpn) where {Dim}
    return zeros(MMatrix{dpn,nodes_per_element(mesh),Int})
end

element_dimension(mesh::Mesh) = sum(mesh.offsets .!= typemax(Int))

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
function measure(mesh::Mesh, ijk)
    # XXX: Assumes linear elements
    dxs = [[(i - 1) * dx i * dx] for (i, dx) in zip(Tuple(ijk), mesh.dx)]
    measure = reshape(Base.product(dxs...) .|> collect, (2^element_dimension(mesh)))
    # Convert from Vector{Vector{...}} to Matrix{...}.
    return permutedims(reduce(hcat, measure))
end

# Create an implicit boundary represention of one of the mesh's sides.
function boundary(mesh::Mesh{Dim}, side) where {Dim}
    @assert 0 < Dim < 4 "Unsupported mesh dimension"

    known_sides = (:left, :right, :bottom, :top, :front, :back)
    msg = @sprintf("Unknown boundary: ':%s', choose from: %s", side, known_sides)
    @assert side in known_sides msg

    length = tuple(mesh.length...)

    if Dim == 1
        nelems = side == :left ? (1,) : (mesh.nelems[1] + 1,)
        origin = side == :left ? (1,) : (mesh.nelems[1] + 1,)
        offset = (1,)
        node_offsets = (0,)
    elseif Dim == 2
        # Extend by one point, these indices fall just outside the mesh.
        nx = mesh.nelems[1] + Int(side == :right)
        ny = mesh.nelems[2] + Int(side == :top)
        nelems = (nx, ny)

        # Shift origins for top/right boundaries.
        ox = side == :right ? nx : 1
        oy = side == :top ? ny : 1
        origin = (ox, oy)

        offx = side in (:top, :bottom)
        offy = side in (:left, :right)
        # StepRange step to infinite results in no offsets.
        offset = (offx ? 1 : typemax(Int), offy ? 1 : typemax(Int))
        node_offsets = (offx, offy)
    elseif Dim == 3
        nx = mesh.nelems[1] + Int(side == :right)
        ny = mesh.nelems[2] + Int(side == :back)
        nz = mesh.nelems[3] + Int(side == :top)
        nelems = (nx, ny, nz)

        ox = side == :right ? nx : 1
        oy = side == :back ? ny : 1
        oz = side == :top ? nz : 1
        origin = (ox, oy, oz)

        offx = side in (:top, :bottom, :front, :back)
        offy = side in (:top, :bottom, :left, :right)
        offz = side in (:left, :right, :front, :back)
        # StepRange step to infinite results in no offsets.
        offset = (offx ? 1 : typemax(Int), offy ? 1 : typemax(Int), offz ? 1 : typemax(Int))
        node_offsets = (offx, offy, offz)
    end

    return Mesh(length, nelems, origin, offset, node_offsets)
end
