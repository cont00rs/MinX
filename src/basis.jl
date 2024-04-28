export AbstractBasis

abstract type AbstractBasis{Dpn} end

struct ScalarBasis{Dpn} <: AbstractBasis{Dpn} end
struct VectorBasis{Dpn} <: AbstractBasis{Dpn} end

ScalarBasis(dpn) = ScalarBasis{dpn}()
VectorBasis(dpn) = VectorBasis{dpn}()

dofs_per_node(::AbstractBasis{Dpn}) where {Dpn} = Dpn

num_derivatives(::ScalarBasis, dim) = dim
num_derivatives(::VectorBasis, dim) = dim == 2 ? 3 : 1

# XXX: Probably want to store `LinearIndices` instance somewhere to prevent reallocs.
dof(mesh, node, dpn) = LinearIndices((dpn, Tuple(mesh.nelems .+ 1)...))[node]

function dofs!(
    array::MMatrix{Dpn,N,T},
    mesh::Mesh{Dim},
    nodes::AbstractVector{CartesianIndex{Dim}},
) where {Dpn,N,T,Dim}
    # For scalar field data is aligned as: Matrix(nx, ny)
    # For vector field data is aligned as: Matirx(2, nx, ny)
    # The node remains the same cartesian index, (ni, nj)
    #
    # Which is then transformed to index different components of the vector:
    #   x: CartesianIndex(1, ci) with ci the CartesianIndex of (ni, nj)
    #   y: CartesianIndex(2, ci) with ci the CartesianIndex of (ni, nj)
    for d in axes(array, 1)
        for (i, node) in enumerate(nodes)
            array[d, i] = dof(mesh, CartesianIndex(d, node), Dpn)
        end
    end
end
