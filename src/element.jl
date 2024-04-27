using LinearAlgebra
using StaticArrays

export Element, element_matrix, dofs_per_node

struct Element
    # Shape function
    N::AbstractMatrix
    # Shape function derivatives
    B::AbstractMatrix
    # Jacobian
    J::AbstractMatrix
    # Constitutive matrix
    D::AbstractMatrix
    # Element matrix
    K::AbstractMatrix
    # The type of element, i.e. material
    material::AbstractMaterial
    # Quadrature for the integration
    quadrature::Quadrature
end

shape_fn(element) = element.N
shape_dfn(element) = element.B
measure(element::Element) = det(element.J)
stencil(element) = element.K

dofs_per_node(element::Element, mesh) = dofs_per_node(element.material, mesh)
dofs_per_node(::Heat, ::Mesh{Dim}) where {Dim} = 1
dofs_per_node(::Elastic, ::Mesh{Dim}) where {Dim} = Dim

quadrature(element::Element) = element.quadrature

shape_function(x::Real) = [(1.0 - x) / 2, (1 + x) / 2]
shape_dfunction(_::Real) = [-0.5, +0.5]

# Combines the product of all entries for a given entry of the iterator yielded
# from the cartesian product of arrays.
collapser = x -> Base.product(x...) .|> prod |> SMatrix{1,2^length(x)}

function shape_function(xyz)
    return collapser(map(shape_function, xyz))
end

function shape_dfunction(xyz)
    # A helper that selects the derivative of the shape function for the
    # "diagonal" entries. These entries correspond to the dimension for which
    # the derivative is evaluated.
    # XXX: Make this less dense.
    f_or_df = (diag, x) -> diag ? shape_dfunction(x) : shape_function(x)
    dfuns = [[f_or_df(d == i, co) for (i, co) in enumerate(xyz)] for d = 1:length(xyz)]
    return vcat(map(x -> collapser(x), dfuns)...)
end

# Expand B to "BB" for vector elements, include cross terms.
expand(::Heat, ::Mesh{Dim}, B) where {Dim} = B

function expand(::Elastic, ::Mesh{Dim}, B) where {Dim}
    if Dim == 1
        BB = B
    elseif Dim == 2
        BB = zeros(3, 8)
        BB[1, 1:2:end] = B[1, :]
        BB[3, 2:2:end] = B[1, :]
        BB[2, 2:2:end] = B[2, :]
        BB[3, 1:2:end] = B[2, :]
    end
    return BB
end

function jacobian(mesh, xyz)
    B = shape_dfunction(xyz)
    XYZ = measure(mesh, ntuple(x -> 1, element_dimension(mesh)))
    return B * XYZ
end

function element_matrix(material::AbstractMaterial, mesh::Mesh{Dim}) where {Dim}
    @assert 1 <= Dim <= 3 "Invalid dimension."

    # The considered quadrature rule.
    quadrature = Quadrature(Dim, 1)

    # Shape fun
    N = shape_function(first(locations(quadrature)))

    # Shape fun derivative
    b = shape_dfunction(first(locations(quadrature)))

    # All elements have the same size, just use the first one here.
    J = jacobian(mesh, first(locations(quadrature)))

    B = inv(J) * b
    B = expand(material, mesh, B)

    D = constitutive(material, mesh)

    # Element matrix
    K = B' * D * B

    # Pack up all information within the element struct.
    return Element(N, B, J, D, K, material, quadrature)
end
