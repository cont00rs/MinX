using LinearAlgebra
using StaticArrays

export Element, element_matrix

struct Element
    # Shape function
    N::AbstractMatrix
    # Shape function derivatives
    B::Any
    # Jacobian
    J::Any
    # Constitutive matrix
    D::Any
    # Element matrix
    K::Any
end

shape_fn(element) = element.N
shape_dfn(element) = element.B
measure(element::Element) = det(element.J)
stencil(element) = element.K

shape_function(x::Real) = [1.0 - x, x]

# XXX: The shape functions definition should be aligned with the integration
# domains [-1, +1]. These are now still defined as [[0, 1], which is OK, until
# the quadrature is introduced too.
function shape_function(dim::Integer)
    # XXX: For now hardcoded evaluated at element center (0.5).
    Base.product(repeat([shape_function(0.5)], dim)...) .|> prod |> SMatrix{1,2^dim}
end

# TODO: Convert to proper struct constructor?
function element_matrix(mesh::Mesh{Dim}) where {Dim}
    # Shape fun
    N = shape_function(Dim)
    # Shape fun derivative
    b = [-1 1]
    J = b * measure(mesh)
    B = inv(J) * b
    # Constitutive
    D = Matrix(I, 1, 1)
    # Element matrix
    K = det(J) * B' * D * B
    return Element(N, B, J, D, K)
end
