using LinearAlgebra

export Element, element_matrix

struct Element
    # Shape function
    N::Any
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

# TODO: Convert to proper struct constructor?
function element_matrix(mesh)
    # Shape fun
    N = [0.5  0.5]
    # Shape fun derivative
    b = [-1, 1]
    J = b' * measure(mesh)
    B = inv(J) * b
    # Constitutive
    D = Matrix(I, 1, 1)
    # Element matrix
    K = det(J) * B * D * B'
    return Element(N, B, J, D, K)
end
