using Printf
using LinearAlgebra

export Mesh, elements, element_matrix
export fix!, solve, measure, calculate_error
export Element

struct Mesh
    nelx::Int
end

elements(mesh) = 1:mesh.nelx
measure(mesh) = [0; 1 / mesh.nelx]

struct Element
    N::Any
    B::Any
    J::Any
    D::Any
    K::Any
end

shape_fn(element) = element.N
shape_dfn(element) = element.B
measure(element::Element) = det(element.J)
stencil(element) = element.K

# TODO: Convert to proper struct constructor?
function element_matrix(mesh)
    # Shape fun
    N = [0.5, 0.5]
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

function dofs!(array, ix)
    array[1] = ix
    array[2] = ix + 1
end

function solve(mesh, Ke, forcing)
    K = assemble(mesh, Ke)
    f = integrate(mesh, Ke, forcing)
    fix!(K, f)
    u = K \ f
    return u
end

function fix!(K, F)
    # TODO Accept points/regions to fix.
    K[1, :] .= 0
    K[:, 1] .= 0
    K[1, 1] = 1
    K[end, :] .= 0
    K[:, end] .= 0
    K[end, end] = 1
    F[1] = 0
    F[end] = 0
end
