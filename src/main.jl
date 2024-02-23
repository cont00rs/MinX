using Printf
using LinearAlgebra

export Mesh, elements, element_matrix
export fix!, solve, measure, calculate_error

struct Mesh
    nelx::Int
end

elements(mesh) = 1:mesh.nelx
measure(mesh) = [0; 1 / mesh.nelx]

function element_matrix(mesh)
    b = [-1, 1]
    J = b' * measure(mesh)
    B = inv(J) * b
    D = Matrix(I, 1, 1)
    return det(J) * B * D * B'
end


function dofs!(array, ix)
    array[1] = ix
    array[2] = ix + 1
end

function solve(mesh, forcing)
    Ke = element_matrix(mesh)
    K = assemble(mesh, Ke)
    f = integrate(mesh, forcing)
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
