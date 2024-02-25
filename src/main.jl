using Printf
using LinearAlgebra

export Mesh
export fix!, solve, measure, calculate_error

struct Mesh
    nelx::Int
end

elements(mesh) = 1:mesh.nelx
measure(mesh) = [0; 1 / mesh.nelx]

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
