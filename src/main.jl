using LinearAlgebra

export fix!, solve, calculate_error

# TODO: Add dispatch on basis type, e.g. P1, P2.
# TODO: Extend dispatch for 2D, i.e. add element::NTuple{2,T}.
function dofs!(array::AbstractVector{T}, ijk::NTuple{1,T}) where {T}
    array[1] = ijk[1]
    array[2] = ijk[1] + 1
end

# TODO: For 2D problems xyz should be a matrix, 4 points, 2 coords.
function coords!(xyz::AbstractVector, mesh, ijk::NTuple{1,T}) where {T}
    xyz[1] = mesh.dx[1] * (ijk[1] - 1)
    xyz[2] = mesh.dx[1] * ijk[1]
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
