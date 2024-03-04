using LinearAlgebra

export fix!, solve, calculate_error, prescribe

# TODO: Add dispatch on basis type, e.g. P1, P2.
# TODO: Extend dispatch for 2D, i.e. add element::NTuple{2,T}.
function dofs!(array::AbstractVector{T}, mesh, ijk::NTuple{1,T}) where {T}
    i, = ijk
    array[1] = i
    array[2] = i + 1
end
# XXX: This has so many assumptions on node/dof numbering...
#      Requires refactoring to move that out!
function dofs!(array::AbstractVector{T}, mesh, ijk::NTuple{2,T}) where {T}
    i, j = ijk
    array[1] = i + (j - 1) * (mesh.nelems[1] + 1)
    array[2] = i + (j - 1) * (mesh.nelems[1] + 1) + 1
    array[3] = i + j * (mesh.nelems[1] + 1)
    array[4] = i + j * (mesh.nelems[1] + 1) + 1
end

# TODO: This needs more thought for non-scalar problems.
function prescribe(mesh, predicate, fn)
    fixed = Tuple{Int,Float64}[]
    for node in nodes(mesh)
        xyz = coords(mesh, node)
        if predicate(xyz...)
            # XXX: Accessing node[1] implicitly converts from node to dof for
            #      scalar problems. Probably requires an additional argument
            #      that passes through a dof selecting range.
            push!(fixed, (node[1], fn(xyz...)))
        end
    end
    fixed
end

prescribe(mesh, predicate) = prescribe(mesh, predicate, (x...) -> 0.0)

function solve(mesh, Ke, forcing, fixed)
    K = assemble(mesh, Ke)
    f = integrate(mesh, Ke, forcing)
    fix!(K, f, fixed)
    u = K \ f
    return u
end

# XXX: This only supports fixing dofs to zero.
function fix!(K, F, prescribed)
    for (dof, val) in prescribed
        K[dof, :] .= 0
        K[:, dof] .= 0
        K[dof, dof] = 1
        @assert isapprox(val, 0) "Only support for zero dirichlet."
        F[dof] = val
    end
end
