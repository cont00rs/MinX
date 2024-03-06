using LinearAlgebra

export fix!, solve, calculate_error, prescribe

# XXX: The `dof`, `dofs!` routines only consider scalar problems yet.
dof(mesh, node) = node
function dofs!(array::AbstractMatrix{T}, mesh, nodes::AbstractVector{T}) where {T}
    for i = 1:length(nodes)
        array[:, i] .= nodes[i]
    end
end

# TODO: This needs more thought for non-scalar problems.
function prescribe(mesh::Mesh{Dim}, element, predicate, fn) where {Dim}
    fixed = Tuple{Int,Float64}[]
    for n in nodes(mesh)
        xyz = coords(mesh, n)
        if predicate(xyz...)
            push!(fixed, (dof(mesh, node(mesh, n...)), fn(xyz...)))
        end
    end
    fixed
end

prescribe(mesh, element, predicate) = prescribe(mesh, element, predicate, (x...) -> 0.0)

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
