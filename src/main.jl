using LinearAlgebra

export fix!, solve, prescribe

# TODO: This needs more thought for non-scalar problems.
function prescribe(mesh::Mesh{Dim}, ::AbstractBasis{Dpn}, predicate, fn) where {Dim,Dpn}
    fixed = Tuple{Int,Float64}[]
    for node in nodes(mesh)
        xyz = coords(mesh, node)
        if predicate(xyz...)
            for d = 1:Dpn
                push!(fixed, (dof(mesh, CartesianIndex(d, node), Dpn), fn(xyz...)))
            end
        end
    end
    fixed
end

prescribe(mesh, basis, predicate) = prescribe(mesh, basis, predicate, (x...) -> 0.0)

function solve(mesh::Mesh{Dim}, Ke, forcing, fixed) where {Dim}
    K = assemble(mesh, Ke)
    # TODO: Should the forcing become a struct with force function + quadrature?
    forcing = Forcing(forcing, basis(Ke), Dim)
    f = integrate(mesh, forcing)
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
