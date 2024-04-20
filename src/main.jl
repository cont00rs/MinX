using LinearAlgebra

export fix!, solve, prescribe

# XXX: Probably want to store `LinearIndices` instance somewhere to prevent reallocs.
dof(mesh, node, dpn) = LinearIndices((dpn, Tuple(mesh.nelems .+ 1)...))[node]

function dofs!(
    array::MMatrix{Dpn, N, T},
    mesh::Mesh{Dim},
    nodes::AbstractVector{CartesianIndex{Dim}},
) where {Dpn,N,T,Dim,}
    # For scalar field data is aligned as: Matrix(nx, ny)
    # For vector field data is aligned as: Matirx(2, nx, ny)
    # The node remains the same cartesian index, (ni, nj)
    #
    # Which is then transformed to index different components of the vector:
    #   x: CartesianIndex(1, ci) with ci the CartesianIndex of (ni, nj)
    #   y: CartesianIndex(2, ci) with ci the CartesianIndex of (ni, nj)
    for d in axes(array, 1)
        for (i, node) in enumerate(nodes)
            array[d, i] = dof(mesh, CartesianIndex(d, node), Dpn)
        end
    end
end

# TODO: This needs more thought for non-scalar problems.
function prescribe(mesh::Mesh{Dim}, element, dpn, predicate, fn) where {Dim}
    fixed = Tuple{Int,Float64}[]
    for node in nodes(mesh)
        xyz = coords(mesh, node)
        if predicate(xyz...)
            for d = 1:dpn
                push!(fixed, (dof(mesh, CartesianIndex(d, node), dpn), fn(xyz...)))
            end
        end
    end
    fixed
end

prescribe(mesh, element, dpn, predicate) =
    prescribe(mesh, element, dpn, predicate, (x...) -> 0.0)

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
