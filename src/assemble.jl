using LinearAlgebra
using SparseArrays
using Statistics: mean

export assemble, integrate, interpolate, derivative

function repeat!(array, entries)
    n = div(length(array), length(entries))
    for i = 1:n
        array[i:n:end] = entries
    end
end

function tile!(array, entries)
    n = div(length(array), length(entries))
    for (i, entry) in enumerate(entries)
        array[i:length(entries):end] .= entry
    end
end

# Assemble the global 'stiffness' matrix.
function assemble(mesh, Ke)
    nnz = length(Ke) * length(elements(mesh))

    # TODO Figure out memory reuse for sparse matrix.
    rows = zeros(Int, nnz)
    cols = zeros(Int, nnz)
    vals = zeros(nnz)

    # TODO Extract dof vector size from helper routine.
    dofs = zeros(Int, 2)

    for (i, el) in enumerate(elements(mesh))
        # TODO Dispatch on element type.
        @views dofs!(dofs, el)

        slice = (1+(i-1)*length(Ke)):i*length(Ke)
        @views repeat!(rows[slice], dofs)
        @views tile!(cols[slice], dofs)
        vals[slice] = Ke
    end

    K = sparse(rows, cols, vals)
end

# Assemble some function onto mesh nodes
function integrate(mesh, fun)
    dofs = zeros(Int, 2)
    nnz = length(dofs) * length(elements(mesh))

    dx = 1 / mesh.nelx

    cols = zeros(Int, nnz)
    vals = zeros(nnz)

    # TODO Extract shape function from helper.
    N = [0.5, 0.5]

    for (i, el) in enumerate(elements(mesh))
        @views dofs!(dofs, el)

        # TODO This should accept some real coordinates.
        slice = (1+(i-1)*length(dofs)):i*length(dofs)
        cols[slice] = dofs

        xco = mean(dx * (dofs .- 1))
        vals[slice] = N * fun(xco) * dx
    end

    F = Vector(sparsevec(cols, vals))
end

# Interpolate some function onto quadrature points
function interpolate(mesh, fun)
    interp = zeros(length(elements(mesh)))
    dofs = zeros(Int, 2)
    dx = 1 / mesh.nelx
    for (i, el) in enumerate(elements(mesh))
        @views dofs!(dofs, el)
        xco = mean(dx * (dofs .- 1))
        interp[i] = fun(xco)
    end
    return interp
end

# Interpolate a state vector onto quadrature points
function interpolate(mesh, state::AbstractVector)
    interp = zeros(length(elements(mesh)))
    dofs = zeros(Int, 2)
    dx = 1 / mesh.nelx
    N = [0.5, 0.5]
    for (i, el) in enumerate(elements(mesh))
        @views dofs!(dofs, el)
        interp[i] = dot(N, state[dofs])
    end
    return interp
end

# Generate derivative of state ate quadrature points
function derivative(mesh, state)
    dofs = zeros(Int, 2)
    dx = 1 / mesh.nelx
    N = [0.5, 0.5]
    B = [-1, 1]
    J = B' * measure(mesh)
    B = B * inv(J)

    du = zeros(length(elements(mesh)))

    for (i, el) in enumerate(elements(mesh))
        @views dofs!(dofs, el)
        xco = mean(dx * (dofs .- 1))
        du[i] = dot(B, state[dofs])
    end

    return du
end
