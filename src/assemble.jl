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
# TODO Figure out memory reuse for sparse matrix.
function assemble(mesh, element)
    Ke = stencil(element)
    nnz = length(Ke) * length(elements(mesh))

    rows = zeros(Int, nnz)
    cols = zeros(Int, nnz)
    vals = zeros(nnz)

    dofs = zeros(Int, length(shape_fn(element)))

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
function integrate(mesh, element, fun)
    N = shape_fn(element)
    dofs = zeros(Int, length(N))
    nnz = length(dofs) * length(elements(mesh))
    dx = measure(element)

    cols = zeros(Int, nnz)
    vals = zeros(nnz)

    for (i, el) in enumerate(elements(mesh))
        @views dofs!(dofs, el)

        slice = (1+(i-1)*length(dofs)):i*length(dofs)
        cols[slice] = dofs

        # TODO This should accept some real coordinates.
        xco = mean(dx * (dofs .- 1))
        vals[slice] = N * fun(xco) * dx
    end

    F = Vector(sparsevec(cols, vals))
end

# Interpolate some function onto quadrature points
function interpolate(mesh, element, fun)
    interp = zeros(length(elements(mesh)))
    dofs = zeros(Int, length(shape_fn(element)))
    dx = measure(element)
    for (i, el) in enumerate(elements(mesh))
        @views dofs!(dofs, el)
        xco = mean(dx * (dofs .- 1))
        interp[i] = fun(xco)
    end
    return interp
end

# Interpolate a state vector onto quadrature points
function interpolate(mesh, element, state::AbstractVector)
    interp = zeros(length(elements(mesh)))
    N = shape_fn(element)
    dofs = zeros(Int, length(N))
    for (i, el) in enumerate(elements(mesh))
        @views dofs!(dofs, el)
        interp[i] = dot(N, state[dofs])
    end
    return interp
end

# Generate derivative of state ate quadrature points
function derivative(mesh, element, state)
    B = shape_dfn(element)
    dofs = zeros(Int, length(shape_fn(element)))
    dx = measure(element)

    du = zeros(length(elements(mesh)))
    for (i, el) in enumerate(elements(mesh))
        @views dofs!(dofs, el)
        xco = mean(dx * (dofs .- 1))
        du[i] = dot(B, state[dofs])
    end

    return du
end
