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
    for (i, entry) in enumerate(entries)
        array[i:length(entries):end] .= entry
    end
end

# Assemble the global 'stiffness' matrix.
# TODO Figure out memory reuse for sparse matrix.
function assemble(mesh::Mesh{Dim}, element) where {Dim}
    Ke = stencil(element)
    nnz = length(Ke) * length(elements(mesh))

    rows = zeros(Int, nnz)
    cols = zeros(Int, nnz)
    vals = zeros(nnz)

    nodes = zeros(CartesianIndex{Dim}, length(shape_fn(element)))

    dpn = dofs_per_node(element)
    dofs = zeros(Int, dpn, length(shape_fn(element)))

    # XXX: Hardcoded, there are no quadrature loops yet.
    quadrature_weight = 2^Dim

    for (i, el) in enumerate(elements(mesh))
        nodes!(nodes, mesh, el)
        dofs!(dofs, mesh, nodes, dpn)

        slice = (1+(i-1)*length(Ke)):i*length(Ke)
        @views repeat!(rows[slice], vec(dofs))
        @views tile!(cols[slice], vec(dofs))
        vals[slice] = Ke * quadrature_weight
    end

    return sparse(rows, cols, vals)
end

# Assemble some function onto mesh nodes
function integrate(mesh::Mesh{Dim}, element, fun) where {Dim}
    N = shape_fn(element)
    nodes = zeros(CartesianIndex{Dim}, length(N))
    dpn = dofs_per_node(element)
    dofs = zeros(Int, dpn, length(N))
    xyz = zeros(Float64, length(N), Dim)
    nnz = length(dofs) * length(elements(mesh))

    cols = zeros(Int, nnz)
    vals = zeros(nnz)

    # XXX: Hardcoded, there are no quadrature loops yet.
    quadrature_weight = 2^Dim

    for (i, el) in enumerate(elements(mesh))
        nodes!(nodes, mesh, el)
        dofs!(dofs, mesh, nodes, dpn)

        slice = (1+(i-1)*length(dofs)):i*length(dofs)
        cols[slice] = dofs

        xyz[:, :] = measure(mesh, el)
        vals[slice] = fun(N * xyz) * N * det(element.J) * quadrature_weight
    end

    return Vector(sparsevec(cols, vals))
end

# Interpolate some function onto quadrature points
function interpolate(mesh::Mesh{Dim}, element, fun) where {Dim}
    interp = zeros(length(elements(mesh)))
    N = shape_fn(element)
    xyz = zeros(Float64, length(N), Dim)
    for (i, el) in enumerate(elements(mesh))
        xyz[:, :] = measure(mesh, el)
        interp[i] = fun(N * xyz)
    end
    return interp
end

# Interpolate a state vector onto quadrature points
function interpolate(mesh::Mesh{Dim}, element, state::AbstractVector) where {Dim}
    dpn = dofs_per_node(element)
    interp = zeros(dpn, length(elements(mesh)))
    N = shape_fn(element)
    nodes = zeros(CartesianIndex{Dim}, length(N))
    dofs = zeros(Int, dpn, length(N))
    for (i, el) in enumerate(elements(mesh))
        nodes!(nodes, mesh, el)
        dofs!(dofs, mesh, nodes, dpn)
        interp[:, i] .= state[dofs] * N'
    end
    return interp
end

# Generate derivative of state at quadrature points
function derivative(mesh::Mesh{Dim}, element, state) where {Dim}
    dpn = dofs_per_node(element)
    B = shape_dfn(element)
    nodes = zeros(CartesianIndex{Dim}, length(shape_fn(element)))
    dofs = zeros(Int, dpn, length(shape_fn(element)))

    kludge = element.eltype == Elastic ? Dim == 2 ? 3 : 1 : Dim
    du = zeros(kludge, length(elements(mesh)))
    for (i, el) in enumerate(elements(mesh))
        nodes!(nodes, mesh, el)
        dofs!(dofs, mesh, nodes, dpn)
        du[:, i] .= B * reshape(state[dofs], :)
    end
    return du
end
