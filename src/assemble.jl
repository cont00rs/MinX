using LinearAlgebra
using SparseArrays
using Statistics: mean
using StaticArrays: MMatrix

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

    # XXX: Hardcoded, there are no quadrature loops yet.
    quadrature = Quadrature(element_dimension(mesh), 1)
    quadrature_weight = first(weights(quadrature))
    quadrature_xyz = first(locations(quadrature))

    # TODO: should we compute Ke here as function of quadrature?
    # So have a function that is like element_matrix(el::Element, quadrature_xyz)
    # And then also implement the quadrature loop here to be more complete?

    J = jacobian(mesh, quadrature_xyz)
    weight = det(J) * quadrature_weight
    Ke = Ke * weight

    nodes = zeros(CartesianIndex{Dim}, length(shape_fn(element)))
    dofs = zeros(MMatrix{dofs_per_node(element, mesh),length(shape_fn(element)),Int})

    for (i, el) in enumerate(elements(mesh))
        nodes!(nodes, mesh, el)
        dofs!(dofs, mesh, nodes)

        slice = (1+(i-1)*length(Ke)):i*length(Ke)
        @views repeat!(rows[slice], vec(dofs))
        @views tile!(cols[slice], vec(dofs))
        vals[slice] = Ke
    end

    return sparse(rows, cols, vals)
end

# Assemble some function onto mesh nodes
function integrate(mesh::Mesh{Dim}, dpn, fun) where {Dim}

    # XXX: Hardcoded, there are no quadrature loops yet.
    quadrature = Quadrature(element_dimension(mesh), 1)
    quadrature_weight = first(weights(quadrature))
    quadrature_xyz = first(locations(quadrature))

    N = shape_function(quadrature_xyz)
    J = jacobian(mesh, quadrature_xyz)
    weight = det(J) * quadrature_weight

    nodes = zeros(CartesianIndex{Dim}, length(N))
    dofs = zeros(MMatrix{dpn,length(N),Int})
    xyz = zeros(Float64, length(N), Dim)
    nnz = length(dofs) * length(elements(mesh))

    cols = zeros(Int, nnz)
    vals = zeros(nnz)

    for (i, el) in enumerate(elements(mesh))
        nodes!(nodes, mesh, el)
        dofs!(dofs, mesh, nodes)

        slice = (1+(i-1)*length(dofs)):i*length(dofs)
        cols[slice] = dofs

        xyz[:, :] = measure(mesh, el)
        vals[slice] = fun(N * xyz) * N * weight
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
    interp = zeros(dofs_per_node(element, mesh), length(elements(mesh)))
    N = shape_fn(element)
    nodes = zeros(CartesianIndex{Dim}, length(N))
    dofs = zeros(MMatrix{dofs_per_node(element, mesh),length(shape_fn(element)),Int})
    for (i, el) in enumerate(elements(mesh))
        nodes!(nodes, mesh, el)
        dofs!(dofs, mesh, nodes)
        interp[:, i] .= state[dofs] * N'
    end
    return interp
end

# Generate derivative of state at quadrature points
function derivative(mesh::Mesh{Dim}, element, state) where {Dim}
    B = shape_dfn(element)
    nodes = zeros(CartesianIndex{Dim}, length(shape_fn(element)))
    dofs = zeros(MMatrix{dofs_per_node(element, mesh),length(shape_fn(element)),Int})

    du = zeros(size(shape_dfn(element), 1), length(elements(mesh)))
    for (i, el) in enumerate(elements(mesh))
        nodes!(nodes, mesh, el)
        dofs!(dofs, mesh, nodes)
        du[:, i] .= B * reshape(state[dofs], :)
    end
    return du
end
