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
function assemble(mesh::Mesh{Dim}, element::Element) where {Dim}
    # XXX: Deal with thermo-elastic?
    npe = nodes_per_element(mesh)
    dpn = dofs_per_node(element)
    nnz_per_element = (dpn * npe)^2

    nnz = nnz_per_element * length(elements(mesh))
    rows = zeros(Int, nnz)
    cols = zeros(Int, nnz)
    vals = zeros(nnz)

    nodes = nodes_buffer(mesh)
    dofs = dofs_buffer(mesh, dpn)

    for (quad_w, quad_xyz) in quadrature(element)
        Ke = quad_w * element_matrix(element, mesh, quad_xyz)

        for (i, el) in enumerate(elements(mesh))
            nodes!(nodes, mesh, el)
            dofs!(dofs, mesh, nodes)

            slice = (1+(i-1)*length(Ke)):i*length(Ke)
            @views repeat!(rows[slice], vec(dofs))
            @views tile!(cols[slice], vec(dofs))
            vals[slice] += reshape(Ke, :)
        end
    end

    return sparse(rows, cols, vals)
end

# Assemble some function onto mesh nodes
# XXX: Only OK when element_dimension == mesh_dimension
function integrate(mesh::Mesh{Dim}, fn::Forcing) where {Dim}
    npe = nodes_per_element(mesh)

    nodes = nodes_buffer(mesh)
    dofs = dofs_buffer(mesh, dofs_per_node(fn))
    xyz = zeros(Float64, npe, Dim)

    nnz = length(dofs) * length(elements(mesh))
    cols = zeros(Int, nnz)
    vals = zeros(nnz)

    for (quad_w, quad_xyz) in quadrature(fn)
        N = shape_function(quad_xyz)
        J = jacobian(mesh, quad_xyz)
        weight = det(J) * quad_w

        for (i, el) in enumerate(elements(mesh))
            nodes!(nodes, mesh, el)
            dofs!(dofs, mesh, nodes)

            slice = (1+(i-1)*length(dofs)):i*length(dofs)
            cols[slice] = dofs

            xyz[:, :] = measure(mesh, el)
            vals[slice] += reshape(fn(N * xyz) * N * weight, :)
        end
    end

    return Vector(sparsevec(cols, vals))
end

# Interpolate some function onto quadrature points
function interpolate(mesh::Mesh{Dim}, fn::Forcing) where {Dim}
    npe = nodes_per_element(mesh)
    qpe = length(quadrature(fn))
    interp = zeros(qpe, length(elements(mesh)))
    xyz = zeros(Float64, npe, Dim)

    for (qi, (quad_w, quad_xyz)) in enumerate(quadrature(fn))
        N = shape_function(quad_xyz)

        for (i, el) in enumerate(elements(mesh))
            xyz[:, :] = measure(mesh, el)
            interp[qi, i] += fn(N * xyz) * quad_w
        end
    end

    return interp
end

# Interpolate a state vector onto quadrature points
function interpolate(mesh::Mesh{Dim}, state::AbstractVector) where {Dim}
    # XXX: Quadrature order still hardcoded.
    quadrature = Quadrature(element_dimension(mesh), 1)

    dpn = div(length(state), length(MinX.nodes(mesh)))
    qpe = length(locations(quadrature))
    interp = zeros(dpn, qpe, length(elements(mesh)))

    nodes = nodes_buffer(mesh)
    dofs = dofs_buffer(mesh, dpn)

    for (qi, (quad_w, quad_xyz)) in enumerate(quadrature())
        N = shape_function(quad_xyz)

        for (i, el) in enumerate(elements(mesh))
            nodes!(nodes, mesh, el)
            dofs!(dofs, mesh, nodes)
            interp[:, qi, i] .+= state[dofs] * N' * quad_w
        end
    end

    return interp
end

# Generate derivative of state at quadrature points
function derivative(
    mesh::Mesh{Dim},
    basis::AbstractBasis{Dpn},
    state::AbstractVector,
) where {Dim,Dpn}
    # XXX: Quadrature order still hardcoded.
    quadrature = Quadrature(element_dimension(mesh), 1)

    qpe = length(locations(quadrature))
    du = zeros(num_derivatives(basis, Dim), qpe, length(elements(mesh)))
    nodes = nodes_buffer(mesh)
    dofs = dofs_buffer(mesh, Dpn)

    for (qi, (quad_w, quad_xyz)) in enumerate(quadrature())
        B = shape_dfunction(mesh, basis, quad_xyz)

        for (i, el) in enumerate(elements(mesh))
            nodes!(nodes, mesh, el)
            dofs!(dofs, mesh, nodes)
            du[:, qi, i] .+= B * reshape(state[dofs], :) * quad_w
        end
    end

    return du
end
