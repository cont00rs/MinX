using LinearAlgebra
using StaticArrays

export Element, element_matrix

struct Element
    # Shape function
    N::AbstractMatrix
    # Shape function derivatives
    B::AbstractMatrix
    # Jacobian
    J::AbstractMatrix
    # Constitutive matrix
    D::AbstractMatrix
    # Element matrix
    K::AbstractMatrix
    # The type of element, i.e. material
    material::Material
    # The basis spanned by the element, e.g. dofs per node information
    basis::Basis
    # Quadrature for the integration
    quadrature::Quadrature
end

shape_fn(element) = element.N
shape_dfn(element) = element.B
measure(element::Element) = det(element.J)
stencil(element) = element.K
dofs_per_node(element::Element) = dofs_per_node(element.basis)
quadrature(element::Element) = element.quadrature


shape_function(x::Real) = [(1.0 - x) / 2, (1 + x) / 2]
shape_dfunction(_::Real) = [-0.5, +0.5]

# Combines the product of all entries for a given entry of the iterator yielded
# from the cartesian product of arrays.
collapser = x -> Base.product(x...) .|> prod |> SMatrix{1,2^length(x)}

function shape_function(xyz)
    return collapser(map(shape_function, xyz))
end

function shape_dfunction(xyz)
    # A helper that selects the derivative of the shape function for the
    # "diagonal" entries. These entries correspond to the dimension for which
    # the derivative is evaluated.
    # XXX: Make this less dense.
    f_or_df = (diag, x) -> diag ? shape_dfunction(x) : shape_function(x)
    dfuns = [[f_or_df(d == i, co) for (i, co) in enumerate(xyz)] for d = 1:length(xyz)]
    return vcat(map(x -> collapser(x), dfuns)...)
end

function element_matrix(material::Material, mesh::Mesh{Dim}) where {Dim}
    @assert 1 <= Dim <= 3 "Invalid dimension."

    # The considered quadrature rule.
    quadrature = Quadrature(Dim, 1)

    # Shape fun
    N = shape_function(first(locations(quadrature)))

    # Shape fun derivative
    b = shape_dfunction(first(locations(quadrature)))

    # All elements have the same size, just use the first one here.
    J = b * measure(mesh, tuple(ones(Dim)...))
    B = inv(J) * b

    # XXX: The expansion of B should probably not be tied to the element type
    # directly. This should be deferred from another type/struct? Ideally this
    # is handled through some form of dispatch as well.
    if type(material) == Elastic && Dim == 2
        BB = zeros(3, 8)
        BB[1, 1:2:end] = B[1, :]
        BB[3, 2:2:end] = B[1, :]
        BB[2, 2:2:end] = B[2, :]
        BB[3, 1:2:end] = B[2, :]
        B = BB
    end

    D = constitutive(material, Dim)

    # Element matrix
    K = det(J) * B' * D * B

    # The spanned basis
    # XXX: It feels backward that this extracts information from material?
    #      Should this information not be included in another way?
    basis = Basis(type(material) == Elastic ? Dim : 1)

    # Pack up all information within the element struct.
    return Element(N, B, J, D, K, material, basis, quadrature)
end
