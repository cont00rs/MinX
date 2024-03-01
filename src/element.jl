using LinearAlgebra
using StaticArrays

export Element, element_matrix

struct Element
    # Shape function
    N::AbstractMatrix
    # Shape function derivatives
    B::Any
    # Jacobian
    J::Any
    # Constitutive matrix
    D::Any
    # Element matrix
    K::Any
end

shape_fn(element) = element.N
shape_dfn(element) = element.B
measure(element::Element) = det(element.J)
stencil(element) = element.K

shape_function(x::Real) = [1.0 - x, x]
shape_dfunction(x::Real) = [-1.0, +1.0]

# Combines the product of all entries for a given entry of the iterator yielded
# from the cartesian product of arrays.
collapser = x -> Base.product(x...) .|> prod |> SMatrix{1,2^length(x)}

# XXX: The shape functions definition should be aligned with the integration
# domains [-1, +1]. These are now still defined as [[0, 1], which is OK, until
# the quadrature is introduced too.
function shape_function(dim::Integer, x = 0.5, y = 0.5, z = 0.5)
    funs = map(shape_function, [x, y, z][1:dim])
    return collapser(funs)
end

function shape_dfunction(dim::Integer, x = 0.5, y = 0.5, z = 0.5)
    xyz = [x, y, z][1:dim]
    # A helper that selects the derivative of the shape function for the
    # "diagonal" entries. These entries correspond to the dimension for which
    # the derivative is evaluated.
    f_or_df = (diag, x) -> diag ? shape_dfunction(x) : shape_function(x)
    dfuns = [[f_or_df(d == i, co) for (i, co) in enumerate(xyz)] for d = 1:dim]
    return vcat(map(x -> collapser(x), dfuns)...)
end

function element_matrix(mesh::Mesh{Dim}) where {Dim}
    @assert 1 <= Dim <= 3 "Invalid dimension."
    # Shape fun
    N = shape_function(Dim)
    # Shape fun derivative
    b = shape_dfunction(Dim)
    J = b * measure(mesh)
    # TODO: Expand B to "BB" for non-scalar problems.
    B = inv(J) * b
    # Constitutive
    # TODO: Accept material properties in dispatched constructors?
    D = Matrix(I, Dim, Dim)
    # Element matrix
    K = det(J) * B' * D * B
    return Element(N, B, J, D, K)
end
