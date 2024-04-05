using LinearAlgebra
using StaticArrays

export Element, ElementType, element_matrix
export Heat, Elastic

@enum ElementType begin
    Heat
    Elastic
end

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
    # The type of element
    eltype::ElementType
end

shape_fn(element) = element.N
shape_dfn(element) = element.B
measure(element::Element) = det(element.J)
stencil(element) = element.K

shape_function(x::Real) = [(1.0 - x) / 2, (1 + x) / 2]
shape_dfunction(_::Real) = [-0.5, +0.5]

# Combines the product of all entries for a given entry of the iterator yielded
# from the cartesian product of arrays.
collapser = x -> Base.product(x...) .|> prod |> SMatrix{1,2^length(x)}

function shape_function(dim::Integer, x = 0.0, y = 0.0, z = 0.0)
    funs = map(shape_function, [x, y, z][1:dim])
    return collapser(funs)
end

function shape_dfunction(dim::Integer, x = 0.0, y = 0.0, z = 0.0)
    xyz = [x, y, z][1:dim]
    # A helper that selects the derivative of the shape function for the
    # "diagonal" entries. These entries correspond to the dimension for which
    # the derivative is evaluated.
    # XXX: Make this less dense.
    f_or_df = (diag, x) -> diag ? shape_dfunction(x) : shape_function(x)
    dfuns = [[f_or_df(d == i, co) for (i, co) in enumerate(xyz)] for d = 1:dim]
    return vcat(map(x -> collapser(x), dfuns)...)
end

function element_matrix(eltype::ElementType, mesh::Mesh{Dim}) where {Dim}
    @assert 1 <= Dim <= 3 "Invalid dimension."
    # Shape fun
    N = shape_function(Dim)
    # Shape fun derivative
    b = shape_dfunction(Dim)
    # All elements have the same size, just use the first one here.
    J = b * measure(mesh, tuple(ones(Dim)...))
    B = inv(J) * b

    # XXX: The expansion of B should probably not be tied to the element type
    # directly. This should be deferred from another type/struct? Ideally this
    # is handled through some form of dispatch as well.
    if eltype == Elastic && Dim == 2
        BB = zeros(3, 8)
        BB[1, 1:2:end] = B[1, :]
        BB[3, 2:2:end] = B[1, :]
        BB[2, 2:2:end] = B[2, :]
        BB[3, 1:2:end] = B[2, :]
        B = BB
    end

    # TODO: Pass in the constitutive matrix through a helper function that is
    # exported. This could take the form of `constitutive(mesh, Heat, kappa)`
    # or `constitutive(mesh, Elastic, E, nu)`. Then, for element types that
    # need it, `constitutive` (or another name) can be dispatched over
    # different mesh dimensions and/or different sets of supplied arguments.
    # That removes the if-branching here as well.
    #
    # Constitutive
    if eltype == Heat
        D = Matrix(I, Dim, Dim)
    elseif eltype == Elastic && Dim == 1
        D = Matrix(I, Dim, Dim)
    elseif eltype == Elastic && Dim == 2
        E = 2.5
        nu = 0.25
        # XXX: Make this configurable too, see refactoring TODO above.
        # D = (E / (1 - nu^2)) * [[1 nu 0]; [nu 1 0]; [0 0 (1-nu)/2]] # plane stress
        D =
            E / ((1 + nu) * (1 - 2 * nu)) *
            [[1 - nu nu 0]; [nu 1 - nu 0]; [0 0 (1 - 2 * nu) / 2]] # plain strain
    else
        @assert false, "Unreachable"
    end

    # Element matrix
    K = det(J) * B' * D * B
    return Element(N, B, J, D, K, eltype)
end
