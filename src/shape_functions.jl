using StaticArrays: SMatrix

shape_function(x::Real) = [(1.0 - x) / 2, (1 + x) / 2]
shape_dfunction(_::Real) = [-0.5, +0.5]

# Combines the product of all entries for a given entry of the iterator yielded
# from the cartesian product of arrays.
collapser = x -> Base.product(x...) .|> prod |> SMatrix{1,2^length(x)}

function shape_function(xyz)
    return collapser(map(shape_function, xyz))
end

# Expand B to "BB" for vector elements, include cross terms.
expand(::Mesh{Dim}, ::ScalarBasis, B) where {Dim} = B

function expand(::Mesh{Dim}, ::VectorBasis, B) where {Dim}
    if Dim == 1
        return B
    elseif Dim == 2
        BB = zeros(3, 8)
        BB[1, 1:2:end] = B[1, :]
        BB[3, 2:2:end] = B[1, :]
        BB[2, 2:2:end] = B[2, :]
        BB[3, 1:2:end] = B[2, :]
        return BB
    end
    @assert false "Unresolved shape dfunction expansion."
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
