export Basis

struct Basis{T<:Integer}
    dpn::T
end

function Basis(dpn::T) where {T<:Number}
    dpn = Int(dpn)
end

dofs_per_node(basis::Basis) = basis.dpn
