export Quadrature

struct Quadrature{N,T<:Real}
    locations::AbstractVector{NTuple{N,T}}
    weights::AbstractVector{T}
end

function Quadrature(locations::NTuple{N,T}, weights::AbstractVector{T}) where {N,T}
    locations = locations
    weights = weights
end

function Quadrature(dim, order)
    if order > 1
        error("Quadrature for order > 1 not implemented.")
    end
    locations = [ntuple(x -> 0.0, dim)]
    weights = [2.0^dim]
    return Quadrature(locations, weights)
end

weights(q::Quadrature) = q.weights
locations(q::Quadrature) = q.locations
