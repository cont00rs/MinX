export AbstractMaterial
export Heat, Elastic

abstract type AbstractMaterial end

struct Heat{T<:Number} <: AbstractMaterial
    kappa::T
end

struct Elastic{T<:Number} <: AbstractMaterial
    E::T
    nu::T
    plane_strain::Bool
end

function constitutive(material::Heat, ::Mesh{Dim}) where {Dim}
    return material.kappa * Matrix(I, Dim, Dim)
end

function constitutive(material::Elastic, ::Mesh{Dim}) where {Dim}
    E = material.E
    nu = material.nu
    plane_strain = material.plane_strain

    if Dim == 1
        return E * Matrix(I, Dim, Dim)
    elseif Dim == 2 && plane_strain
        return E / ((1 + nu) * (1 - 2 * nu)) *
               [[1 - nu nu 0]; [nu 1 - nu 0]; [0 0 (1 - 2 * nu) / 2]]
    elseif Dim == 2
        return E / (1 - nu^2) * [[1 nu 0]; [nu 1 0]; [0 0 (1 - nu) / 2]]
    end
end
