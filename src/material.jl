export Material, MaterialType, Heat, Elastic, type
export heat, elastic

# XXX: This enum is only used to determine if B should remain B,
# or become expanded towards BB. It seems there would be better ways
# to encode that that are not directly based on the material type?
@enum MaterialType begin
    Heat
    Elastic
end

struct Material
    type::MaterialType
    constitutive::Function
end

function heat(kappa::Number)
    constitutive(dim) = kappa * Matrix(I, dim, dim)
    return Material(Heat, constitutive)
end

function elastic(E::Number, nu::Number, plane_strain = true)
    # XXX: If the situation appears where constitutive needs to be
    # reevaluated while assembling, the if/else should be rearranged.
    function constitutive(dim)
        if dim == 1
            return E * Matrix(I, dim, dim)
        elseif dim == 2 && plane_strain
            return E / ((1 + nu) * (1 - 2 * nu)) *
                   [[1 - nu nu 0]; [nu 1 - nu 0]; [0 0 (1 - 2 * nu) / 2]]
        elseif dim == 2
            return E / (1 - nu^2) * [[1 nu 0]; [nu 1 0]; [0 0 (1 - nu) / 2]]
        end
    end
    return Material(Elastic, constitutive)
end

type(m::Material) = m.type
constitutive(m::Material, dim) = m.constitutive(dim)
