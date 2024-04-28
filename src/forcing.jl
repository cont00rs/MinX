export Forcing

struct Forcing
    fn::Function
    basis::AbstractBasis
    quadrature::Quadrature
end

# XXX: The quadrature should be the 'element dimension'.. not mesh dimension?
function Forcing(fn::Function, basis::AbstractBasis, dim)
    quadrature = Quadrature(dim, 1)
    return Forcing(fn, basis, quadrature)
end

basis(forcing::Forcing) = forcing.basis
dofs_per_node(forcing::Forcing) = dofs_per_node(basis(forcing))

# Invoke underlying function by calling struct
(fn::Forcing)(xyz) = fn.fn(xyz)

# TODO: Generate quadrature on the fly?
quadrature(fn::Forcing) = zip(weights(fn.quadrature), locations(fn.quadrature))
