export Element, element_matrix, dofs_per_node

struct Element{Basis<:AbstractBasis}
    # Element matrix
    material::AbstractMaterial
    # Quadrature for the integration
    quadrature::Quadrature
    # Basis on which the element is defined
    basis::Basis
end

basis(::Heat, ::Mesh{Dim}) where {Dim} = ScalarBasis(1)
basis(::Elastic, ::Mesh{Dim}) where {Dim} = VectorBasis(Dim)

function Element(material::AbstractMaterial, mesh::Mesh{Dim}) where {Dim}
    # XXX: The quadrature cannot be set yet externally.
    quadrature = Quadrature(Dim, 1)
    b = basis(material, mesh)
    return Element(material, quadrature, b)
end

basis(element::Element) = element.basis
quadrature(element::Element) = element.quadrature()
dofs_per_node(element::Element) = dofs_per_node(element.basis)

# Wraps shape function derivative with Jacobian mapping and expansion.
function shape_dfunction(mesh, basis, xyz)
    b = shape_dfunction(xyz)
    J = jacobian(mesh, xyz)
    B = inv(J) * b
    return expand(mesh, basis, B)
end

function jacobian(mesh, xyz)
    B = shape_dfunction(xyz)
    XYZ = measure(mesh, ntuple(x -> 1, element_dimension(mesh)))
    return B * XYZ
end

function element_matrix(element::Element, mesh::Mesh, quadrature_xyz)
    J = jacobian(mesh, quadrature_xyz)
    B = shape_dfunction(mesh, element.basis, quadrature_xyz)
    D = constitutive(element.material, mesh)
    return det(J) * B' * D * B
end
