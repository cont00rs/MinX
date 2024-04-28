using Printf

export check_convergence_rate, check_sensitivity
export convergence_rate, convergence_rates

function check_sensitivity(f, df, x)
    deltas, errors = remainder_test(f, df, x)
    check_convergence_rate(deltas, errors)
end

# Taylor remainder convergence test
# http://www.dolfin-adjoint.org/en/latest/documentation/verification.html
function remainder(f, df, x, eps)
    Jp = f(x + eps)
    Jx = f(x)
    dJdx = df(x)
    return abs(Jp - Jx - eps * dJdx)
end

function remainder_test(f, df, x)
    nsteps = 4
    dx = map(x -> 0.01 / 2^(x - 1), 1:nsteps)
    errors = map(dx -> remainder(f, df, x, dx), dx)
    return dx, errors
end

function convergence_rates(xs, ys)
    rates = [0.0]
    for i in Iterators.drop(eachindex(xs), 1)
        dx = log(xs[i] / xs[i-1])
        dy = log(ys[i] / ys[i-1])
        push!(rates, dy / dx)
    end
    return rates
end

function check_convergence_rate(deltas, errors)
    rates = convergence_rates(deltas, errors)
    @printf("%12s,%12s,%12s\n", "Delta", "Error", "Rate")
    for (d, e, r) in zip(deltas, errors, rates)
        @printf("%12.3e,%12.3e,%12.3e\n", d, e, r)
    end
end

# TODO forcing should become a problem description of some sort
function convergence_rate(dim, material, forcing, boundary, f, df)
    nsteps = 4

    # Restrict 3D problems to smaller ultimate mesh size.
    nel0 = dim == 3 ? 4 : 10
    nels = map(i -> nel0 * 2^(i - 1), 1:nsteps)
    dxs = zeros(length(nels))

    l2_norm = zeros(length(nels))
    energy_norm = zeros(length(nels))

    for (i, nel) in enumerate(nels)
        mesh = Mesh(tuple(ones(dim)...), tuple((ones(dim) .* nel)...))
        Ke = Element(material, mesh)

        # Filter prescribed boundary nodes.
        fixed = prescribe(mesh, basis(Ke), boundary)

        # Obtain solution field.
        u = solve(mesh, Ke, forcing, fixed)

        # Consider shortest element edge length as equivalent mesh size.
        dxs[i] = min(mesh.dx...)

        # L2 norm.
        u_h = interpolate(mesh, u)
        u_exact = similar(u_h)
        for i in axes(u_exact, 1)
            u_exact[i, :, :] = interpolate(mesh, Forcing(f[i], basis(Ke), dim))
        end

        l2_norm[i] = sqrt(sum((u_exact - u_h) .^ 2) / sum(u_exact .^ 2))

        # Energy norm.
        du_h = derivative(mesh, basis(Ke), u)
        du_exact = similar(du_h)
        for i in axes(du_exact, 1)
            du_exact[i, :, :] = interpolate(mesh, Forcing(df[i], basis(Ke), dim))
        end

        # XXX: Isn't this typically combined with constitutive too?
        energy_norm[i] = sqrt(sum((du_exact - du_h) .^ 2) / sum(du_exact .^ 2))
    end

    dxs, l2_norm, energy_norm
end
