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
    for i = 2:length(xs)
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
function convergence_rate(forcing, boundary, f, df)
    nsteps = 8

    nels = map(i -> 10 * 2^(i - 1), 1:nsteps)
    dxs = zeros(length(nels))

    l2_norm = zeros(length(nels))
    energy_norm = zeros(length(nels))

    for (i, nel) in enumerate(nels)
        mesh = Mesh((1,), (nel,))
        Ke = element_matrix(mesh)

        # Filter prescribed boundary nodes.
        fixed = prescribe(mesh, boundary)

        # Obtain solution field.
        u = solve(mesh, Ke, forcing, fixed)

        # TODO: Update equivalent mesh size for higher dimensions.
        dxs[i] = min(mesh.dx...)

        # L2 norm.
        u_exact = interpolate(mesh, Ke, f)
        u_h = interpolate(mesh, Ke, u)

        # Energy norm.
        du_exact = interpolate(mesh, Ke, df)
        du_h = derivative(mesh, Ke, u)

        l2_norm[i] = sqrt(sum((u_exact - u_h) .^ 2) / sum(u_exact .^ 2))
        energy_norm[i] = sqrt(sum((du_exact - du_h) .^ 2) / sum(du_exact .^ 2))
    end

    dxs, l2_norm, energy_norm
end
