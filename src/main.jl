using Printf

function remainder(f, df, x, eps)
    Jp = f(x + eps)
    Jx = f(x)
    dJdx = df(x)
    return abs(Jp - Jx - eps * dJdx)
end

function remainder_test(f, df, x)
    nsteps = 8
    dx = map(x -> 0.01 / 2^(x - 1), 1:nsteps)
    errors = map(dx -> remainder(f, df, x, dx), dx)
    return dx, errors
end

function convergence_rates(xs, ys)
    rates = []
    for i = 2:length(xs)
        dx = log(abs(xs[i] / xs[i-1]))
        dy = log(abs(ys[i] / ys[i-1]))
        push!(rates, dy / dx)
    end
    return rates
end

function check_sensitivity(f, df, x)
    deltas, errors = remainder_test(f, df, x)
    rates = convergence_rates(deltas, errors)
    @printf("%12s,%12s,%12s\n", "Delta", "Error", "Rate")
    for (d, e, r) in zip(deltas, errors, rates)
        @printf("%12.3e,%12.3e,%12.3e\n", d, e, r)
    end
end
