using MinX

# The plotting utilities are located here and kept separately from the MinX
# module. This avoids the explicit dependency on large plotting packages, e.g.
# Plots.jl, or (GL,Cairo)Makie.jl. The plots can be loaded explicitly by
# `include("src/plots.jl")` and then (seem to) rely on the plotting library in
# the global package environment.
using GLMakie
using LaTeXStrings

function plot_convergence_rates(f, df, x)
    xs, ys = MinX.remainder_test(f, df, x)
    plot_convergence_rates(xs, ys, 2)
end

function plot_convergence_rates(xs::AbstractVector, ys::AbstractVector, slope)
    dy = ys[1] / xs[1]
    target_slope = map(x -> (x * dy)^slope, xs)

    f = Figure()
    Axis(f[1, 1], xscale = log10, yscale = log10, xlabel = L"\Delta h", ylabel = L"R")
    # Needs both `scatterlines!` to properly cycle colors;
    # Using `lines!` for the slope ends up same color...
    scatterlines!(xs, ys, label = L"R(f)")
    scatterlines!(xs, target_slope, label = L"\Delta %$(slope)", markersize = 0)
    axislegend(position = :rb)
    f
end

function plot_solution3(nelem)
    mesh = MinX.Mesh((1, 1, 1), (nelem, nelem, nelem))
    Ke = element_matrix(mesh)

    forcing(xyz) = sin(pi * xyz[1]) * pi^2 * sin(pi * xyz[2]) * 2.0 * sin(pi * xyz[3]) * 1.5

    boundary = (x, y, z) -> x ≈ 0.0 || x ≈ 1.0 || y ≈ 0.0 || y ≈ 1.0 || z ≈ 0.0 || z ≈ 1.0

    fixed = prescribe(mesh, Ke, boundary)
    u = MinX.solve(mesh, Ke, forcing, fixed)

    # XXX: Improve the coord extraction for plotting.
    xs = map(node -> coords(mesh, node)[1], nodes(mesh))[:, 1, 1]
    ys = map(node -> coords(mesh, node)[2], nodes(mesh))[1, :, 1]
    zs = map(node -> coords(mesh, node)[3], nodes(mesh))[1, 1, :]

    data = reshape(u, length(xs), length(ys), length(zs))
    volume(xs, ys, zs, data)
end

function plot_solution_elastic(nelem)
    # TODO: Refactor all these plots together. Its getting out of hand.
    mesh = MinX.Mesh((1, 1), (nelem, nelem))
    Ke = element_matrix(elastic(2.5, 0.25), mesh)

    boundary =
        (x, y) -> isapprox(x, 0) || isapprox(y, 0) || isapprox(x, 1) || isapprox(y, 1)
    fixed = prescribe(mesh, Ke, 2, boundary)

    forcing(lambda, mu) =
        xyz -> [
            -pi^2 * (
                -(lambda + 3mu) * sin(pi * xyz[1]) * sin(pi * xyz[2]) +
                (lambda + mu) * cos(pi * xyz[1]) * cos(pi * xyz[2])
            ),
            -pi^2 * (
                -(lambda + 3mu) * sin(pi * xyz[1]) * sin(pi * xyz[2]) +
                (lambda + mu) * cos(pi * xyz[1]) * cos(pi * xyz[2])
            ),
        ]

    u = MinX.solve(mesh, Ke, forcing(1, 1), fixed)

    xs = map(node -> coords(mesh, node)[1], nodes(mesh))[:, 1]
    ys = map(node -> coords(mesh, node)[2], nodes(mesh))[1, :]

    dpn = 2
    data = reshape(u, dpn, length(xs), length(ys))
    us = data[1, :, :]
    vs = data[2, :, :]

    strength = vec(sqrt.(us .^ 2 .+ vs .^ 2))

    f = Figure()
    Axis(f[1, 1], xlabel = L"x", ylabel = L"y")
    arrows!(xs, ys, us, vs, lengthscale = 0.1, arrowcolor = strength, linecolor = strength)

    f
end

function plot_solution2(nelem)
    # NOTE: The `heatmap!` plot seems to generate plots with data being cell
    # centered. This is fine for data defined at/interpolated to the element
    # centers. For state data, defined at the nodes, this seems less feasible.
    # For those plots, the `contourf` option seems best, although it is to be
    # tested what number of levels look OK and if these lines do not interfere
    # with the data.
    #
    # NOTE: For non-scalar data a good option might be to use `arrow` plots on
    # top of a contour of the absolute/norm values?

    mesh = MinX.Mesh((1, 1), (nelem, nelem))

    Ke = element_matrix(mesh)

    forcing(xyz) = sin(pi * xyz[1]) * pi^2 * sin(pi * xyz[2]) * 2
    solution(xyz) = sin(pi * xyz[1]) * sin(pi * xyz[2])
    dsolution(xyz) = 0
    boundary =
        (x, y) -> isapprox(x, 0) || isapprox(x, 1) || isapprox(y, 0) || isapprox(y, 1)

    #fixed = prescribe(mesh, (x, y) -> isapprox(x, 0))
    fixed = prescribe(mesh, Ke, boundary)
    u = MinX.solve(mesh, Ke, forcing, fixed)

    xs = map(node -> coords(mesh, node)[1], nodes(mesh))[:, 1]
    ys = map(node -> coords(mesh, node)[2], nodes(mesh))[1, :]

    #data = reshape(1:(nelem+1)^2, length(xs), length(ys))
    data = reshape(u, length(xs), length(ys))

    f = Figure()
    Axis(f[1, 1], xlabel = L"x", ylabel = L"y")
    #co = heatmap!(xs, ys, data)
    co = contourf!(xs, ys, data, levels = 25)
    Colorbar(f[1, 2], co, label = L"T")
    f
end

# These are defined solely for plotting purposes. At some time the plots will
# be handled differently and should probably be moved to example files.
omega = 2pi
forcing(xyz) = sin(omega * xyz[1]) * omega^2
solution(xyz) = sin(omega * xyz[1])
dsolution(xyz) = cos(omega * xyz[1]) * omega

function plot_solution(nelem)
    mesh = MinX.Mesh((1,), (nelem,))
    Ke = element_matrix(mesh)
    fixed = prescribe(mesh, Ke, x -> (isapprox(x, 0) || isapprox(x, 1)))
    u = MinX.solve(mesh, Ke, forcing, fixed)

    f = Figure()
    Axis(f[1, 1], xlabel = L"x", ylabel = L"T(x)")

    xs = map(node -> coords(mesh, node)[1], nodes(mesh))
    scatterlines!(xs, solution, label = "Analytical", markersize = 0)
    scatterlines!(xs, u, label = "Numerical")
    axislegend()
    f
end

function plot_dsolution(nelem)
    mesh = MinX.Mesh((1,), (nelem,))
    Ke = element_matrix(mesh)
    fixed = prescribe(mesh, Ke, x -> (isapprox(x, 0) || isapprox(x, 1)))
    u = MinX.solve(mesh, Ke, forcing, fixed)
    du = derivative(mesh, Ke, u)

    f = Figure()
    Axis(f[1, 1], xlabel = L"x", ylabel = L"T(x)")

    # TODO: This should probably be retrieved through the mesh/basis.
    xs = collect(1:nelem) .* MinX.measure(Ke)
    scatterlines!(xs, dsolution, label = "Analytic", markersize = 0)
    scatterlines!(xs, du, label = "Numerical")
    axislegend()
    f
end
