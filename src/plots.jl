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

# These are defined solely for plotting purposes. At some time the plots will
# be handled differently and should probably be moved to example files.
omega = 2pi
forcing(xyz) = sin(omega * xyz[1]) * omega^2
solution(xyz) = sin(omega * xyz[1])
dsolution(xyz) = cos(omega * xyz[1]) * omega

function plot_solution(nelem)
    mesh = MinX.Mesh((1,), (nelem,))
    Ke = element_matrix(mesh)
    u = MinX.solve(mesh, Ke, forcing)

    f = Figure()
    Axis(f[1, 1], xlabel = L"x", ylabel = L"T(x)")

    xs = collect(0:nelem) .* MinX.measure(Ke)
    scatterlines!(xs, solution, label = "Analytical", markersize = 0)
    scatterlines!(xs, u, label = "Numerical")
    axislegend()
    f
end

function plot_dsolution(nelem)
    mesh = MinX.Mesh((1,), (nelem,))
    Ke = element_matrix(mesh)
    u = MinX.solve(mesh, Ke, forcing)
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
