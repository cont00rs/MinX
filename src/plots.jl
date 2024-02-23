using MinX

# The plotting utilities are located here and kept separately from the MinX
# module. This avoids the explicit dependency on large plotting packages, e.g.
# Plots.jl, or (GL,Cairo)Makie.jl. The plots can be loaded explicitly by
# `include("src/plots.jl")` and then (seem to) rely on the plotting library in
# the global package environment.
using GLMakie

function plot_convergence_rates(f, df, x)
    xs, ys = MinX.remainder_test(f, df, x)
    plot_convergence_rates(xs, ys)
end

function plot_convergence_rates(xs, ys)
    target_slope = map(x -> (x/2)^2, xs)

    # Plotting with Makie.jl
    f = Figure()
    Axis(f[1,1], xscale=log10, yscale=log10,
         xlabel=L"\Delta x", ylabel=L"R")

    # Needs both `scatterlines!` to properly cycle colors;
    # Using `lines!` for the slope ends up same color...
    scatterlines!(xs, ys, label=L"R(f)")
    scatterlines!(xs, target_slope, label=L"\Delta 2", markersize=0)
    axislegend()
    f
end
