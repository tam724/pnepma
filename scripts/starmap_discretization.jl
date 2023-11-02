include("../pn_solver/pn_solver.jl")
using Plots


##
pgfplotsx()

##
default(legendfontsize=15)
gspec = make_gridspec((2, 2, 2), (0., 1.), (0., 1.), (0., 1.))

function get_markershape(::Type{EvenOddClassification{Even, A, B}}) where {A, B}
    return :circle
end

function get_markershape(::Type{EvenOddClassification{Odd, A, B}}) where {A, B}
    return :diamond
end

function get_markercolor(::Type{EvenOddClassification{A, Even, Even}}) where {A}
    return 1
end

function get_markercolor(::Type{EvenOddClassification{A, Odd, Even}}) where {A}
    return 2
end

function get_markercolor(::Type{EvenOddClassification{A, B, Odd}}) where {A, B}
    return nothing
end

function get_markerstrokecolor(::Type{EvenOddClassification{A, Even, B}}) where {A, B}
    return 1
end

function get_markerstrokecolor(::Type{EvenOddClassification{A, Odd, B}}) where {A, B}
    return 2
end

function get_markeralpha(::Type{EvenOddClassification{A, B, Even}}) where {A, B}
    return 1.0
end

function get_markeralpha(::Type{EvenOddClassification{A, B, Odd}}) where {A, B}
    return 0.0
end

function get_markerstrokealpha(::Type{EvenOddClassification{A, B, Even}}) where {A, B}
    return 0.0
end

function get_markerstrokealpha(::Type{EvenOddClassification{A, B, Odd}}) where {A, B}
    return 1.0
end

function get_markersize(::Type{EvenOddClassification{Even, A, B}}) where {A, B}
    return 6
end

function get_markersize(::Type{EvenOddClassification{Odd, A, B}}) where {A, B}
    return 7
end

get_label(::Type{Even}) = "e"
get_label(::Type{Odd}) = "o"
get_label(::Type{EvenOddClassification{A, B, C}}) where {A, B, C} = string(get_label.((A, B, C))...)

p = Plots.plot(figsize=(300, 200), aspect_ratio=:equal, xlims=(-0.6, 1.6), ylims=(-0.6, 1.6), zlims=(-0.6, 1.6), background_color_legend=:transparent)
all_pts = []
for eo in all_eos()
    ps_ = collect(points(eo, gspec))
    for point in ps_
        push!(all_pts, (point, eo))
    end
end
f(el) = el[1][2]

all_pts = sort(all_pts, by=f, rev=true)

comp(y) = 0.5*abs((2.5 - y)/2.) + 0.5
for (point, eo) in all_pts
    ps_ = [point]
    p = Plots.scatter!(getindex.(ps_, 1)[:], getindex.(ps_, 2)[:], getindex.(ps_, 3)[:], markershape=get_markershape(eo), markercolor=get_markercolor(eo), markersize=get_markersize(eo), markeralpha=get_markeralpha(eo), markerstrokecolor=get_markerstrokecolor(eo), markerstrokewidth=2, markerstrokealpha=get_markerstrokealpha(eo), label=nothing)
end

for (eo, name) in zip(all_eos(), ["eee", "eeo", "eoe", "eoo", "oee", "oeo", "ooe", "ooo"])
    Plots.scatter!([-5], [-5], [-5], markershape=get_markershape(eo), markercolor=get_markercolor(eo), markersize=get_markersize(eo), markeralpha=get_markeralpha(eo), markerstrokecolor=get_markerstrokecolor(eo), markerstrokewidth=2, markerstrokealpha=get_markerstrokealpha(eo), label=name)
end

clr = :gray
p = Plots.plot!([0., 0., 0., 0., 0.], [0., 1., 1., 0., 0.], [0., 0., 1., 1., 0.], color=clr, width=1, label=:none)
p = Plots.plot!([1., 1., 1., 1., 1.], [0., 1., 1., 0., 0.], [0., 0., 1., 1., 0.], color=clr, width=1, label=:none)
p = Plots.plot!([1., 0.], [1., 1.], [1., 1.], color=clr, width=1, label=:none)
p = Plots.plot!([1., 0.], [0., 0.], [0., 0.], color=clr, width=1, label=:none)
p = Plots.plot!([1., 0.], [1., 1.], [0., 0.], color=clr, width=1, label=:none)
p = Plots.plot!([1., 0.], [0., 0.], [1., 1.], color=clr, width=1, label=:none) #, camera=(10, 15))
p = Plots.plot!(aspect_ratio=:equal, grid=true, camera=(-10, 15), ticks=nothing)

p
savefig("scripts/master_thesis_figures/starmap_grid.pdf")