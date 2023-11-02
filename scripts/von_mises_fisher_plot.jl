using Plots
using LaTeXStrings
using Distributions

pgfplotsx()
##

θ = range(0, π, length=100)
ϕ = range(0, 2*π, length=200)

x = [sin(t)*cos(p) for t in θ, p in ϕ];
y = [sin(t)*sin(p) for t in θ, p in ϕ];
z = [cos(t) for t in θ, p in ϕ];

##

normed(x) = x ./ norm(x)
ppdf(θ, ϕ) = pdf(VonMisesFisher(normed([1.0, -1.0, 0.5]), 50), [sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ)])

begin
	θ_ = [t for t in θ, p in ϕ]
	ϕ_ = [p for t in θ, p in ϕ]
end

F = ppdf.(θ_, ϕ_)

##
Plots.plot(x, y, z, surfacecolor=F, st=:surface, colorbar=false)
# savefig("scripts/master_thesis_figures/von_mises50.pdf")