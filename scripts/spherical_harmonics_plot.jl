using SphericalHarmonics
using Plots
using LaTeXStrings

plotlyjs()
##

θ = range(0, π, length=100)
ϕ = range(0, 2*π, length=200)

x = [sin(t)*cos(p) for t in θ, p in ϕ];
y = [sin(t)*sin(p) for t in θ, p in ϕ];
z = [cos(t) for t in θ, p in ϕ];

##
sphh(θ, ϕ) = SphericalHarmonics.computeYlm(θ, ϕ; lmax=21, SHType=SphericalHarmonics.RealHarmonics())

begin
	θ_ = [t for t in θ, p in ϕ]
	ϕ_ = [p for t in θ, p in ϕ]
end

F = sphh.(θ_, ϕ_)

# surface!(subplot=3, x, y, z, surfacecolor=getindex.(F, Ref((0, 0))), st=:surface, colorbar=false)
# title!(L"Y^0_0")

# plot!(subplot=7, x, y, z, surfacecolor=getindex.(F, Ref((1, -1))), st=:surface, colorbar=false)
# plot!(subplot=8, x, y, z, surfacecolor=getindex.(F, Ref((1, 0))), st=:surface, colorbar=false)
# plot!(subplot=9, x, y, z, surfacecolor=getindex.(F, Ref((1, 1))), st=:surface, colorbar=false)

# plot!(subplot=11, x, y, z, surfacecolor=getindex.(F, Ref((2, -2))), st=:surface, colorbar=false)
# plot!(subplot=12, x, y, z, surfacecolor=getindex.(F, Ref((2, -1))), st=:surface, colorbar=false)
# plot!(subplot=13, x, y, z, surfacecolor=getindex.(F, Ref((2, 0))), st=:surface, colorbar=false)
# plot!(subplot=14, x, y, z, surfacecolor=getindex.(F, Ref((2, 1))), st=:surface, colorbar=false)
# plot!(subplot=15, x, y, z, surfacecolor=getindex.(F, Ref((2, 2))), st=:surface, colorbar=false)

##
plot(x, y, z, surfacecolor=getindex.(F, Ref((0, 0))), st=:surface, colorbar=false)
savefig("Y00.pdf")

plot(x, y, z, surfacecolor=getindex.(F, Ref((1, -1))), st=:surface, colorbar=false)
savefig("Y1-1.pdf")

plot(x, y, z, surfacecolor=getindex.(F, Ref((1, 0))), st=:surface, colorbar=false)
savefig("Y10.pdf")

plot(x, y, z, surfacecolor=getindex.(F, Ref((1, 1))), st=:surface, colorbar=false)
savefig("Y1+1.pdf")

plot(x, y, z, surfacecolor=getindex.(F, Ref((2, -2))), st=:surface, colorbar=false)
savefig("Y2-2.pdf")

plot(x, y, z, surfacecolor=getindex.(F, Ref((2, -1))), st=:surface, colorbar=false)
savefig("Y2-1.pdf")

plot(x, y, z, surfacecolor=getindex.(F, Ref((2, 0))), st=:surface, colorbar=false)
savefig("Y20.pdf")

plot(x, y, z, surfacecolor=getindex.(F, Ref((2, 1))), st=:surface, colorbar=false)
savefig("Y2+1.pdf")

plot(x, y, z, surfacecolor=getindex.(F, Ref((2, 2))), st=:surface, colorbar=false)
savefig("Y2+2.pdf")
