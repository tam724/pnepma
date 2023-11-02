"""
finds the index in the range object, that is smaller than x.
x is between range[i] and range[i + 1]
"""
function find_index_smaller_and_partition(rg::StepRangeLen{T}, x) where T
    if length(rg) == 1
        @assert isapproxzero(x - first(rg))
        return 1, zero(T)
    end
    ref = T(rg.ref)
    step = T(rg.step)
    d = x - ref
    i = Int64(floor(d/step))
    return i + rg.offset, (d - i*step)/step
end
##

function find_index_smaller_and_partition(rg::Vector{T}, x) where T
    if length(rg) == 1
        @assert isapproxzero(x - first(rg))
        return 1, zero(T)
    end 
    i = 1
    while x > rg[i+1] && i < length(rg) - 1
        i = i+1
    end
    # finds the position i inside the vector rg left of x
    d = (x - rg[i])/(rg[i+1] - rg[i])
    if d < 0.
        i = i - 1
    elseif d > 1.0
        i = i + 1
    end
    # now it is similar to the StepRangeLen function
    return i, d
end

##
function find_index_smaller(rg::StepRangeLen{T}, x) where T
    d = x - rg.ref
    i = Int64(floor(T(d/rg.step)))
    return i + rg.offset
end

function isapproxzero(a; val=1e-20)
    return abs(a) < val
end

function bilinear_interpolation(rg_x, rg_y, rg_z, vals, embedding, x, y, z)
    res = zeros(vals[1])
    bilinear_interpolation!(res, rg_x, rg_y, rg_z, vals, embedding, x, y, z)
    return res
end

@inline function val_or_embedding(vals, embedding, ix, iy, iz, nx, ny, nz)
    if ix < 1 || ix > nx || iy < 1 || iy > ny || iz < 1 || iz > nz
        return embedding
    else
        return vals[ix, iy, iz]
    end
end

function bilinear_interpolation!(res, rg_x, rg_y, rg_z, vals, embedding, x, y, z)
    ix, px = find_index_smaller_and_partition(rg_x, x)
    iy, py = find_index_smaller_and_partition(rg_y, y)
    iz, pz = find_index_smaller_and_partition(rg_z, z)
    if isapproxzero(px) && isapproxzero(py) && isapproxzero(pz)
        res .= val_or_embedding(vals, embedding, ix, iy, iz, length(rg_x), length(rg_y), length(rg_z))
        return
    elseif isapproxzero(px) && isapproxzero(py)
        res .= (1. - pz).*val_or_embedding(vals, embedding, ix, iy, iz, length(rg_x), length(rg_y), length(rg_z)) .+
                (  pz   ).*val_or_embedding(vals, embedding, ix, iy, iz + 1, length(rg_x), length(rg_y), length(rg_z))
        return
    elseif isapproxzero(py) && isapproxzero(pz)
        res .= (1. - px).*val_or_embedding(vals, embedding, ix, iy, iz, length(rg_x), length(rg_y), length(rg_z)) .+
                (  px   ).*val_or_embedding(vals, embedding, ix + 1, iy, iz, length(rg_x), length(rg_y), length(rg_z))
        return
    elseif isapproxzero(px) && isapproxzero(pz)
        res .=  (1. - py).*val_or_embedding(vals, embedding, ix, iy, iz, length(rg_x), length(rg_y), length(rg_z)) .+
                (  py   ).*val_or_embedding(vals, embedding, ix, iy + 1, iz, length(rg_x), length(rg_y), length(rg_z))
        return
    elseif isapproxzero(px)
        res .=  (1. - py)*(1. - pz).*val_or_embedding(vals, embedding, ix, iy, iz, length(rg_x), length(rg_y), length(rg_z)) .+
                (  py   )*(1. - pz).*val_or_embedding(vals, embedding, ix, iy + 1, iz, length(rg_x), length(rg_y), length(rg_z)) .+
                (1. - py)*(  pz   ).*val_or_embedding(vals, embedding, ix, iy, iz + 1, length(rg_x), length(rg_y), length(rg_z)) .+
                (  py   )*(  pz   ).*val_or_embedding(vals, embedding, ix, iy + 1, iz + 1, length(rg_x), length(rg_y), length(rg_z))
        return
    elseif isapproxzero(py)
        res .=  (1. - px)*(1. - pz).*val_or_embedding(vals, embedding, ix, iy, iz, length(rg_x), length(rg_y), length(rg_z)) .+
                (  px   )*(1. - pz).*val_or_embedding(vals, embedding, ix + 1, iy, iz, length(rg_x), length(rg_y), length(rg_z)) .+
                (1. - px)*(  pz   ).*val_or_embedding(vals, embedding, ix, iy, iz + 1, length(rg_x), length(rg_y), length(rg_z)) .+
                (  px   )*(  pz   ).*val_or_embedding(vals, embedding, ix + 1, iy, iz + 1, length(rg_x), length(rg_y), length(rg_z))
        return
    elseif isapproxzero(pz)
        res .=  (1. - px)*(1. - py).*val_or_embedding(vals, embedding, ix, iy, iz, length(rg_x), length(rg_y), length(rg_z)) .+
                (  px   )*(1. - py).*val_or_embedding(vals, embedding, ix + 1, iy, iz, length(rg_x), length(rg_y), length(rg_z)) .+
                (1. - px)*(  py   ).*val_or_embedding(vals, embedding, ix, iy + 1, iz, length(rg_x), length(rg_y), length(rg_z)) .+
                (  px   )*(  py   ).*val_or_embedding(vals, embedding, ix + 1, iy + 1, iz, length(rg_x), length(rg_y), length(rg_z))
        return
    else
        res .=  (1. - px)*(1. - py)*(1. - pz).*val_or_embedding(vals, embedding, ix, iy, iz, length(rg_x), length(rg_y), length(rg_z)) .+
                (  px   )*(1. - py)*(1. - pz).*val_or_embedding(vals, embedding, ix + 1, iy, iz, length(rg_x), length(rg_y), length(rg_z)) .+
                (1. - px)*(  py   )*(1. - pz).*val_or_embedding(vals, embedding, ix, iy + 1, iz, length(rg_x), length(rg_y), length(rg_z)) .+
                (1. - px)*(1. - py)*(  pz   ).*val_or_embedding(vals, embedding, ix, iy, iz + 1, length(rg_x), length(rg_y), length(rg_z)) .+
                (  px   )*(  py   )*(1. - pz).*val_or_embedding(vals, embedding, ix + 1, iy + 1, iz, length(rg_x), length(rg_y), length(rg_z)) .+
                (  px   )*(1. - py)*(  pz   ).*val_or_embedding(vals, embedding, ix + 1, iy, iz + 1, length(rg_x), length(rg_y), length(rg_z)) .+
                (1. - px)*(  py   )*(  pz   ).*val_or_embedding(vals, embedding, ix, iy + 1, iz + 1, length(rg_x), length(rg_y), length(rg_z)) .+
                (  px   )*(  py   )*(  pz   ).*val_or_embedding(vals, embedding, ix + 1, iy + 1, iz + 1, length(rg_x), length(rg_y), length(rg_z)) 
        return
    end
end

function bilinear_interpolation_no_embedding!(res, rg_x, rg_y, rg_z, vals, x, y, z)
    ix, px = find_index_smaller_and_partition(rg_x, x)
    iy, py = find_index_smaller_and_partition(rg_y, y)
    iz, pz = find_index_smaller_and_partition(rg_z, z)
    if isapproxzero(px) && isapproxzero(py) && isapproxzero(pz)
        res .= vals[ix, iy, iz]
        return
    elseif isapproxzero(px) && isapproxzero(py)
        res .= (1. - pz).*vals[ix, iy, iz] .+
                (  pz   ).*vals[ix, iy, iz + 1]
        return
    elseif isapproxzero(py) && isapproxzero(pz)
        res .= (1. - px).*vals[ix, iy, iz] .+
                (  px   ).*vals[ix + 1, iy, iz]
        return
    elseif isapproxzero(px) && isapproxzero(pz)
        res .=  (1. - py).*vals[ix, iy, iz] .+
                (  py   ).*vals[ix, iy + 1, iz]
        return
    elseif isapproxzero(px)
        res .=  (1. - py)*(1. - pz).*vals[ix, iy, iz] .+
                (  py   )*(1. - pz).*vals[ix, iy + 1, iz] .+
                (1. - py)*(  pz   ).*vals[ix, iy, iz + 1] .+
                (  py   )*(  pz   ).*vals[ix, iy + 1, iz + 1]
        return
    elseif isapproxzero(py)
        res .=  (1. - px)*(1. - pz).*vals[ix, iy, iz] .+
                (  px   )*(1. - pz).*vals[ix + 1, iy, iz] .+
                (1. - px)*(  pz   ).*vals[ix, iy, iz + 1] .+
                (  px   )*(  pz   ).*vals[ix + 1, iy, iz + 1]
        return
    elseif isapproxzero(pz)
        res .=  (1. - px)*(1. - py).*vals[ix, iy, iz] .+
                (  px   )*(1. - py).*vals[ix + 1, iy, iz] .+
                (1. - px)*(  py   ).*vals[ix, iy + 1, iz] .+
                (  px   )*(  py   ).*vals[ix + 1, iy + 1, iz]
        return
    else
        res .=  (1. - px)*(1. - py)*(1. - pz).*vals[ix, iy, iz] .+
                (  px   )*(1. - py)*(1. - pz).*vals[ix + 1, iy, iz] .+
                (1. - px)*(  py   )*(1. - pz).*vals[ix, iy + 1, iz] .+
                (1. - px)*(1. - py)*(  pz   ).*vals[ix, iy, iz + 1] .+
                (  px   )*(  py   )*(1. - pz).*vals[ix + 1, iy + 1, iz] .+
                (  px   )*(1. - py)*(  pz   ).*vals[ix + 1, iy, iz + 1] .+
                (1. - px)*(  py   )*(  pz   ).*vals[ix, iy + 1, iz + 1] .+
                (  px   )*(  py   )*(  pz   ).*vals[ix + 1, iy + 1, iz + 1] 
        return
    end
end

@inline function into_val_or_embedding!(vals_, embedding_, ix, iy, iz, nx, ny, nz, res_)
    if ix < 1 || ix > nx || iy < 1 || iy > ny || iz < 1 || iz > nz
        embedding_ .+= res_
    else
        vals_[ix, iy, iz] .+= res_
    end
end

@inline function bilinear_interpolation!_pullback(rg_x, rg_y, rg_z, vals, x, y, z, res_, vals_, embedding_)
    ix, px = find_index_smaller_and_partition(rg_x, x)
    iy, py = find_index_smaller_and_partition(rg_y, y)
    iz, pz = find_index_smaller_and_partition(rg_z, z)
    if isapproxzero(px) && isapproxzero(py) && isapproxzero(pz)
        into_val_or_embedding!(vals_, embedding_, ix, iy, iz, length(rg_x), length(rg_y), length(rg_z), res_)
        return
    elseif isapproxzero(px) && isapproxzero(py)
        into_val_or_embedding!(vals_, embedding_, ix, iy, iz, length(rg_x), length(rg_y), length(rg_z), (1. - pz) .* res_)
        into_val_or_embedding!(vals_, embedding_, ix, iy, iz+1, length(rg_x), length(rg_y), length(rg_z), pz .* res_)
        return
    elseif isapproxzero(py) && isapproxzero(pz)
        into_val_or_embedding!(vals_, embedding_, ix, iy, iz, length(rg_x), length(rg_y), length(rg_z), (1. - px) .* res_)
        into_val_or_embedding!(vals_, embedding_, ix + 1, iy, iz, length(rg_x), length(rg_y), length(rg_z), px .* res_)
        return
    elseif isapproxzero(px) && isapproxzero(pz)
        into_val_or_embedding!(vals_, embedding_, ix, iy, iz, length(rg_x), length(rg_y), length(rg_z), (1. - py) .* res_)
        into_val_or_embedding!(vals_, embedding_, ix, iy + 1, iz, length(rg_x), length(rg_y), length(rg_z), py .* res_)
        return
    elseif isapproxzero(px)
        into_val_or_embedding!(vals_, embedding_, ix, iy, iz, length(rg_x), length(rg_y), length(rg_z), ((1. - py)*(1. - pz)) .* res_)
        into_val_or_embedding!(vals_, embedding_, ix, iy + 1, iz, length(rg_x), length(rg_y), length(rg_z), ((  py   )*(1. - pz)) .* res_)
        into_val_or_embedding!(vals_, embedding_, ix, iy, iz + 1, length(rg_x), length(rg_y), length(rg_z), ((1. - py)*(  pz   )) .* res_)
        into_val_or_embedding!(vals_, embedding_, ix, iy + 1, iz + 1, length(rg_x), length(rg_y), length(rg_z), ((  py   )*(  pz   )) .* res_)
        return 
    elseif isapproxzero(py)
        into_val_or_embedding!(vals_, embedding_, ix, iy, iz, length(rg_x), length(rg_y), length(rg_z), ((1. - px)*(1. - pz)) .* res_)
        into_val_or_embedding!(vals_, embedding_, ix + 1, iy, iz, length(rg_x), length(rg_y), length(rg_z), ((  px   )*(1. - pz)) .* res_)
        into_val_or_embedding!(vals_, embedding_, ix, iy, iz + 1, length(rg_x), length(rg_y), length(rg_z), ((1. - px)*(  pz   )) .* res_)
        into_val_or_embedding!(vals_, embedding_, ix + 1, iy, iz + 1, length(rg_x), length(rg_y), length(rg_z), ((  px   )*(  pz   )) .* res_)
        return
    elseif isapproxzero(pz)
        into_val_or_embedding!(vals_, embedding_, ix, iy, iz, length(rg_x), length(rg_y), length(rg_z), ((1. - px)*(1. - py)) .* res_)
        into_val_or_embedding!(vals_, embedding_, ix + 1, iy, iz, length(rg_x), length(rg_y), length(rg_z), ((  px   )*(1. - py)) .* res_)
        into_val_or_embedding!(vals_, embedding_, ix, iy + 1, iz, length(rg_x), length(rg_y), length(rg_z), ((1. - px)*(  py   )) .* res_)
        into_val_or_embedding!(vals_, embedding_, ix + 1, iy + 1, iz, length(rg_x), length(rg_y), length(rg_z), ((  px   )*(  py   )) .* res_)
        return 
    else
        into_val_or_embedding!(vals_, embedding_, ix, iy, iz, length(rg_x), length(rg_y), length(rg_z), ((1. - px)*(1. - py)*(1. - pz)) .* res_)
        into_val_or_embedding!(vals_, embedding_, ix + 1, iy, iz, length(rg_x), length(rg_y), length(rg_z), ((  px   )*(1. - py)*(1. - pz)) .* res_)
        into_val_or_embedding!(vals_, embedding_, ix, iy + 1, iz, length(rg_x), length(rg_y), length(rg_z), ((1. - px)*(  py   )*(1. - pz)) .* res_)
        into_val_or_embedding!(vals_, embedding_, ix, iy, iz + 1, length(rg_x), length(rg_y), length(rg_z), ((1. - px)*(1. - py)*(  pz   )) .* res_)
        into_val_or_embedding!(vals_, embedding_, ix + 1, iy + 1, iz, length(rg_x), length(rg_y), length(rg_z), ((  px   )*(  py   )*(1. - pz)) .* res_)
        into_val_or_embedding!(vals_, embedding_, ix + 1, iy, iz + 1, length(rg_x), length(rg_y), length(rg_z), ((  px   )*(1. - py)*(  pz   )) .* res_)
        into_val_or_embedding!(vals_, embedding_, ix, iy + 1, iz + 1, length(rg_x), length(rg_y), length(rg_z), ((1. - px)*(  py   )*(  pz   )) .* res_)
        into_val_or_embedding!(vals_, embedding_, ix + 1, iy + 1, iz + 1, length(rg_x), length(rg_y), length(rg_z), ((  px   )*(  py   )*(  pz   )) .* res_)
        return
    end
end