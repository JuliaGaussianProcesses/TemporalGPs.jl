import Stheno: ew, pw

"""
    Separable{Tl<:Kernel, Tr<:Kernel} <: Kernel

The kernel `k` given by
```julia
k((xl, xr), (yl, yr)) = k.l(xl, yl) * k.r(xr, yr)
```
"""
struct Separable{Tl<:Kernel, Tr<:Kernel} <: Kernel
    l::Tl
    r::Tr
end

# Unary methods.
ew(k::Separable, x::AV{<:Tuple{Any, Any}}) = ew(k.l, first.(x)) .* ew(k.r, last.(x))
pw(k::Separable, x::AV{<:Tuple{Any, Any}}) = pw(k.l, first.(x)) .* pw(k.r, last.(x))

# Binary methods.
function ew(k::Separable, x::AV{<:Tuple{Any, Any}}, y::AV{<:Tuple{Any, Any}})
    return ew(k.l, first.(x), first.(y)) .* ew(k.r, last.(x), last.(y))
end
function pw(k::Separable, x::AV{<:Tuple{Any, Any}}, y::AV{<:Tuple{Any, Any}})
    return pw(k.l, first.(x), first.(y)) .* pw(k.r, last.(x), last.(y))
end
