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
function KernelFunctions.kerneldiagmatrix(
    k::Separable, x::AbstractVector{<:Tuple{Any, Any}},
)
    return kerneldiagmatrix(k.l, first.(x)) .* kerneldiagmatrix(k.r, last.(x))
end
function KernelFunctions.kernelmatrix(
    k::Separable, x::AbstractVector{<:Tuple{Any, Any}},
)
    return kernelmatrix(k.l, first.(x)) .* kernelmatrix(k.r, last.(x))
end

# Binary methods.
function KernelFunctions.kerneldiagmatrix(
    k::Separable,
    x::AbstractVector{<:Tuple{Any, Any}},
    y::AbstractVector{<:Tuple{Any, Any}},
)
    return kerneldiagmatrix(k.l, first.(x), first.(y)) .*
        kerneldiagmatrix(k.r, last.(x), last.(y))
end
function KernelFunctions.kernelmatrix(
    k::Separable,
    x::AbstractVector{<:Tuple{Any, Any}},
    y::AbstractVector{<:Tuple{Any, Any}},
)
    return kernelmatrix(k.l, first.(x), first.(y)) .* kernelmatrix(k.r, last.(x), last.(y))
end
