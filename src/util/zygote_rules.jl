using Zygote: @adjoint, accum, AContext


# This context doesn't allow any globals.
struct NoContext <: Zygote.AContext end

# Stupid implementation to obtain type-stability.
Zygote.cache(cx::NoContext) = (cache_fields=nothing)

# Stupid implementation.
Base.haskey(cx::NoContext, x) = false

Zygote.accum_param(::NoContext, x, Δ) = Δ

function context_free_gradient(f, args...)
    _, pb = Zygote._pullback(NoContext(), f, args...)
    return pb(1.0)
end


Zygote.accum(as::Tuple...) = map(accum, as...)

# Not a rule, but a helpful utility.
show_grad_type(x, S) = Zygote.hook(x̄ -> ((@show S, typeof(x̄)); x̄), x)

@adjoint function SVector{D}(x::AbstractVector) where {D}
    return SVector{D}(x), Δ::AbstractVector -> (convert(typeof(x), Δ),)
end

@adjoint function SMatrix{D1, D2}(X::AbstractMatrix) where {D1, D2}
    return SMatrix{D1, D2}(X), Δ::AbstractMatrix -> (convert(typeof(X), Δ),)
end

@adjoint function SMatrix{1, 1}(a)
    return SMatrix{1, 1}(a), Δ::AbstractMatrix -> (first(Δ),)
end

# Implementation of the matrix exponential that assumes one doesn't require access to the
# gradient w.r.t. `A`, only `t`. The former is a bit compute-intensive to get at, while the
# latter is very cheap.

time_exp(A, t) = exp(A * t)
ZygoteRules.@adjoint function time_exp(A, t)
    B = exp(A * t)
    return B, Δ->(nothing, sum(Δ .*  (A * B)))
end

# THIS IS A TEMPORARY FIX WHILE I WAIT FOR #445 IN ZYGOTE TO BE MERGED.
# FOR SOME REASON THIS REALLY HELPS...
@adjoint function (::Type{T})(x, sz) where {T <: Fill}
    back(Δ::AbstractArray) = (sum(Δ), nothing)
    back(Δ::NamedTuple) = (Δ.value, nothing)
    return Fill(x, sz), back
end

@adjoint function collect(x::Fill)
    function collect_Fill_back(Δ)
        return ((value=reduce(accum, Δ), axes=nothing),)
    end
    return collect(x), collect_Fill_back
end

# Implement a restrictive-as-possible implementation of the adjoint because this is a
# dangerous operation that causes segfaults (etc) if its done wrong.
@adjoint function reinterpret(T::Type{<:SVector{1, V}}, x::Vector{V}) where {V<:Real}
    function reinterpret_back(Δ::Vector{<:SVector{1, V}})
        return (nothing, reinterpret(V, Δ))
    end
    return reinterpret(T, x), reinterpret_back
end

@adjoint function reinterpret(T::Type{V}, x::Vector{<:SVector{1, V}}) where {V<:Real}
    function reinterpret_back(Δ::Vector{V})
        return (nothing, reinterpret(SVector{1, V}, Δ))
    end
    return reinterpret(V, x), reinterpret_back
end

@adjoint function Base.StepRangeLen(ref, step, len::Integer, offset::Integer = 1)
    function StepRangeLen_pullback(Δ::NamedTuple)
        return (Δ.ref, Δ.step, nothing, nothing)
    end
    return StepRangeLen(ref, step, len, offset), StepRangeLen_pullback
end

@adjoint function Base.:*(a::Real, x::StepRangeLen)
    function mul_Real_StepRangeLen_adjoint(Δ)
        Δref = Δ.ref === nothing ? zero(a) : a * Δ.ref
        Δstep = Δ.step === nothing ? zero(a) : a * Δ.step
        return (Δref * x.ref + Δstep * x.step, (
            ref = a * Δref,
            step = a * Δstep,
            len = nothing,
            offset = nothing,
        ),)
    end
    return a * x, mul_Real_StepRangeLen_adjoint
end

@adjoint function step(x::StepRangeLen)
    return step(x), Δ -> ((ref=nothing, step=Δ, len=nothing, offset=nothing),)
end

# @adjoint function *(x::Base.TwicePrecision, v::Number)
#     function mul_TwicePrecision_Number(Δ::NamedTuple)
#         Δ_num = TwicePrecision
#     end
#     return x * v, Δ -> (Δ * v, Δ * x)
# end

# @adjoint function *(x::Real, r::StepRangeLen{<:Real, <:Base.TwicePrecision})
#     function mul_Real_StepRangeLen_adjoint(Δ::NamedTuple)
#         @show typeof(Δ.ref), typeof(r.ref), typeof(Δ.step), typeof(r.step)
#         # SOMETHING HERE TO DO WITH HANDLING TWICE-PRECISION PROPERLY.
#         return (
#             accum(
#                 Δ.ref === nothing ? nothing : Δ.ref * r.ref,
#                 Δ.step === nothing ? nothing : Δ.step * r.step,
#             ),
#             (
#                 ref = Δ.ref === nothing ? nothing : x * Δ.ref,
#                 step = Δ.step === nothing ? nothing : x * Δ.step,
#             ),
#         )
#     end
#     return x * r, mul_Real_StepRangeLen_adjoint
# end

@adjoint function BlockDiagonal(blocks::Vector)
    function BlockDiagonal_pullback(Δ::NamedTuple{(:blocks,)})
        return (Δ.blocks,)
    end
    return BlockDiagonal(blocks), BlockDiagonal_pullback
end

@adjoint function Base.map(f::Tf, x::Fill) where {Tf}
    y_el, back = Zygote._pullback(f, x.value)
    function map_Fill_pullback(Δ::NamedTuple{(:value,)})
        Δf, Δx_el = back(Δ.value)
        return Δf, (value = Δx_el,)
    end
    return Fill(y_el, size(x)), map_Fill_pullback
end

function Base.map(f::Tf, x1::Fill, x2::Fill) where {Tf}
    @assert size(x1) == size(x2)
    y_el = f(x1.value, x2.value)
    return Fill(y_el, size(x1))
end

Zygote.@adjoint function Base.map(f::Tf, x1::Fill, x2::Fill) where {Tf}
    @assert size(x1) == size(x2)
    y_el, back = Zygote._pullback(f, x1.value, x2.value)
    function map_Fill_pullback(Δ::NamedTuple{(:value,)})
        Δf, Δx1_el, Δx2_el = back(Δ.value)
        return (Δf, (value = Δx1_el,), (value = Δx2_el,))
    end
    return Fill(y_el, size(x1)), map_Fill_pullback
end

@adjoint function Base.getindex(x::Fill, n::Int)
    function getindex_FillArray(Δ)
        return ((value = Δ, axes = nothing), nothing)
    end
    return x[n], getindex_FillArray
end


#
# AD-free pullbacks for a few things. These are primitives that will be used to write the
# gradients.
#

function cholesky_pullback(Σ::Symmetric{<:Real, <:StridedMatrix})
    C = cholesky(Σ)
    return C, function(Δ::NamedTuple)
        U, Ū = C.U, Δ.factors
        Σ̄ = Ū * U'
        Σ̄ = LinearAlgebra.copytri!(Σ̄, 'U')
        Σ̄ = ldiv!(U, Σ̄)
        BLAS.trsm!('R', 'U', 'T', 'N', one(eltype(Σ)), U.data, Σ̄)

        @inbounds for n in diagind(Σ̄)
            Σ̄[n] /= 2
        end
        return (UpperTriangular(Σ̄),)
    end
end

function cholesky_pullback(S::Symmetric{<:Real, <:StaticMatrix{N, N}}) where {N}
    C = cholesky(S)
    return C, function(Δ::NamedTuple)
        U, ΔU = C.U, Δ.factors
        ΔS = U \ (U \ SMatrix{N, N}(Symmetric(ΔU * U')))'
        ΔS = ΔS - Diagonal(ΔS ./ 2)
        return (UpperTriangular(ΔS),)
    end
end

@adjoint function cholesky(S::Symmetric{<:Real, <:StaticMatrix{N, N}}) where {N}
    return cholesky_pullback(S)
end

function logdet_pullback(C::Cholesky)
    return logdet(C), function(Δ)
        return ((uplo=nothing, info=nothing, factors=Diagonal(2 .* Δ ./ diag(C.factors))),)
    end
end

AtA_pullback(A::AbstractMatrix{<:Real}) = A'A, Δ->(A * (Δ + Δ'),)


function Zygote.accum(a::UpperTriangular, b::UpperTriangular)
    return UpperTriangular(Zygote.accum(a.data, b.data))
end

function Zygote.accum(D::Diagonal{<:Real}, U::UpperTriangular{<:Real, <:SMatrix})
    return UpperTriangular(D + U.data)
end

function Zygote.accum(a::Diagonal, b::UpperTriangular)
    return UpperTriangular(a + b.data)
end

Zygote.accum(a::UpperTriangular, b::Diagonal) = Zygote.accum(b, a)

Zygote._symmetric_back(Δ::UpperTriangular{<:Any, <:SArray}, uplo) = Δ
