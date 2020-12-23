using Zygote: @adjoint, accum, AContext


# This context doesn't allow any globals.
struct NoContext <: Zygote.AContext end

# Stupid implementation to obtain type-stability.
Zygote.cache(cx::NoContext) = (cache_fields=nothing)

# Stupid implementation.
Base.haskey(cx::NoContext, x) = false

Zygote.accum_param(::NoContext, x, Δ) = Δ

@inline Zygote.accum(as::Tuple...) = map(accum, as...)

@inline Zygote.accum(as::AbstractArray...) = map(accum, as...)

@inline Zygote.accum(a::Tuple, b::Tuple, c::Nothing) = map(accum, a, b)

@adjoint function SVector{D}(x::AbstractVector) where {D}
    return SVector{D}(x), Δ::AbstractVector -> (convert(typeof(x), Δ),)
end

function Zygote._pullback(::AContext, ::Type{<:SVector{1}}, x::Real)
    SVector_pullback(Δ::SVector{1}) = (nothing, only(Δ))
    return SVector{1}(x), SVector_pullback
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

@adjoint function step(x::StepRangeLen)
    return step(x), Δ -> ((ref=nothing, step=Δ, len=nothing, offset=nothing),)
end

@adjoint function BlockDiagonal(blocks::Vector)
    function BlockDiagonal_pullback(Δ::NamedTuple{(:blocks,)})
        return (Δ.blocks,)
    end
    return BlockDiagonal(blocks), BlockDiagonal_pullback
end

@adjoint function Base.map(f::Tf, x::Fill) where {Tf}
    y_el, back = Zygote._pullback(__context__, f, x.value)
    function map_Fill_pullback(Δ::NamedTuple)
        Δf, Δx_el = back(Δ.value)
        return Δf, (value = Δx_el, axes=nothing)
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
    y_el, back = Zygote._pullback(__context__, f, x1.value, x2.value)
    function map_Fill_pullback(Δ::NamedTuple)
        Δf, Δx1_el, Δx2_el = back(Δ.value)
        return (Δf, (value = Δx1_el, axes=nothing), (value = Δx2_el, axes=nothing))
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

function cholesky_rrule(Σ::Symmetric{<:Real, <:StridedMatrix})
    C = cholesky(Σ)
    function cholesky_pullback(Δ::NamedTuple)
        U, Ū = C.U, Δ.factors
        Σ̄ = Ū * U'
        Σ̄ = LinearAlgebra.copytri!(Σ̄, 'U')
        Σ̄ = ldiv!(U, Σ̄)
        BLAS.trsm!('R', 'U', 'T', 'N', one(eltype(Σ)), U.data, Σ̄)

        for n in diagind(Σ̄)
            Σ̄[n] /= 2
        end
        return (UpperTriangular(Σ̄),)
    end
    return C, cholesky_pullback
end

function cholesky_rrule(S::Symmetric{<:Real, <:StaticMatrix{N, N}}) where {N}
    C = cholesky(S)
    function cholesky_pullback(Δ::NamedTuple)
        U, ΔU = C.U, Δ.factors
        ΔS = U \ (U \ SMatrix{N, N}(Symmetric(ΔU * U')))'
        ΔS = ΔS - Diagonal(ΔS ./ 2)
        return ((data=UpperTriangular(ΔS), ),)
    end
    return C, cholesky_pullback
end

@adjoint function cholesky(S::Symmetric{<:Real, <:StaticMatrix{N, N}}) where {N}
    return cholesky_rrule(S)
end

function logdet_pullback(C::Cholesky)
    return logdet(C), function(Δ)
        return ((uplo=nothing, info=nothing, factors=Diagonal(2 .* Δ ./ diag(C.factors))),)
    end
end

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
Zygote._symmetric_back(Δ::SArray, uplo) = Δ


# Temporary hacks.

using Zygote: literal_getproperty, literal_indexed_iterate, literal_getindex

function Zygote._pullback(::NoContext, ::typeof(literal_getproperty), C::Cholesky, ::Val{:U})
    function literal_getproperty_pullback(Δ)
        return (nothing, (uplo=nothing, info=nothing, factors=UpperTriangular(Δ)))
    end
    literal_getproperty_pullback(Δ::Nothing) = nothing
    return literal_getproperty(C, Val(:U)), literal_getproperty_pullback
end


function Zygote._pullback(cx::AContext, ::typeof(literal_indexed_iterate), xs::Tuple, ::Val{i}) where i
  y, b = Zygote._pullback(cx, literal_getindex, xs, Val(i))
  back(::Nothing) = nothing
  back(ȳ) = b(ȳ[1])
  (y, i+1), back
end

function Zygote._pullback(cx::AContext, ::typeof(literal_indexed_iterate), xs::Tuple, ::Val{i}, st) where i
  y, b = Zygote._pullback(cx, literal_getindex, xs, Val(i))
  back(::Nothing) = nothing
  back(ȳ) = (b(ȳ[1])..., nothing)
  (y, i+1), back
end

Zygote._pullback(cx::AContext, ::typeof(getproperty), x, f::Symbol) =
  Zygote._pullback(cx, Zygote.literal_getproperty, x, Val(f))

Zygote._pullback(cx::AContext, ::typeof(getfield), x, f::Symbol) =
  Zygote._pullback(cx, Zygote.literal_getproperty, x, Val(f))

Zygote._pullback(cx::AContext, ::typeof(literal_getindex), x::NamedTuple, ::Val{f}) where f =
  Zygote._pullback(cx, Zygote.literal_getproperty, x, Val(f))

Zygote._pullback(cx::AContext, ::typeof(literal_getproperty), x::Tuple, ::Val{f}) where f =
  Zygote._pullback(cx, Zygote.literal_getindex, x, Val(f))
