using Zygote: @adjoint, accum, AContext


# This context doesn't allow any globals.
struct NoContext <: Zygote.AContext end

# Stupid implementation to obtain type-stability.
Zygote.cache(cx::NoContext) = (cache_fields=nothing)

# Stupid implementation.
Base.haskey(cx::NoContext, x) = false

Zygote.accum_param(::NoContext, x, Δ) = Δ

nograd_pullback(Δ) = nothing

Zygote._pullback(::AContext, ::typeof(eltype), x) = eltype(x), nograd_pullback

# Hacks to help the compiler out in very specific situations.
Zygote.accum(a::Array{T}, b::Array{T}) where {T<:Real} = a + b

Zygote.accum(a::SArray{size, T}, b::SArray{size, T}) where {size, T<:Real} = a + b

Zygote.accum(a::Tuple, b::Tuple, c::Tuple) = map(Zygote.accum, a, b, c)

@adjoint function SVector{D}(x::AbstractVector) where {D}
    SVector_pullback(Δ) = (convert(typeof(x), Δ),)
    return SVector{D}(x), SVector_pullback
end

function Zygote._pullback(::AContext, ::Type{<:SVector{1}}, x::Real)
    SVector_pullback(Δ::AbstractVector) = (nothing, only(Δ))
    return SVector{1}(x), SVector_pullback
end

@adjoint function SMatrix{D1, D2}(X::AbstractMatrix) where {D1, D2}
    SMatrix_pullback(Δ::AbstractMatrix) = (convert(typeof(X), Δ), )
    return SMatrix{D1, D2}(X), SMatrix_pullback
end

@adjoint function SMatrix{1, 1}(a)
    SMatrix_pullback(Δ::AbstractMatrix) = (first(Δ), )
    return SMatrix{1, 1}(a), SMatrix_pullback
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

function Zygote._pullback(::Zygote.AContext, ::typeof(vcat), x::Zeros, y::Zeros)
    vcat_pullback(Δ) = (nothing, nothing, nothing)
    return vcat(x, y), vcat_pullback
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
    function map_Fill_pullback(Δ::Union{NamedTuple, Composite})
        if Δ isa Composite
            Δ_ = (value=Δ.value, axes=Δ.axes)
        else
            Δ_ = Δ
        end
        Δf, Δx_el = back(Δ_.value)
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
    function getindex_FillArray_pullback(Δ)
        return ((value = Δ, axes = nothing), nothing)
    end
    return x[n], getindex_FillArray_pullback
end

@adjoint function Base.getindex(x::SVector{1}, n::Int)
    getindex_SArray_pullback(Δ) = (SVector{1}(Δ), nothing)
    return x[n], getindex_SArray_pullback
end

@adjoint function Base.getindex(x::SVector{1, 1}, n::Int)
    getindex_pullback(Δ) = (SMatrix{1, 1}(Δ), nothing)
    return x[n], getindex_SArray_pullback
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
        return ((data=SMatrix{N, N}(UpperTriangular(ΔS)), ),)
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

function Zygote._pullback(::NoContext, ::typeof(*), A::Adjoint, B::AbstractMatrix)
    times_pullback(::Nothing) = nothing
    times_pullback(Δ) = nothing, Adjoint(B * Δ'), A' * Δ
    return A * B, times_pullback
end

function Zygote._pullback(::NoContext, ::typeof(literal_getproperty), C::Cholesky, ::Val{:U})
    function literal_getproperty_pullback(Δ)
        return (nothing, (uplo=nothing, info=nothing, factors=UpperTriangular(Δ)))
    end
    literal_getproperty_pullback(Δ::Nothing) = nothing
    return literal_getproperty(C, Val(:U)), literal_getproperty_pullback
end

Zygote.accum(x::Adjoint...) = Adjoint(Zygote.accum(map(parent, x)...))

Zygote.accum(x::NamedTuple{(:parent,)}, y::Adjoint) = (parent=accum(x.parent, y.parent),)

function Zygote.accum(A::UpperTriangular{<:Any, <:SMatrix{P}}, B::SMatrix{P, P}) where {P}
    return Zygote.accum(SMatrix{P, P}(A), B)
end

function Zygote.accum(B::SMatrix{P, P}, A::UpperTriangular{<:Any, <:SMatrix{P}}) where {P}
    return Zygote.accum(B, SMatrix{P, P}(A))
end

function Zygote.accum(a::Composite{T}, b::NamedTuple) where {T}
    return Zygote.accum(a, Composite{T}(; b...))
end

Base.:(+)(::Composite, ::Nothing) = Zero()

# function Zygote._pullback(cx::AContext, ::typeof(literal_indexed_iterate), xs::Tuple, ::Val{i}) where i
#   y, b = Zygote._pullback(cx, literal_getindex, xs, Val(i))
#   back(::Nothing) = nothing
#   back(ȳ) = b(ȳ[1])
#   (y, i+1), back
# end

# function Zygote._pullback(cx::AContext, ::typeof(literal_indexed_iterate), xs::Tuple, ::Val{i}, st) where i
#   y, b = Zygote._pullback(cx, literal_getindex, xs, Val(i))
#   back(::Nothing) = nothing
#   back(ȳ) = (b(ȳ[1])..., nothing)
#   (y, i+1), back
# end

# Zygote._pullback(cx::AContext, ::typeof(getproperty), x, f::Symbol) =
#   Zygote._pullback(cx, Zygote.literal_getproperty, x, Val(f))

# Zygote._pullback(cx::AContext, ::typeof(getfield), x, f::Symbol) =
#   Zygote._pullback(cx, Zygote.literal_getproperty, x, Val(f))

# Zygote._pullback(cx::AContext, ::typeof(literal_getindex), x::NamedTuple, ::Val{f}) where f =
#   Zygote._pullback(cx, Zygote.literal_getproperty, x, Val(f))

# Zygote._pullback(cx::AContext, ::typeof(literal_getproperty), x::Tuple, ::Val{f}) where f =
#   Zygote._pullback(cx, Zygote.literal_getindex, x, Val(f))


# function Zygote._pullback(
#     ::AContext,
#     T::Type{<:StructArray{T, N, C} where {T, N, C<:NamedTuple}},
#     x::Union{Tuple, NamedTuple},
# )
#     function StructArray_pullback(Δ::NamedTuple{(:fieldarrays, )})
#         @show typeof(x), typeof(Δ.fieldarrays)
#         return (nothing, Δ.fieldarrays)
#     end
#     return T(x), StructArray_pullback
# end

function Zygote._pullback(::AContext, T::Type{<:StructArray}, x::Tuple)
    function StructArray_pullback(Δ::NamedTuple{(:fieldarrays, )})
        return (nothing, values(Δ.fieldarrays))
    end
    return T(x), StructArray_pullback
end

# `getproperty` accesses the `fieldarrays` field of a `StructArray`. This rule makes that
# explicit. 
function Zygote._pullback(
    ctx::AContext, ::typeof(Zygote.literal_getproperty), x::StructArray, ::Val{p},
) where {p}
    value, pb = Zygote._pullback(
        ctx, Zygote.literal_getproperty, getfield(x, :fieldarrays), Val(p),
    )
    function literal_getproperty_pullback(Δ)
        return nothing, (fieldarrays=pb(Δ)[2], ), nothing
    end
    return value, literal_getproperty_pullback
end

function time_ad(label::String, f, x...)
    println("primal: ", label)
    return @time f(x...)
end

time_ad(::Val{:disabled}, label::String, f, x...) = f(x...)

function Zygote._pullback(ctx::AContext, ::typeof(time_ad), label::String, f, x...)
    println("Forward: ", label)
    out, pb = @time Zygote._pullback(ctx, f, x...)
    function time_ad_pullback(Δ)
        println("Pullback: ", label)
        Δinputs = @time pb(Δ)
        return (nothing, nothing, Δinputs...)
    end
    return out, time_ad_pullback
end

function Zygote._pullback(ctx::AContext, ::typeof(\), A::Diagonal{<:Real}, x::Vector{<:Real})
    out, pb = Zygote._pullback(ctx, (a, x) -> a .\ x, diag(A), x)
    function ldiv_pullback(Δ)
        if Δ === nothing
            return nothing
        else
            _, Δa, Δx = pb(Δ)
            return nothing, Diagonal(Δa), Δx
        end
    end
    return out, ldiv_pullback
end

function Zygote._pullback(ctx::AContext, ::typeof(\), A::Diagonal{<:Real}, x::Matrix{<:Real})
    out, pb = Zygote._pullback(ctx, (a, x) -> a .\ x, diag(A), x)
    function ldiv_pullback(Δ)
        if Δ === nothing
            return nothing
        else
            _, Δa, Δx = pb(Δ)
            return nothing, Diagonal(Δa), Δx
        end
    end
    return out, ldiv_pullback
end

using Base.Broadcast: broadcasted

function Zygote._pullback(
    ::AContext, ::typeof(broadcasted), ::typeof(\), a::Vector{<:Real}, x::Vector{<:Real},
)
    y = a .\ x
    # function broadcast_ldiv_pullback(Δ::Nothing)
    #     return nothing
    # end
    function broadcast_ldiv_pullback(Δ::Union{Nothing, Vector{<:Real}})
        if Δ === nothing
            return nothing
        else
            return nothing, nothing, -(Δ .* y ./ a), a .\ Δ
        end
    end
    return y, broadcast_ldiv_pullback
end

function Zygote._pullback(
    ::AContext, ::typeof(broadcasted), ::typeof(\), a::Vector{<:Real}, x::Matrix{<:Real},
)
    y = a .\ x
    # broadcast_ldiv_pullback(::Nothing) = nothing
    function broadcast_ldiv_pullback(
        Δ::Union{Nothing, Matrix{<:Real}, Adjoint{<:Real, <:Matrix{<:Real}}},
    )
        if Δ === nothing
            return nothing
        else
            return nothing, nothing, -vec(sum(Δ .* y ./ a; dims=2)), a .\ Δ
        end
    end
    return y, broadcast_ldiv_pullback
end
