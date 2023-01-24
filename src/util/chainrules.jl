# This is all AD-related stuff. If you're looking to understand TemporalGPs, this can be
# safely ignored.

using Zygote: accum, AContext

# This context doesn't allow any globals.
struct NoContext <: Zygote.AContext end

# Stupid implementation to obtain type-stability.
Zygote.cache(::NoContext) = (; cache_fields=nothing)

# Stupid implementation.
Base.haskey(cx::NoContext, x) = false

Zygote.accum_param(::NoContext, x, Δ) = Δ

ChainRulesCore.@non_differentiable eltype(x)

# Hacks to help the compiler out in very specific situations.
Zygote.accum(a::Array{T}, b::Array{T}) where {T<:Real} = a + b

Zygote.accum(a::SArray{size, T}, b::SArray{size, T}) where {size, T<:Real} = a + b

Zygote.accum(a::Tuple, b::Tuple, c::Tuple) = map(Zygote.accum, a, b, c)

function ChainRulesCore.rrule(::RuleConfig{>:HasReverseMode}, ::Type{SArray{S, T, N, L}}, x::NTuple{L, T}) where {S, T, N, L}
    SArray_rrule(::AbstractZero) = NoTangent(), NoTangent()
    SArray_rrule(Δ::NamedTuple{(:data,)}) = NoTangent(), Δ.data
    SArray_rrule(Δ::StaticArray{S}) = NoTangent(), Δ.data
    return SArray{S, T, N, L}(x), SArray_rrule
end

function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode}, ::Type{X}, x::NTuple{L, Any},
) where {S, T, N, L, X <: SArray{S, T, N, L}}
    new_x, convert_pb = rrule_via_ad(config, StaticArrays.convert_ntuple, T, x)
    _, pb = rrule_via_ad(config, SArray{S, T, N, L}, new_x)
    SArray_rrule(::AbstractZero) = NoTangent(), NoTangent()
    SArray_rrule(Δ::SArray{S}) = SArray_rrule(Tangent{X}(data=Δ.data))
    SArray_rrule(Δ::SizedArray{S}) = SArray_rrule(Tangent{X}(data=Tuple(Δ.data)))
    SArray_rrule(Δ::AbstractVector) = SArray_rrule(Tangent{X}(data=Tuple(Δ)))
    SArray_rrule(Δ::Matrix) = SArray_rrule(Tangent{X}(data=Δ))
    function SArray_rrule(Δ::Tangent{X,<:NamedTuple{(:data,)}}) where {X}
        _, Δnew_x = pb(backing(Δ))
        _, ΔT, Δx = convert_pb(Δnew_x)
        return ΔT, Δx
    end
    return SArray{S, T, N, L}(x), SArray_rrule
end

function ChainRulesCore.rrule(::typeof(collect), x::X) where {S, T, N, L, X<:SArray{S, T, N, L}}
    collect_rrule(Δ::AbstractArray) = NoTangent(), Tangent{X}(data = ntuple(i -> Δ[i], Val(L)))
    return collect(x), collect_rrule
end

function ChainRulesCore.rrule(::typeof(vcat), A::SVector{DA}, B::SVector{DB}) where {DA, DB}
    function vcat_rrule(Δ::SVector)
        ΔA = Δ[SVector{DA}(1:DA)]
        ΔB = Δ[SVector{DB}((DA+1):(DA+DB))]
        return NoTangent(), ΔA, ΔB
    end
    return vcat(A, B), vcat_rrule
end

@non_differentiable vcat(x::Zeros, y::Zeros)

# Implementation of the matrix exponential that assumes one doesn't require access to the
# gradient w.r.t. `A`, only `t`. The former is a bit compute-intensive to get at, while the
# latter is very cheap.

time_exp(A, t) = exp(A * t)
function ChainRulesCore.rrule(::typeof(time_exp), A, t::Real)
    B = exp(A * t)
    time_exp_rrule(Ω̄) = NoTangent(), NoTangent(), sum(Ω̄ .*  (A * B))
    return B, time_exp_rrule
end

function ChainRulesCore.rrule(::Zygote.ZygoteRuleConfig, ::typeof(collect), x::F) where {F<:Fill}
    function collect_Fill_rrule(Δ)
        return NoTangent(), Tangent{F}(value=reduce(Zygote.accum, Δ), axes=NoTangent())
    end
    return collect(x), collect_Fill_rrule
end

function ChainRulesCore.rrule(::typeof(step), x::T) where {T<:StepRangeLen}
    function step_StepRangeLen_rrule(Δ)
        return NoTangent(), Tangent{T}(step=Δ)
    end
    return step(x), step_StepRangeLen_rrule
end

# We have an alternative map to avoid Zygote untouchable specialisation on map.
_map(f, args...) = map(f, args...)

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(_map), f::Tf, x::F) where {Tf,F<:Fill}
    y_el, back = ChainRulesCore.rrule_via_ad(config, f, x.value)
    function _map_Fill_rrule(Δ::Tangent)
        Δf, Δx_el = back(Δ.value)
        return NoTangent(), Δf, Tangent{F}(value = Δx_el)
    end
    return Fill(y_el, size(x)), _map_Fill_rrule
end

function _map(f, x::Fill)
    y_el = f(x.value)
    return Fill(y_el, size(x))
end

function _map(f, x1::Fill, x2::Fill)
    @assert size(x1) == size(x2)
    y_el = f(x1.value, x2.value)
    return Fill(y_el, size(x1))
end

function _map(f::Tf, x1::Fill, x2::Fill) where {Tf<:Function}
    @assert size(x1) == size(x2)
    y_el = f(x1.value, x2.value)
    return Fill(y_el, size(x1))
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(_map), f, x1::F1, x2::F2) where {F1<:Fill,F2<:Fill}
    @assert size(x1) == size(x2)
    y_el, back = ChainRulesCore.rrule_via_ad(config, f, x1.value, x2.value)
    _map_Fill_rrule(Δ::AbstractArray) = _map_Fill_rrule(Tangent{Any}(value = first(Δ)))
    function _map_Fill_rrule(Δ::Tangent)
        Δf, Δx1_el, Δx2_el = back(Δ.value)
        return NoTangent(), Δf, Tangent{F1}(value = Δx1_el, axes = NoTangent()), Tangent{F2}(value = Δx2_el, axes = NoTangent())
    end
    return Fill(y_el, size(x1)), _map_Fill_rrule
end

function ChainRulesCore.rrule(::typeof(Base.getindex), x::F, n::Int) where {F<:Fill}
    function getindex_FillArray_rrule(Δ)
        return NoTangent(), Tangent{F}(value = Δ, axes = NoTangent()), NoTangent()
    end
    return x[n], getindex_FillArray_rrule
end

function ChainRulesCore.rrule(::typeof(Base.getindex), x::SVector{1,1}, n::Int)
    getindex_SArray_rrule(Δ) = NoTangent(), SVector{1}(Δ), ZeroTangent()
    return x[n], getindex_SArray_rrule
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
        return NoTangent(), UpperTriangular(Σ̄)
    end
    return C, cholesky_pullback
end

function cholesky_rrule(S::Symmetric{<:Real, <:StaticMatrix{N, N}}) where {N}
    C = cholesky(S)
    function cholesky_pullback(Δ::Tangent)
        U, Ū = C.U, Δ.factors
        Σ̄ = SMatrix{N,N}(Symmetric(Ū * U'))
        Σ̄ = U \ (U \ Σ̄)'
        Σ̄ = Σ̄ - Diagonal(Σ̄) / 2
        return NoTangent(), Tangent{typeof(S)}(data=SMatrix{N, N}(UpperTriangular(Σ̄)))
    end
    return C, cholesky_pullback
end

function ChainRulesCore.rrule(::typeof(cholesky), S::Symmetric{<:Real, <:StaticMatrix{N, N}}) where {N}
    return cholesky_rrule(S)
end

# Not used anywhere
# function logdet_pullback(C::Cholesky)
#     return logdet(C), function(Δ)
#         return ((uplo=nothing, info=nothing, factors=Diagonal(2 .* Δ ./ diag(C.factors))),)
#     end
# end

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
function Zygote._symmetric_back(Δ::SMatrix{N, N}, uplo) where {N}
    if uplo === 'U'
        return SMatrix{N, N}(UpperTriangular(Δ) + UpperTriangular(Δ') - Diagonal(Δ))
    else
        return SMatrix{N, N}(LowerTriangular(Δ) + LowerTriangular(Δ') - Diagonal(Δ))
    end
end


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

function Zygote.accum(a::Tangent{T}, b::NamedTuple) where {T}
    return Zygote.accum(a, Tangent{T}(; b...))
end

function Base.:(-)(
    A::UpperTriangular{<:Real, <:SMatrix{N, N}}, B::Diagonal{<:Real, <:SVector{N}},
) where {N}
    return UpperTriangular(A.data - B)   
end

function _symmetric_back(Δ, uplo)
    L, U, D = LowerTriangular(Δ), UpperTriangular(Δ), Diagonal(Δ)
    return collect(uplo == Symbol(:U) ? U .+ transpose(L) - D : L .+ transpose(U) - D)
end
_symmetric_back(Δ::Diagonal, uplo) = Δ
_symmetric_back(Δ::UpperTriangular, uplo) = collect(uplo == Symbol('U') ? Δ : transpose(Δ))
_symmetric_back(Δ::LowerTriangular, uplo) = collect(uplo == Symbol('U') ? transpose(Δ) : Δ)

function ChainRulesCore.rrule(::Type{Symmetric}, X::StridedMatrix{<:Real}, uplo=:U)
    function Symmetric_pullback(Δ)
        ΔX = Δ isa AbstractZero ? NoTangent() : _symmetric_back(Δ, uplo)
        return NoTangent(), ΔX, NoTangent()
    end
    return Symmetric(X, uplo), Symmetric_pullback
end

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
#     function StructArray_pullback(Δ::NamedTuple{(:components, )})
#         @show typeof(x), typeof(Δ.components)
#         return (nothing, Δ.components)
#     end
#     return T(x), StructArray_pullback
# end

function ChainRulesCore.rrule(T::Type{<:StructArray}, x::Union{Tuple,NamedTuple})
    function StructArray_pullback(Δ::Tangent)
        return NoTangent(), values(backing(Δ.components))
    end
    return T(x), StructArray_pullback
end

# `getproperty` accesses the `components` field of a `StructArray`. This rule makes that
# explicit. 
function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(Zygote.literal_getproperty), x::StructArray, ::Val{p},
) where {p}
    value, pb = rrule_via_ad(config, Zygote.literal_getproperty, getfield(x, :components), Val(p))
    function literal_getproperty_pullback(Δ)
        return NoTangent(), Tangent{typeof(x)}(components=pb(Δ)[2]), NoTangent()
    end
    return value, literal_getproperty_pullback
end

function time_ad(label::String, f, x...)
    println("primal: ", label)
    return @time f(x...)
end

time_ad(::Val{:disabled}, label::String, f, x...) = f(x...)

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(time_ad), label::String, f, x...)
    println("Forward: ", label)
    out, pb = @time rrule_via_ad(config, f, x...)
    function time_ad_pullback(Δ)
        println("Pullback: ", label)
        Δinputs = @time pb(Δ)
        return (NoTangent(), NoTangent(), NoTangent(), Δinputs...)
    end
    return out, time_ad_pullback
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(\), A::Diagonal{<:Real}, x::Vector{<:Real})
    out, pb = rrule_via_ad(config, (a, x) -> a .\ x, diag(A), x)
    function ldiv_pullback(Δ)
        if Δ isa AbstractZero
            return NoTangent()
        else
            _, Δa, Δx = pb(Δ)
            return NoTangent(), Diagonal(Δa), Δx
        end
    end
    return out, ldiv_pullback
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(\), A::Diagonal{<:Real}, x::Matrix{<:Real})
    out, pb = rrule_via_ad(config, (a, x) -> a .\ x, diag(A), x)
    function ldiv_pullback(Δ)
        if Δ isa AbstractZero
            return NoTangent()
        else
            _, Δa, Δx = pb(Δ)
            return NoTangent(), Diagonal(Δa), Δx
        end
    end
    return out, ldiv_pullback
end

using Base.Broadcast: broadcasted

function ChainRulesCore.rrule(::typeof(broadcasted), ::typeof(\), a::Vector{<:Real}, x::Vector{<:Real})
    y = a .\ x
    broadcast_ldiv_pullback(::AbstractZero) = NoTangent(), NoTangent(), NoTangent()
    broadcast_ldiv_pullback(Δ::AbstractVector{<:Real}) = NoTangent(), NoTangent(), -(Δ .* y ./ a), a .\ Δ
    return y, broadcast_ldiv_pullback
end

function ChainRulesCore.rrule(::typeof(broadcasted), ::typeof(\), a::Vector{<:Real}, x::Matrix{<:Real})
    y = a .\ x
    broadcast_ldiv_pullback(::AbstractZero) = NoTangent(), NoTangent(), NoTangent()
    broadcast_ldiv_pullback(Δ::AbstractMatrix{<:Real}) = NoTangent(), NoTangent(), -vec(sum(Δ .* y ./ a; dims=2)), a .\ Δ
    return y, broadcast_ldiv_pullback
end