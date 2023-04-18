# This is all AD-related stuff. If you're looking to understand TemporalGPs, this can be
# safely ignored.

using Zygote: accum, AContext
import ChainRulesCore: ProjectTo, rrule, _eltype_projectto

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

# ---------------------------------------------------------------------------- #
#                                 StaticArrays                                 #
# ---------------------------------------------------------------------------- #

function ProjectTo(x::SArray{S,T}) where {S, T}
    return ProjectTo{SArray}(; element=_eltype_projectto(T), axes=axes(x), static_size=S)
end

(proj::ProjectTo{SArray})(dx::SArray) = SArray{proj.static_size}(dx.data)
(proj::ProjectTo{SArray})(dx::AbstractArray) = SArray{proj.static_size}(Tuple(dx))

function rrule(::Type{T}, x::Tuple) where {T<:SArray}
    SArray_rrule(Δ) = begin
        (NoTangent(), Tangent{typeof(x)}(unthunk(Δ).data...))
    end
    return T(x), SArray_rrule
end

function rrule(::RuleConfig{>:HasReverseMode}, ::Type{SArray{S, T, N, L}}, x::NTuple{L, T}) where {S, T, N, L}
    SArray_rrule(::AbstractZero) = NoTangent(), NoTangent()
    SArray_rrule(Δ::NamedTuple{(:data,)}) = NoTangent(), Δ.data
    SArray_rrule(Δ::StaticArray{S}) = NoTangent(), Δ.data
    return SArray{S, T, N, L}(x), SArray_rrule
end

function rrule(
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
        _, ΔT, Δx = convert_pb(Tuple(Δnew_x))
        return ΔT, Δx
    end
    return SArray{S, T, N, L}(x), SArray_rrule
end

function rrule(::typeof(collect), x::X) where {S, T, N, L, X<:SArray{S, T, N, L}}
    y = collect(x)
    proj = ProjectTo(y)
    collect_rrule(Δ) = NoTangent(),  proj(Δ)
    return y, collect_rrule
end

function rrule(::typeof(vcat), A::SVector{DA}, B::SVector{DB}) where {DA, DB}
    function vcat_rrule(Δ)  # SVector
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
function rrule(::typeof(time_exp), A, t::Real)
    B = exp(A * t)
    time_exp_rrule(Ω̄) = NoTangent(), NoTangent(), sum(Ω̄ .*  (A * B))
    return B, time_exp_rrule
end


# Following is taken from https://github.com/JuliaArrays/FillArrays.jl/pull/153
# Until a solution has been found this code will be needed here.
"""
    ProjectTo(::Fill) -> ProjectTo{Fill}
    ProjectTo(::Ones) -> ProjectTo{NoTangent}

Most FillArrays arrays store one number, and so their gradients under automatic
differentiation represent the variation of this one number. 

The exception is those like `Ones` and `Zeros` whose type fixes their value,
which have no graidient.
"""
ProjectTo(x::Fill) = ProjectTo{Fill}(; element = ProjectTo(FillArrays.getindex_value(x)), axes = axes(x))

ProjectTo(::AbstractFill{Bool}) = ProjectTo{NoTangent}()  # Bool is always regarded as categorical

ProjectTo(::Zeros) = ProjectTo{NoTangent}()
ProjectTo(::Ones) = ProjectTo{NoTangent}()

(project::ProjectTo{Fill})(x::Fill) = x
function (project::ProjectTo{Fill})(dx::AbstractArray)
    for d in 1:max(ndims(dx), length(project.axes))
        size(dx, d) == length(get(project.axes, d, 1)) || throw(_projection_mismatch(axes_x, size(dx)))
    end
    Fill(sum(dx), project.axes)
end

function (project::ProjectTo{Fill})(dx::Tangent{<:Fill})
    # This would need a definition for length(::NoTangent) to be safe:
    # for d in 1:max(length(dx.axes), length(project.axes))
        # length(get(dx.axes, d, 1)) == length(get(project.axes, d, 1)) || throw(_projection_mismatch(dx.axes, size(dx)))
    # end
    Fill(dx.value / prod(length, project.axes), project.axes)
end
function (project::ProjectTo{Fill})(dx::Tangent{Any,<:NamedTuple{(:value, :axes)}})
    Fill(dx.value / prod(length, project.axes), project.axes)
end

# Yet another thing that should not happen
function Zygote.accum(x::Fill, y::NamedTuple{(:value, :axes)})
    Fill(x.value + y.value, x.axes)
end

# We have an alternative map to avoid Zygote untouchable specialisation on map.
_map(f, args...) = map(f, args...) 

function rrule(::Type{<:Fill}, x, sz)
    Fill_rrule(Δ::Union{Fill,Thunk}) = NoTangent(), FillArrays.getindex_value(unthunk(Δ)), NoTangent()
    Fill_rrule(Δ::Tangent{T,<:NamedTuple{(:value, :axes)}}) where {T} = NoTangent(), Δ.value, NoTangent()
    Fill_rrule(::AbstractZero) = NoTangent(), NoTangent(), NoTangent()
    Fill_rrule(Δ::Tangent{T,<:NTuple}) where {T} = NoTangent(), sum(Δ), NoTangent()
    function Fill_rrule(Δ::AbstractArray)
        # all(==(first(Δ)), Δ) || error("Δ should be a vector of the same value")
        # sum(Δ)
        # TODO Fix this rule, or what seems to be a downstream bug.
        return NoTangent(), sum(Δ), NoTangent()
    end
    Fill(x, sz), Fill_rrule 
end

function rrule(::typeof(Base.collect), x::Fill)
    y = collect(x)
    proj = ProjectTo(x)
    function collect_Fill_rrule(Δ)
        NoTangent(), proj(Δ)
    end
    return y, collect_Fill_rrule
end


function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(_map), f, x::Fill)
    y_el, back = ChainRulesCore.rrule_via_ad(config, f, x.value)
    function _map_Fill_rrule(Δ::AbstractArray)
        all(==(first(Δ)), Δ) || error("Δ should be a vector of the same value")
        Δf, Δx_el = back(first(Δ))
        NoTangent(), Δf, Fill(Δx_el, axes(x)) 
    end
    function _map_Fill_rrule(Δ::Union{Thunk,Fill,Tangent})
        Δf, Δx_el = back(unthunk(Δ).value)
        return NoTangent(), Δf, Fill(Δx_el, axes(x))
    end
    _map_Fill_rrule(::AbstractZero) = NoTangent(), NoTangent(), NoTangent()
    return Fill(y_el, axes(x)), _map_Fill_rrule
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(_map), f, x::Fill, y::Fill)
    z_el, back = ChainRulesCore.rrule_via_ad(config, f, x.value, y.value)
    function _map_Fill_rrule(Δ)
        Δf, Δx_el, Δy_el = back(unthunk(Δ).value)
        return NoTangent(), Δf, Fill(Δx_el, axes(x)), Fill(Δy_el, axes(x))
    end
    return Fill(z_el, axes(x)), _map_Fill_rrule
end

### Same thing for `StructArray`


function rrule(::typeof(step), x::T) where {T<:StepRangeLen}
    function step_StepRangeLen_rrule(Δ)
        return NoTangent(), Tangent{T}(step=Δ)
    end
    return step(x), step_StepRangeLen_rrule
end

function rrule(::typeof(Base.getindex), x::SVector{1,1}, n::Int)
    getindex_SArray_rrule(Δ) = NoTangent(), SVector{1}(Δ), NoTangent()
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

function rrule(::typeof(cholesky), S::Symmetric{<:Real, <:StaticMatrix{N, N}}) where {N}
    return cholesky_rrule(S)
end

function Zygote.accum(a::UpperTriangular, b::UpperTriangular)
    return UpperTriangular(Zygote.accum(a.data, b.data))
end

Zygote.accum(D::Diagonal{<:Real}, U::UpperTriangular{<:Real}) = UpperTriangular(D + U.data)
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
    function Symmetric_rrule(Δ)
        ΔX = Δ isa AbstractZero ? NoTangent() : _symmetric_back(Δ, uplo)
        return NoTangent(), ΔX, NoTangent()
    end
    return Symmetric(X, uplo), Symmetric_rrule
end

function rrule(::Type{StructArray}, x::T) where {T<:Union{Tuple,NamedTuple}}
    y = StructArray(x)
    StructArray_rrule(Δ::Thunk) = StructArray_rrule(unthunk(Δ))
    function StructArray_rrule(Δ)
        return NoTangent(), Tangent{T}(StructArrays.components(backing.(Δ))...)
    end
    function StructArray_rrule(Δ::AbstractArray)
        return NoTangent(), Tangent{T}((getproperty.(Δ, p) for p in propertynames(y))...)
    end
    return y, StructArray_rrule
end
function rrule(::Type{StructArray{X}}, x::T) where {X,T<:Union{Tuple,NamedTuple}}
    y = StructArray{X}(x)
    function StructArray_rrule(Δ)
        return NoTangent(), Tangent{T}(StructArrays.components(backing.(Δ))...)
    end
    function StructArray_rrule(Δ::Tangent)
        return NoTangent(), Tangent{T}(Δ.components...)
    end
    return y, StructArray_rrule
end


# `getproperty` accesses the `components` field of a `StructArray`. This rule makes that
# explicit. 
# function rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(Base.getproperty), x::StructArray, ::Val{p},
# ) where {p}
#     value, pb = rrule_via_ad(config, Base.getproperty, StructArrays.components(x), Val(p))
#     function getproperty_rrule(Δ)
#         return NoTangent(), Tangent{typeof(x)}(components=pb(Δ)[2]), NoTangent()
#     end
#     return value, getproperty_rrule
# end

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
