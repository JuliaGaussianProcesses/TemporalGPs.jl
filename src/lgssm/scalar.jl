using Stheno: MeanFunction, ConstMean, ZeroMean

"""
    ScalarLGSSM{Tmodel<:AbstractSSM} <: AbstractSSM

Linear Gaussian SSM whose outputs should be scalars. A lightweight wrapper around a regular
(vector-valued) LGSSM. Most of what this wrapper does is transform `AbstractVector`s of
`T <: Real`s into `AbstractVector`s of `SVector{1, T}`s, and then pass the data on to a
vector-valued ssm.
"""
struct ScalarLGSSM{Tmodel<:AbstractSSM} <: AbstractSSM
    model::Tmodel
end

hybrid_ihlgssm(model::ScalarLGSSM) = ScalarLGSSM(hybrid_ihlgssm(model.model))

Base.length(model::ScalarLGSSM) = length(model.model)
dim_obs(model::ScalarLGSSM) = 1
dim_latent(model::ScalarLGSSM) = dim_latent(model.model)

pick_first_scal(a::SVector{1, <:Real}, b) = first(a)
function get_pb(::typeof(pick_first_scal))
    pullback_pick_first_scal(Δ) = (SVector(Δ), nothing)
    pullback_pick_first_scal(::Nothing) = (nothing, nothing)
    return pullback_pick_first_scal
end

function correlate(model::ScalarLGSSM, αs::AbstractVector{<:Real}, f=pick_first_scal)
    αs_vec = reinterpret(SVector{1, eltype(αs)}, αs)
    lml, ys = correlate(model.model, αs_vec, f)
    return lml, ys
end

function decorrelate(model::ScalarLGSSM, ys::AbstractVector{<:Real}, f=pick_first_scal)
    ys_vec = reinterpret(SVector{1, eltype(ys)}, ys)
    lml, αs = decorrelate(model.model, ys_vec, f)
    return lml, αs
end

function whiten(model::ScalarLGSSM, ys::AbstractVector{<:Real})
    return last(decorrelate(model, ys))
end

function rand(rng::AbstractRNG, model::ScalarLGSSM)
    αs = randn(rng, length(model))
    return last(correlate(model, αs))
end

function unwhiten(model::ScalarLGSSM, αs::AbstractVector{<:Real})
    return last(correlate(model, αs))
end

function logpdf_and_rand(rng::AbstractRNG, model::ScalarLGSSM)
    αs = randn(rng, length(model))
    return correlate(model, αs)
end

function smooth(model::ScalarLGSSM, ys::AbstractVector{T}) where {T<:Real}
    return smooth(model.model, reinterpret(SVector{1, T}, ys))
end

function posterior_rand(rng::AbstractRNG, model::ScalarLGSSM, y::Vector{<:Real})
    fs = posterior_rand(rng, model.model, [SVector{1}(yn) for yn in y], 1)
    return first.(fs)
end

"""
    ssm(
        f::FiniteGP{<:GP, <:AbstractVector{<:Real}, <:Diagonal{<:Real, <:Fill}},
        storage::StorageType=DenseStorage(),
    )

Convert a `FiniteGP` into an `LGSSM`. 
"""
function ssm(
    f::FiniteGP{<:GP, <:AV{<:Real}, <:Diagonal{<:Real}},
    storage::StorageType=DenseStorage(),
)
    sde = to_sde(f.f, storage)
    ts = f.x
    ms = build_ms(get_ms(f.f.m, ts))
    Σs = build_Σs(f.Σy.diag)
    return ScalarLGSSM(ssm(sde, ts, ms, Σs))
end

#
# This is all a bit ugly, and would ideally go. IIRC there's some issue with the
# interactions between `FillArrays` and `Zygote` here that is problematic.
#

build_Σs(σ²_ns::AbstractVector{<:Real}) = SMatrix{1, 1}.(σ²_ns)

@adjoint function build_Σs(σ²_ns::Vector{<:Real})
    function build_Σs_Vector_back(Δ)
        return (first.(Δ),)
    end
    return build_Σs(σ²_ns), build_Σs_Vector_back
end

@adjoint function build_Σs(σ²_ns::Fill{<:Real})
    function build_Σs_Fill_back(Δ::NamedTuple)
        return ((value=first(Δ.value),),)
    end
    return build_Σs(σ²_ns), build_Σs_Fill_back
end

get_ms(m::MeanFunction, t::AV) = Stheno.ew(m, t)
get_ms(m::ConstMean, t::AV) = Fill(m.c, length(t))
get_ms(m::ZeroMean, t::AV) = Zeros(length(t))

build_ms(ms::AbstractVector{<:Real}) = SVector{1}.(ms)

@adjoint function build_ms(ms::Vector{<:Real})
    function build_ms_Vector_back(Δ)
        return (first.(Δ),)
    end
    return build_ms(ms), build_ms_Vector_back
end

@adjoint function build_ms(ms::Fill{<:Real})
    function build_ms_Fill_back(Δ)
        return ((value=first(Δ.value),),)
    end
    return build_ms(ms), build_ms_Fill_back
end

@adjoint build_ms(ms::Zeros{<:Real}) = SVector{1}.(ms), Δ->(nothing,)
