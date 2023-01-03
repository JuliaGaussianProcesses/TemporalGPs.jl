my_I(T, N) = Matrix{T}(I, N, N)
ChainRulesCore.@non_differentiable my_I(args...)

function lgssm_components(k::Separable, x::SpaceTimeGrid, storage)

    # Compute spatial covariance, and temporal GaussMarkovModel.
    t = get_times(x)
    r = get_space(x)

    kr, kt = k.l, k.r
    Kr = kernelmatrix(kr, r)
    As_t, as_t, Qs_t, emission_proj_t, x0_t = lgssm_components(kt, t, storage)

    # Compute components of complete LGSSM.
    Nr = length(r)
    ident = my_I(eltype(storage), Nr)
    As = _map(Base.Fix1(kron, ident), As_t)
    as = _map(Base.Fix2(repeat, Nr), as_t)
    Qs = _map(Base.Fix1(kron, Kr + ident_eps(1e-12)), Qs_t)
    emission_proj = _build_st_proj(emission_proj_t, Nr, ident)
    x0 = Gaussian(repeat(x0_t.m, Nr), kron(Kr, x0_t.P))
    return As, as, Qs, emission_proj, x0
end

function _build_st_proj((Hs, hs)::Tuple{AbstractVector, AbstractVector}, Nr::Integer, ident)
    return (_map(H -> kron(ident, H), Hs), _map(h -> Fill(h, Nr), hs))
end

_map(f, args...) = map(f, args...)

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(_map), f::Tf, x::Fill) where {Tf}
    y_el, back = ChainRulesCore.rrule_via_ad(config, f, x.value)
    function map_Fill_pullback(Δ::Tangent)
        _, Δx_el = back(Δ.value)
        return NoTangent(), NoTangent(), (value = Δx_el, axes=nothing)
    end
    return Fill(y_el, size(x)), map_Fill_pullback
end

# function ChainRulesCore.rrule(::typeof(_build_st_proj), (Hs, hs)::Tuple{AbstractVector, AbstractVector}, Nr::Integer, ident::AbstractMatrix)
#     return _build_st_proj((Hs, hs), Nr, ident), Δ -> @show typeof.(Δ[1]), typeof.(Δ[2])
# end

function build_prediction_obs_vars(
    pr_indices::AbstractVector{<:Integer},
    r_full::AbstractVector{<:AbstractVector},
    σ²s_pr::AbstractVector{<:Diagonal{T}},
) where {T<:Real}
    σ²s_pr_full = map(r -> Diagonal(zeros(T, length(r))), r_full)
    σ²s_pr_full[pr_indices] .= σ²s_pr
    return σ²s_pr_full
end
