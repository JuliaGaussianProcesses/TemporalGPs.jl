my_I(T, N) = Matrix{T}(I, N, N)

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
    As = map(Base.Fix1(kron, ident), As_t)
    as = map(Base.Fix2(repeat, Nr), as_t)
    Qs = map(Base.Fix1(kron, Kr + ident_eps(1e-12)), Qs_t)
    emission_proj = _build_st_proj(emission_proj_t, Nr, ident)
    x0 = Gaussian(repeat(x0_t.m, Nr), kron(Kr, x0_t.P))
    return As, as, Qs, emission_proj, x0
end

function _build_st_proj((Hs, hs)::Tuple{AbstractVector, AbstractVector}, Nr::Integer, ident)
    return (map(H -> kron(ident, H), Hs), map(h -> Fill(h, Nr), hs))
end

function build_prediction_obs_vars(
    pr_indices::AbstractVector{<:Integer},
    r_full::AbstractVector{<:AbstractVector},
    σ²s_pr::AbstractVector{<:Diagonal{T}},
) where {T<:Real}
    σ²s_pr_full = map(r -> Diagonal(zeros(T, length(r))), r_full)
    σ²s_pr_full[pr_indices] .= σ²s_pr
    return σ²s_pr_full
end
