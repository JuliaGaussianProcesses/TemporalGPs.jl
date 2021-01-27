my_I(T, N) = Matrix{T}(I, N, N)

Zygote._pullback(::AContext, ::typeof(my_I), args...) = my_I(args...), nograd_pullback

function lgssm_components(k::Separable, x::SpaceTimeGrid, storage)

    # Compute spatial covariance, and temporal GaussMarkovModel.
    r, t = x.xl, x.xr
    kr, kt = k.l, k.r
    Kr = pw(kr, r)
    As_t, as_t, Qs_t, emission_proj_t, x0_t = lgssm_components(kt, t, storage)

    # Compute components of complete LGSSM.
    Nr = length(r)
    ident = my_I(eltype(storage), Nr)
    As = map(A -> kron(ident, A), As_t)
    as = map(a -> repeat(a, Nr), as_t)
    Qs = map(Q -> kron(Kr, Q) + UniformScaling(1e-12), Qs_t)
    emission_proj = _build_st_proj(emission_proj_t, Nr, ident)
    x0 = Gaussian(repeat(x0_t.m, Nr), kron(Kr, x0_t.P))
    return As, as, Qs, emission_proj, x0
end

function _build_st_proj((Hs, hs)::Tuple{AbstractVector, AbstractVector}, Nr::Integer, ident)
    return (map(H -> kron(ident, H), Hs), map(h -> fill(h, Nr), hs))
end

function build_Σs(x::RectilinearGrid, Σ::Diagonal{<:Real})
    return Diagonal.(collect.(eachcol(reshape(Σ.diag, :, length(x.xr)))))
end

function Zygote._pullback(
    ::AContext, ::typeof(build_Σs), x::RectilinearGrid, Σ::Diagonal{<:Real},
)
    build_Σs_pullback(Δ) = nothing, nothing, (diag=vcat(getfield.(Δ, :diag)...), )
    return build_Σs(x, Σ), build_Σs_pullback
end

function build_Σs(x::RegularInTime, Σ::Diagonal{<:Real})
    vs = restructure(Σ.diag, length.(x.vs))
    return Diagonal.(collect.(vs))
end

function restructure(y::AbstractVector{<:Real}, lengths::AbstractVector{<:Integer})
    idxs_start = cumsum(vcat(0, lengths)) .+ 1
    idxs_end = idxs_start[1:end-1] .+ lengths .- 1
    return map(n -> y[idxs_start[n]:idxs_end[n]], eachindex(lengths))
end

function restructure(y::AbstractVector{T}, lengths::AbstractVector{<:Integer}) where {T}
    idxs_start = cumsum(vcat(0, lengths)) .+ 1
    idxs_end = idxs_start[1:end-1] .+ lengths .- 1
    return map(eachindex(lengths)) do n
        y_missing = Vector{T}(undef, lengths[n])
        y_missing .= y[idxs_start[n]:idxs_end[n]]
        return y_missing
    end
end

function Zygote._pullback(
    ::AContext, ::typeof(restructure), y::Vector, lengths::AbstractVector{<:Integer},
)
    restructure_pullback(Δ::Vector) = nothing, vcat(Δ...), nothing
    return restructure(y, lengths), restructure_pullback
end

# Implement specific to Fills for AD's sake.
function restructure(y::Fill{<:Real}, lengths::AbstractVector{<:Integer})
    return map(l -> Fill(y.value, l), Zygote.dropgrad(lengths))
end

function restructure(y::AbstractVector, emissions::StructArray)
    return restructure(y, Zygote.dropgrad(map(dim_out, emissions)))
end

destructure(y::AbstractVector{<:AbstractVector{<:Real}}) = vcat(y...)
