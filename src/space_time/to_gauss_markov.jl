using Kronecker: KroneckerProduct

my_I(T, N) = Matrix{T}(I, N, N)

function Zygote._pullback(::AContext, ::typeof(my_I), args...)
    my_I_pullback(Δ) = nothing
    return my_I(args...), my_I_pullback
end

Zygote.@nograd my_I

function lgssm_components(k::Separable, x::SpaceTimeGrid, storage)

    # Compute spatial covariance, and temporal GaussMarkovModel.
    r, t = x.xl, x.xr
    kr, kt = k.l, k.r
    Kr = pw(kr, r)
    As_t, as_t, Qs_t, Hs_t, hs_t, x0_t = lgssm_components(kt, t, storage)

    # Compute components of complete LGSSM.
    Nr = length(r)
    ident = my_I(eltype(storage), Nr)
    As = map(A -> kron(ident, A), As_t)
    as = map(a -> repeat(a, Nr), as_t)
    Qs = map(Q -> kron(Kr, Q), Qs_t)
    Hs = map(H -> kron(ident, H), Hs_t)
    hs = map(h -> fill(h, Nr), hs_t)
    x0 = Gaussian(repeat(x0_t.m, Nr), kron(Kr, x0_t.P))
    return As, as, Qs, Hs, hs, x0
end

function build_Σs(x::RectilinearGrid, Σ::Diagonal{<:Real})
    return Diagonal.(collect.(eachcol(reshape(Σ.diag, :, length(x.xr)))))
end

function Zygote._pullback(
    ::AContext, ::typeof(build_Σs), x::RectilinearGrid, Σ::Diagonal{<:Real},
)
    function build_Σs_pullback(Δ)
        return nothing, nothing, (diag=vcat(getfield.(Δ, :diag)...), )
    end
    return build_Σs(x, Σ), build_Σs_pullback
end

function build_Σs(x::RegularInTime, Σ::Diagonal{<:Real})
    vs = restructure(Σ.diag, length.(x.vs))
    return Diagonal.(collect.(vs))
end

function restructure(y::AbstractVector{<:Real}, emissions::StructArray)
    return restructure(y, map(dim_out, emissions))
end

function restructure(y::AbstractVector{<:Real}, lengths::AbstractVector{<:Integer})
    idxs_start = cumsum(vcat(0, lengths)) .+ 1
    idxs_end = idxs_start[1:end-1] .+ lengths .- 1
    return map(n -> y[idxs_start[n]:idxs_end[n]], eachindex(lengths))
end
