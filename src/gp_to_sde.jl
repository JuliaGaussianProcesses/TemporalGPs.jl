"""
    LTISDE{TF, TL, TQ, Tm₀, TP₀}

```julia
dx = F x dt + L dβ
```
where `β` is a Brownian motion with diffusion matrix `Q`, and `x(t₀) ∼ N(m₀, P₀)`.
"""
struct LTISDE{TF, TL, TQ, TH, Tx₀, Tv}
    F::TF
    L::TL
    Q::TQ
    H::TH
    x₀::Tx₀
    v::Tv
end

dim_latent(sde::LTISDE) = size(H, 2)

dim_obs(sde::LTISDE) = size(H, 1)


const UniLTISDE = LTISDE{<:AM, <:AV, <:Real, <:AM, <:Gaussian, <:Real}

function Base.:(==)(a::LTISDE, b::LTISDE)
    return a.F == b.F && a.L == b.L && a.Q == b.Q &&
        a.H == b.H && a.x₀ == b.x₀ && a.v == b.v
end

function to_dense(model::LTISDE)
    return LTISDE(
        collect(model.F),
        collect(model.L),
        collect(model.Q),
        collect(model.H),
        Gaussian(collect(model.x₀.m), collect(model.x₀.P)),
        model.v,
    )
end

function to_dense(model::UniLTISDE)
    return LTISDE(
        collect(model.F),
        collect(model.L),
        model.Q,
        collect(model.H),
        Gaussian(collect(model.x₀.m), collect(model.x₀.P)),
        model.v,
    )
end



# Used to specify what type of internal storage to use for the parameters of an LTISDE.
abstract type StorageType end
struct DenseStorage <: StorageType end
struct StaticStorage <: StorageType end


#
# Convert GPs / kernels to SDEs
#

function to_sde(k::Stheno.ConstKernel, ::DenseStorage)
    σ² = k.c
    F = zeros(1, 1)
    L = ones(1, 1)
    q = 0.0
    H = ones(1, 1)
    m = zeros(1)
    P = reshape([σ²], 1, 1)
    return LTISDE(F, L, Q, m, P), H
end

# function to_sde(k::Stheno.Linear, σ²::Real, t0::Real)
#     F = [0.0 1.0; 0.0 0.0]
#     L = reshape([0.0, 1.0], 2, 1)
#     q = 0.0
#     H = reshape([1.0, 0.0], 1, 2)
#     m = zeros(2)
#     P = σ² .* [t0^2 t0; t0 1]
#     return LTISDE(F, L, Q, m, P), H
# end

function to_sde(k::Stheno.Exp, ::StaticStorage)
    F = SMatrix{1, 1}(-1.0)
    L = @SVector [1.0]
    q = 2.0
    H = SMatrix{1, 1}(1.0)
    m = @SVector [0.0]
    P = SMatrix{1, 1}(1.0)
    return LTISDE(F, L, q, H, Gaussian(m, P), 1.0)
end
to_sde(k::Stheno.Exp, ::DenseStorage) = to_dense(to_sde(k, StaticStorage()))

function to_sde(k::Stheno.Matern32, ::StaticStorage)
    λ = sqrt(3)
    F = @SMatrix [0.0 1.0; -3.0 -2λ]
    L = @SVector [0.0, 1.0]
    q = 4.0 * λ^3
    H = SMatrix{1, 2}(1.0, 0.0)
    m = @SVector [0.0, 0.0]
    P = @SMatrix [1.0 0.0; 0.0 3.0]
    return LTISDE(F, L, q, H, Gaussian(m, P), 1.0)
end
to_sde(k::Stheno.Matern32, ::DenseStorage) = to_dense(to_sde(k, StaticStorage()))

function to_sde(k::Stheno.Matern52, ::StaticStorage)
    λ = sqrt(5)
    F = @SMatrix [0.0  1.0   0.0;
                  0.0  0.0   1.0;
                  -λ^3 -3λ^2 -3λ]
    L = @SVector [0.0, 0.0, 1.0]
    q = 8 * λ^5 / 3
    H = SMatrix{1, 3}(1.0, 0.0, 0.0)
    m = @SVector [0.0, 0.0, 0.0]
    κ = 5.0 / 3.0
    P = @SMatrix [1.0  0.0 -κ;
                  0.0  κ   0.0;
                  -κ   0.0 25]
    return LTISDE(F, L, q, H, Gaussian(m, P), 1.0)
end
to_sde(k::Stheno.Matern52, ::DenseStorage) = to_dense(to_sde(k, StaticStorage()))

Zygote.@adjoint function to_sde(k::Stheno.Exp, storage::StorageType)
    return to_sde(k, storage), Δ->(nothing, nothing)
end
Zygote.@adjoint function to_sde(k::Stheno.Matern32, storage::StorageType)
    return to_sde(k, storage), Δ->(nothing, nothing)
end
Zygote.@adjoint function to_sde(k::Stheno.Matern52, storage::StorageType)
    return to_sde(k, storage), Δ->(nothing, nothing)
end

# Scaling a kernel means that we observe a scaled version of the process.
function to_sde(k::Stheno.Scaled, storage::StorageType)
    sde = to_sde(k.k, storage)
    return LTISDE(sde.F, sde.L, sde.Q, sqrt(k.σ²) .* sde.H, sde.x₀, sde.v)
end

# Stretching a kernel makes time run at a different rate.
function to_sde(k::Stheno.Stretched, storage::StorageType)
    sde = to_sde(k.k, storage)
    return LTISDE(sde.F, sde.L, sde.Q, sde.H, sde.x₀, sde.v * k.a)
end

function to_sde(k::Stheno.Sum, storage::StorageType)

    # Compute sdes for each process in the sum.
    sde_l = to_sde(k.l, storage)
    sde_r = to_sde(k.r, storage)

    # Concatenate left-hand process dynamics with right-hand process dynamics.
    F = blk_diag(sde_l.F, sde_r.F)
    L = blk_diag(sde_l.L, sde_r.L)
    Q = blk_diag(sde_l.Q, sde_r.Q)
    H = hcat(sde_l.H, sde_r.H)
    x₀ = Gaussian(
        vcat(sde_l.x₀.m, sde_r.x₀.m),
        blk_diag(sde_l.x₀.P, sde_r.x₀.P),
    )

    # Make new SDE.
    # IMPLEMENTATION OF v IS GOING TO BE SLIGHTLY TRICKY WITH THE CURRENT FRAMEWORK!
    return LTISDE(F, L, Q, H, x₀, v)
end

function blk_diag(A, B)
    return hvcat(
        (2, 2),
        A, zeros(size(A, 1), size(B, 2)),
        zeros(size(A, 2), size(B, 1)), B,
    )
end

to_sde(f::GP{<:Stheno.ZeroMean}, storage::StorageType) = to_sde(f.k, storage)

# By default, use dense storage.
to_sde(f::GP{<:Stheno.ZeroMean}) = to_sde(f, DenseStorage())
