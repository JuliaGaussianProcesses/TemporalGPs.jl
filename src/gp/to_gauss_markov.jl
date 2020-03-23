using Stheno: MeanFunction, ConstMean, ZeroMean



#
# Generic constructors for base kernels. A kernel `k` is a base kernel if
# kernel_trait(k) isa BaseKernel.
#

function GaussMarkovModel(k::Kernel, t::AV{<:Real}, storage_type)

    # Compute stationary distribution and sde.
    x0 = stationary_distribution(k, storage_type)
    P = x0.P
    F, q, H = to_sde(k, storage_type)

    # Use stationary distribution + sde to compute finite-dimensional Gauss-Markov model.
    t = vcat([first(t) - 1], t)
    As = map(Δt -> time_exp(F, Δt), diff(t))
    as = Fill(Zeros(size(first(As), 1)), length(t))
    Qs = map(A -> P - A * P * A', As)
    Hs = Fill(H, length(As))
    hs = Fill(Zeros(size(H, 1)), length(As))

    return GaussMarkovModel(As, as, Qs, Hs, hs, x0)
end

function GaussMarkovModel(k::Kernel, t::StepRangeLen, storage_type)

    # Compute stationary distribution and sde.
    x0 = stationary_distribution(k, storage_type)
    P = x0.P
    F, q, H = to_sde(k, storage_type)

    # Use stationary distribution + sde to compute finite-dimensional Gauss-Markov model.
    A = time_exp(F, step(t))
    As = Fill(A, length(t))
    as = Fill(Zeros(size(F, 1)), length(t))
    Q = P - A * P * A'
    Qs = Fill(Q, length(t))
    Hs = Fill(H, length(t))
    hs = Fill(Zeros(size(H, 1)), length(As))

    return GaussMarkovModel(As, as, Qs, Hs, hs, x0)
end

# Fallback definitions for most base kernels.
to_sde(k::Kernel, ::DenseStorage) = map(collect, to_sde(k, StaticStorage()))

function stationary_distribution(k::Kernel, ::DenseStorage)
    x = stationary_distribution(k, StaticStorage())
    return Gaussian(collect(x.m), collect(x.P))
end



#
# Matern-1/2
#

function to_sde(k::Matern12, ::StaticStorage)
    F = SMatrix{1, 1}(-1.0)
    q = 2.0
    H = SMatrix{1, 1}(1.0)
    return F, q, H
end

function stationary_distribution(k::Matern12, ::StaticStorage)
    return Gaussian((@SVector [0.0]), (SMatrix{1, 1}(1.0)))
end

Zygote.@adjoint function to_sde(k::Matern12, storage_type)
    return to_sde(k, storage_type), Δ->(nothing, nothing)
end

Zygote.@adjoint function stationary_distribution(k::Matern12, storage_type)
    return stationary_distribution(k, storage_type), Δ->(nothing, nothing)
end



#
# Matern - 3/2
#

function to_sde(k::Matern32, ::StaticStorage)
    λ = sqrt(3)
    F = @SMatrix [0.0 1.0; -3.0 -2λ]
    q = 4.0 * λ^3
    H = SMatrix{1, 2}(1.0, 0.0)
    return F, q, H
end

function stationary_distribution(k::Matern32, ::StaticStorage)
    return Gaussian((@SVector [0.0, 0.0]), (@SMatrix [1.0 0.0; 0.0 3.0]))
end

Zygote.@adjoint function to_sde(k::Matern32, storage_type)
    return to_sde(k, storage_type), Δ->(nothing, nothing)
end

Zygote.@adjoint function stationary_distribution(k::Matern32, storage_type)
    return stationary_distribution(k, storage_type), Δ->(nothing, nothing)
end



#
# Matern - 5/2
#

function to_sde(k::Matern52, ::StaticStorage)
    λ = sqrt(5)
    F = @SMatrix [0.0  1.0   0.0;
                  0.0  0.0   1.0;
                  -λ^3 -3λ^2 -3λ]
    q = 8 * λ^5 / 3
    H = SMatrix{1, 3}(1.0, 0.0, 0.0)
    return F, q, H
end

function stationary_distribution(k::Matern52, ::StaticStorage)
    κ = 5.0 / 3.0
    m = @SVector [0.0, 0.0, 0.0]
    P = @SMatrix [1.0  0.0 -κ;
                  0.0  κ   0.0;
                  -κ   0.0 25]
    return Gaussian(m, P)
end

Zygote.@adjoint function to_sde(k::Matern52, storage_type)
    return to_sde(k, storage_type), Δ->(nothing, nothing)
end

Zygote.@adjoint function stationary_distribution(k::Matern52, storage_type)
    return stationary_distribution(k, storage_type), Δ->(nothing, nothing)
end



#
# Scaled
#

function GaussMarkovModel(k::Stheno.Scaled, ts::AV{<:Real}, storage_type)
    model = GaussMarkovModel(k.k, ts, storage_type)
    σ = sqrt(k.σ²)
    Hs = map(n->σ * model.H[n], 1:length(model.H))
    hs = map(n->σ * model.h[n], 1:length(model.h))
    return GaussMarkovModel(model.A, model.a, model.Q, Hs, hs, model.x0)
end



#
# Stretched
#

function GaussMarkovModel(k::Stheno.Stretched, ts::AV{<:Real}, storage_type)
    return GaussMarkovModel(k.k, apply_stretch(k.a, ts), storage_type)
end

apply_stretch(a, ts::AV{<:Real}) = a * ts

function apply_stretch(a, ts::StepRangeLen)
    return Zygote.@showgrad(range(a * first(ts), a * last(ts); length=length(ts)))
end



#
# Sum
#

function GaussMarkovModel(k::Stheno.Sum, ts::AV{<:Real}, storage_type)
    model_l = GaussMarkovModel(k.kl, ts, storage_type)
    model_r = GaussMarkovModel(k.kr, ts, storage_type)

    return GaussMarkovModel(
        blk_diag.(model_l.A, model_r.A),
        vcat.(model_l.a, model_r.a),
        blk_diag.(model_l.Q, model_r.Q),
        hcat.(model_l.H, model_r.H),
        vcat.(model_l.h, model_r.h),
        Gaussian(
            collect(vcat(model_l.x0.m, model_r.x0.m)),
            blk_diag(model_l.x0.P, model_r.x0.P),
        ),
    )
end

function blk_diag(A, B)
    return hvcat(
        (2, 2),
        A, zeros(size(A, 1), size(B, 2)), zeros(size(B, 1), size(A, 2)), B,
    )
end

Zygote.@adjoint function blk_diag(A, B)
    function blk_diag_adjoint(Δ)
        ΔA = Δ[1:size(A, 1), 1:size(A, 2)]
        ΔB = Δ[size(A, 1)+1:end, size(A, 2)+1:end]
        return (ΔA, ΔB)
    end
    return blk_diag(A, B), blk_diag_adjoint
end



#
# Product
#

function GaussMarkovModel(k::Stheno.Product, ts::AV{<:Real}, storage_type)
    error("Not implemented")
    model_l = GaussMarkovModel(k.kl, ts, storage_type)
    model_r = GaussMarkovModel(k.kr, ts, storage_type)
end
