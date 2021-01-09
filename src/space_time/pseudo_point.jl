using Stheno: Scaled, Stretched, Sum

"""
    DTCSeparable{Tz<:AbstractVector, Tk<:SeparableKernel} <: Kernel

Specifies a low-rank approximation to a kernel `k` through pseudo-inputs `z`. `z` are the
locations of the pseudo-inputs in _space_, since they are automatically replicated
(implicitly) at each point in time.
"""
struct DTCSeparable{Tz<:AbstractVector, Tk<:Separable} <: Kernel
    z::Tz
    k::Tk
end

"""
    dtcify(z::AbstractVector, x)

Recurse through `x` and replace any `Separable` kernels with a `DTCSeparable`. The resulting
kernel can be used to construct the approximate model utilised by the DTC, and is used to
compute the ELBO.
"""
dtcify(z::AbstractVector, k::Separable) = DTCSeparable(z, k)

dtcify(z::AbstractVector, k::Scaled) = Scaled(k.σ², dtcify(z, k.k), k.f)

dtcify(z::AbstractVector, k::Stretched) = Stretched(k.a, dtcify(z, k.k), k.f)

dtcify(z::AbstractVector, k::Sum) = Sum(dtcify(z, k.kl), dtcify(z, k.kr))

dtcify(z::AbstractVector, fx::FiniteLTISDE) = FiniteGP(dtcify(z, fx.f), fx.x, fx.Σy)

dtcify(z::AbstractVector, fx::LTISDE) = LTISDE(dtcify(z, fx.f), fx.storage)

dtcify(z::AbstractVector, f::GP) = GP(f.m, dtcify(z, f.k), GPC())

"""
    dtc(fx::FiniteLTISDE, y::AbstractVector{<:Real}, z_r::AbstractVector)

Compute the DTC (Deterministic Training Conditional) in state-space form [insert reference].

`fx` and `y` are the same as would be provided to `logpdf`, and `z_r` is a specification of
the spatial location of the pseudo-points at each point in time.

Note that this API is slightly different from Stheno.jl's API, in which `z_r` is replaced
by a `FiniteGP`.

WARNING: this API is unstable, and subject to change in future versions of TemporalGPs. It
was thrown together quickly in pursuit of a conference deadline, and has yet to receive the
attention it deserves.
"""
function Stheno.dtc(fx::FiniteLTISDE, y::AbstractVector{<:Real}, z_r::AbstractVector)
    return logpdf(dtcify(z_r, fx), y)
end

"""
    elbo(fx::FiniteLTISDE, y::AbstractVector{<:Real}, z_r::AbstractVector)

Compute the ELBO (Evidence Lower BOund) in state-space form [insert reference].
"""
function Stheno.elbo(fx::FiniteLTISDE, y::AbstractVector{<:Real}, z_r::AbstractVector)
    fx_dtc = dtcify(z_r, fx)

    # Compute diagonals over prior marginals.
    lgssm = build_lgssm(fx_dtc)
    Σs = lgssm.emissions.Q
    marg_diag = marginals_diag(lgssm)

    k = fx_dtc.f.f.k
    Cf_diags = kernel_diagonals(k, fx_dtc.x)

    tmp = map(eachindex(marg_diag)) do n
        sum(diag(Σs[n] \ (Cf_diags[n] - marg_diag[n].P + Σs[n])))
    end
    return logpdf(lgssm, restructure(y, lgssm.emissions)) - sum(tmp) / 2
end

function kernel_diagonals(k::DTCSeparable, x::RectilinearGrid)
    space_kernel = k.k.l
    time_kernel = k.k.r
    Cr_rpred_diag = Stheno.elementwise(space_kernel, get_space(x))
    time_vars = Stheno.elementwise(time_kernel, get_time(x))
    return map(s_t -> Diagonal(Cr_rpred_diag * s_t), time_vars)
end

function kernel_diagonals(k::DTCSeparable, x::RegularInTime)
    space_kernel = k.k.l
    time_kernel = k.k.r
    time_vars = Stheno.elementwise(time_kernel, get_time(x))
    return map(zip(time_vars, x.vs)) do (s_t, x_r)
        Diagonal(Stheno.elementwise(space_kernel, x_r) * s_t)
    end
end

function kernel_diagonals(k::Scaled, x::AbstractVector)
    return k.σ²[1] .* kernel_diagonals(k.k, x)
end

function kernel_diagonals(k::Sum, x::AbstractVector)
    return kernel_diagonals(k.kl, x) .+ kernel_diagonals(k.kr, x)
end

function lgssm_components(k_dtc::DTCSeparable, x::SpaceTimeGrid, storage::StorageType)

    # Construct temporal model.
    k = k_dtc.k
    ts = get_time(x)
    time_kernel = k.r
    As_t, as_t, Qs_t, Hs_t, hs_t, x0_t = lgssm_components(time_kernel, ts, storage)

    # Compute spatial covariance between inducing inputs, and inducing points + obs. points.
    space_kernel = k.l
    x_space = x.xl
    z_space = k_dtc.z
    K_space_z = pw(space_kernel, z_space)
    K_space_zx = pw(space_kernel, z_space, x_space)

    # Get some size info.
    M = length(z_space)
    N = length(x_space)
    ident_M = my_I(eltype(storage), M)
    ident_N = my_I(eltype(storage), N)

    # G is the time-invariant component of the H-matrices. It is only time-invariant because
    # we have the same obsevation locations at each point in time.
    Λu_Cuf = cholesky(Symmetric(K_space_z + 1e-9I)) \ K_space_zx

    # Construct approximately low-rank model spatio-temporal LGSSM.
    As = map(A -> kron(ident_M, A), As_t)
    as = map(a -> repeat(a, M), as_t)
    Qs = map(Q -> kron(K_space_z, Q), Qs_t)
    Hs = map(H -> kron(Λu_Cuf, H), Hs_t) # This is currently O(N^2).
    hs = map(h -> fill(h, N), hs_t) # This should currently be zero.
    x0 = Gaussian(repeat(x0_t.m, M), kron(K_space_z, x0_t.P))
    return As, as, Qs, Hs, hs, x0
end

function lgssm_components(k_dtc::DTCSeparable, x::RegularInTime, storage::StorageType)

    # Construct temporal model.
    k = k_dtc.k
    ts = get_time(x)
    time_kernel = k.r
    As_t, as_t, Qs_t, Hs_t, hs_t, x0_t = lgssm_components(time_kernel, ts, storage)

    # Compute spatial covariance between inducing inputs, and inducing points + obs. points.
    space_kernel = k.l
    z_space = k_dtc.z
    K_space_z = pw(space_kernel, z_space)
    K_space_z_chol = cholesky(Symmetric(K_space_z + 1e-9I))

    # Get some size info.
    M = length(z_space)
    ident_M = my_I(eltype(storage), M)

    # Construct approximately low-rank model spatio-temporal LGSSM.
    As = map(A -> kron(ident_M, A), As_t)
    as = map(a -> repeat(a, M), as_t)
    Qs = map(Q -> kron(K_space_z, Q), Qs_t)
    Hs = map(
        (H, v) -> kron(K_space_z_chol \ pw(space_kernel, z_space, v), H),
        Hs_t, x.vs,
    )
    hs = map((h, v) -> fill(h, length(v)), hs_t, x.vs) # This should currently be zero.
    x0 = Gaussian(repeat(x0_t.m, M), kron(K_space_z, x0_t.P))
    return As, as, Qs, Hs, hs, x0
end

"""
    approx_posterior_marginals(
        ::typeof(dtc),
        fx::FiniteLTISDE,
        y::AbstractVector,
        z_r::AbstractVector,
        x_r::AbstractVector,
    )

Compute the DTC (Deterministic Training Conditional) approximation to the posterior
marginals at the times provided by `fx`, but at the new spatial locations `x_r`, given
observations `y` and spatial pseudo-input locations `z_r`.

WARNING: this API is unstable, and subject to change in future versions of TemporalGPs. It
was thrown together quickly in pursuit of a conference deadline, and has yet to receive the
attention it deserves.
"""
function approx_posterior_marginals(
    ::typeof(dtc),
    fx::FiniteLTISDE,
    y::AbstractVector,
    z_r::AbstractVector,
    x_r::AbstractVector,
)
    fx.f.f.m isa Stheno.ZeroMean || throw(error("Prior mean of GP isn't zero."))

    # Compute approximate posterior LGSSM.
    lgssm = build_lgssm(dtcify(z_r, fx))
    fx_post = posterior(lgssm, restructure(y, lgssm.emissions))

    # Compute the new emission distributions + approx posterior model.
    x_pr = RectilinearGrid(x_r, get_time(fx.x))
    k_dtc = dtcify(z_r, fx.f.f.k)
    Hs, hs, Σs = dtc_post_emissions(k_dtc, x_pr, fx.f.storage)
    new_fx_post = LGSSM(fx_post.transitions, build_emissions(Hs, hs, Σs))

    # Compute marginals under modified posterior.
    return vcat(map(marginals, marginals_diag(new_fx_post))...)
end

function dtc_post_emissions(k::DTCSeparable, x_new::AbstractVector, storage::StorageType)
    _, _, _, new_Hs, new_hs, _ = lgssm_components(k, x_new, storage)

    # Compute cross-covariance between target points and pseudo-points.
    # We only need to do this once over all times because the kernel is separable and we're
    # considering the same locations in space at each point in time.
    # This could be simplified further, but it's a hack that works fine for now.
    space_kernel = k.k.l
    z_r = k.z
    C_fp_u = Stheno.pairwise(space_kernel, get_space(x_new), z_r)
    C_u = cholesky(Symmetric(Stheno.pairwise(space_kernel, z_r)))
    Cr_rpred_diag = Stheno.elementwise(space_kernel, get_space(x_new))
    spatial_Q_diag = Cr_rpred_diag - Stheno.diag_Xt_invA_X(C_u, C_fp_u')

    time_kernel = k.k.r
    time_vars = Stheno.ew(time_kernel, get_time(x_new))
    new_Σs = map(s_t -> Diagonal(spatial_Q_diag * s_t), time_vars)

    return map(adjoint, new_Hs), new_hs, new_Σs
end

function dtc_post_emissions(k::Scaled, x_new::AbstractVector, storage::StorageType)
    Hs, hs, Σs = dtc_post_emissions(k.k, x_new, storage)
    σ = sqrt(convert(eltype(storage_type), only(k.σ²)))
    return map(H->σ * H, Hs), map(h->σ * h, hs), map(Σ->σ^2 * Σ, Σs)
end

function dtc_post_emissions(k::Sum, x_new::AbstractVector, storage::StorageType)
    Hs_l, hs_l, Σs_l = dtc_post_emissions(k.kl, x_new, storage)
    Hs_r, hs_r, Σs_r = dtc_post_emissions(k.kr, x_new, storage)
    return map(hcat, Hs_l, Hs_r), map(+, hs_l, hs_r), map(+, Σs_l, Σs_r)
end
