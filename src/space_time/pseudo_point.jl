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

dtcify(z::AbstractVector, k::ScaledKernel) = ScaledKernel(dtcify(z, k.kernel), k.σ²)

function dtcify(z::AbstractVector, k::TransformedKernel{<:Kernel, <:ScaleTransform})
    return TransformedKernel(dtcify(z, k.kernel), k.transform)
end

function dtcify(z::AbstractVector, k::KernelSum)
    return KernelSum(dtcify(z, k.kernels[1]), dtcify(z, k.kernels[2]))
end

dtcify(z::AbstractVector, fx::FiniteLTISDE) = FiniteGP(dtcify(z, fx.f), fx.x, fx.Σy)

dtcify(z::AbstractVector, fx::LTISDE) = LTISDE(dtcify(z, fx.f), fx.storage)

dtcify(z::AbstractVector, f::GP) = GP(f.mean, dtcify(z, f.kernel))

"""
    dtc(fx::FiniteLTISDE, y::AbstractVector{<:Real}, z_r::AbstractVector)

Compute the DTC (Deterministic Training Conditional) in state-space form [insert reference].

`fx` and `y` are the same as would be provided to `logpdf`, and `z_r` is a specification of
the spatial location of the pseudo-points at each point in time.

Note that this API is slightly different from AbstractGPS.jl's API, in which `z_r` is
replaced by a `FiniteGP`.

WARNING: this API is unstable, and subject to change in future versions of TemporalGPs. It
was thrown together quickly in pursuit of a conference deadline, and has yet to receive the
attention it deserves.
"""
function dtc(fx::FiniteLTISDE, y::AbstractVector, z_r::AbstractVector)
    return logpdf(dtcify(z_r, fx), y)
end

# This stupid pullback saves an absurb amount of compute time.
function Zygote._pullback(::AContext, ::typeof(count), ::typeof(ismissing), yn)
    return count(ismissing, yn), nograd_pullback
end

"""
    elbo(fx::FiniteLTISDE, y::AbstractVector{<:Real}, z_r::AbstractVector)

Compute the ELBO (Evidence Lower BOund) in state-space form [insert reference].
"""
function AbstractGPs.elbo(fx::FiniteLTISDE, y::AbstractVector, z_r::AbstractVector)

    fx_dtc = time_ad(Val(:disabled), "fx_dtc", dtcify, z_r, fx)

    # Compute diagonals over prior marginals.
    lgssm = time_ad(Val(:disabled), "lgssm", build_lgssm, fx_dtc)
    Σs = lgssm.emissions.fan_out.Q
    marg_diags = time_ad(Val(:disabled), "marg_diags", marginals_diag, lgssm)

    k = fx_dtc.f.f.kernel
    Cf_diags = time_ad(Val(:disabled), "Cf_diags", kernel_diagonals, k, fx_dtc.x)

    # Transform a vector into a vector-of-vectors.
    y_vecs = time_ad(Val(:disabled), "y_vecs", restructure, y, lgssm.emissions)

    tmp = time_ad(Val(:disabled), "tmp", zygote_friendly_map,
        ((Σ, Cf_diag, marg_diag, yn), ) -> begin
            Σ_, _ = fill_in_missings(Σ, yn)
            return sum(diag(Σ_ \ (Cf_diag - marg_diag.P))) -
                count(ismissing, yn) + size(Σ_, 1)
        end,
        zip(Σs, Cf_diags, marg_diags, y_vecs),
    )

    return time_ad(Val(:disabled), "logpdf", logpdf, lgssm, y_vecs) - sum(tmp) / 2
end

Zygote.accum(x::NamedTuple{(:diag, )}, y::Diagonal) = Zygote.accum(x, (diag=y.diag, ))

function kernel_diagonals(k::DTCSeparable, x::RectilinearGrid)
    space_kernel = k.k.l
    time_kernel = k.k.r
    Cr_rpred_diag = kernelmatrix_diag(space_kernel, get_space(x))
    time_vars = kernelmatrix_diag(time_kernel, get_time(x))
    return map(s_t -> Diagonal(Cr_rpred_diag * s_t), time_vars)
end

function kernel_diagonals(k::DTCSeparable, x::RegularInTime)
    space_kernel = k.k.l
    time_kernel = k.k.r
    time_vars = kernelmatrix_diag(time_kernel, get_time(x))
    return map(
        (s_t, x_r) -> Diagonal(kernelmatrix_diag(space_kernel, x_r) * s_t),
        time_vars,
        x.vs,
    )
end

function kernel_diagonals(k::ScaledKernel, x::AbstractVector)
    return k.σ²[1] .* kernel_diagonals(k.kernel, x)
end

function kernel_diagonals(k::KernelSum, x::AbstractVector)
    return kernel_diagonals(k.kernels[1], x) .+ kernel_diagonals(k.kernels[2], x)
end

function lgssm_components(k_dtc::DTCSeparable, x::SpaceTimeGrid, storage::StorageType)

    # Construct temporal model.
    k = k_dtc.k
    ts = get_time(x)
    time_kernel = k.r
    As_t, as_t, Qs_t, emission_proj, x0_t = lgssm_components(time_kernel, ts, storage)
    Hs_t, hs_t = _extract_emission_proj(emission_proj)

    # Compute spatial covariance between inducing inputs, and inducing points + obs. points.
    space_kernel = k.l
    x_space = x.xl
    z_space = k_dtc.z
    K_space_z = kernelmatrix(space_kernel, z_space)
    K_space_zx = kernelmatrix(space_kernel, z_space, x_space)


    # Get some size info.
    M = length(z_space)
    N = length(x_space)
    ident_M = my_I(eltype(storage), M)

    # G is the time-invariant component of the H-matrices. It is only time-invariant because
    # we have the same obsevation locations at each point in time.
    Λu_Cuf = cholesky(Symmetric(K_space_z + 1e-12I)) \ K_space_zx

    # Construct approximately low-rank model spatio-temporal LGSSM.
    As = map(A -> kron(ident_M, A), As_t)
    as = map(a -> repeat(a, M), as_t)
    Qs = map(Q -> kron(K_space_z, Q), Qs_t)
    Cs = Fill(Λu_Cuf, length(ts))
    cs = map(h -> Fill(h, N), hs_t) # This should currently be zero.
    Hs = map(H -> kron(ident_M, H), Hs_t)
    hs = Fill(Zeros(M), length(ts))
    x0 = Gaussian(repeat(x0_t.m, M), kron(K_space_z, x0_t.P))
    return As, as, Qs, (Cs, cs, Hs, hs), x0
end

function lgssm_components(k_dtc::DTCSeparable, x::RegularInTime, storage::StorageType)

    # Construct temporal model.
    k = k_dtc.k
    ts = get_time(x)
    time_kernel = k.r
    As_t, as_t, Qs_t, emission_proj, x0_t = lgssm_components(time_kernel, ts, storage)
    Hs_t, hs_t = _extract_emission_proj(emission_proj)

    # Compute spatial covariance between inducing inputs, and inducing points + obs. points.
    space_kernel = k.l
    z_space = k_dtc.z
    K_space_z = kernelmatrix(space_kernel, z_space)
    K_space_z_chol = cholesky(Symmetric(K_space_z + 1e-9I))

    # Get some size info.
    M = length(z_space)
    N = length(ts)
    ident_M = my_I(eltype(storage), M)

    # Construct approximately low-rank model spatio-temporal LGSSM.
    As = zygote_friendly_map(
        ((I, A), ) -> kron(I, A),
        zip(Fill(ident_M, N), As_t),
    )
    as = zygote_friendly_map(a -> repeat(a, M), as_t)
    Qs = zygote_friendly_map(
        ((K_space_z, Q), ) -> kron(K_space_z, Q),
        zip(Fill(K_space_z, N), Qs_t),
    )
    x_big = time_ad(Val(:disabled), "x_big", _reduce, vcat, x.vs)
    C__ = time_ad(Val(:disabled), "C__", kernelmatrix, space_kernel, z_space, x_big)
    C = time_ad(Val(:disabled), "C", \, K_space_z_chol, C__)
    Cs = time_ad(Val(:disabled), "Cs", partition, Zygote.dropgrad(map(length, x.vs)), C)

    cs = map((h, v) -> fill(h, length(v)), hs_t, x.vs) # This should currently be zero.
    Hs = zygote_friendly_map(
        ((I, H_t), ) -> kron(I, H_t),
        zip(Fill(ident_M, N), Hs_t),
    )
    hs = Fill(Zeros(M), N)
    x0 = Gaussian(repeat(x0_t.m, M), kron(K_space_z, x0_t.P))

    return As, as, Qs, (Cs, cs, Hs, hs), x0
end

_extract_emission_proj((Hs, hs)::Tuple{AbstractVector, AbstractVector}) = Hs, hs

_reduce(::typeof(vcat), xs::Vector{<:Vector{<:Real}}) = reduce(vcat, xs)

function _reduce(::typeof(vcat), xs::Vector{<:ColVecs})
    return ColVecs(reduce(hcat, getfield.(xs, :X)))
end

function partition(lengths::AbstractVector{<:Integer}, A::Matrix{<:Real})
    starts = vcat(1, cumsum(lengths) .+ 1)
    starts = starts[1:end-1]
    return map((s, d) -> collect(view(A, :, s:s+d-1)), starts, lengths)
end

function Zygote._pullback(
    ctx::AContext,
    ::typeof(partition),
    lengths::AbstractVector{<:Integer},
    A::Matrix{<:Real},
)
    partition_pullback(::Nothing) = nothing
    partition_pullback(Δ::Vector) = nothing, nothing, reduce(hcat, Δ)
    return partition(lengths, A), partition_pullback
end

function build_emissions(
    (Cs, cs, Hs, hs)::Tuple{AbstractVector, AbstractVector, AbstractVector, AbstractVector},
    Σs::AbstractVector,
)
    Hst = map(adjoint, Hs)
    Cst = map(adjoint, Cs)
    fan_outs = StructArray{LargeOutputLGC{eltype(Cs), eltype(cs), eltype(Σs)}}((Cst, cs, Σs))
    return StructArray{BottleneckLGC{eltype(Hst), eltype(hs), eltype(fan_outs)}}((Hst, hs, fan_outs))
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
    fx.f.f.mean isa AbstractGPs.ZeroMean || throw(error("Prior mean of GP isn't zero."))

    # Compute approximate posterior LGSSM.
    lgssm = build_lgssm(dtcify(z_r, fx))
    fx_post = posterior(lgssm, restructure(y, lgssm.emissions))

    # Compute the new emission distributions + approx posterior model.
    x_pr = RectilinearGrid(x_r, get_time(fx.x))
    k_dtc = dtcify(z_r, fx.f.f.kernel)
    new_proj, Σs = dtc_post_emissions(k_dtc, x_pr, fx.f.storage)
    new_fx_post = LGSSM(fx_post.transitions, build_emissions(new_proj, Σs))

    # Compute marginals under modified posterior.
    return vcat(map(marginals, marginals_diag(new_fx_post))...)
end

"""
    approx_posterior_marginals(
        ::typeof(dtc),
        fx::FiniteLTISDE,
        y::AbstractVector,
        z_r::AbstractVector,
        x_r::AbstractVector,
        t::Int,
    )

Same as other method, but only returns the predictions at index `t` in `fx`.

As with the other method of this function, it's a bit of a hack. It's correct of course, but
needs to be tidied up at some point.
"""
function approx_posterior_marginals(
    ::typeof(dtc),
    fx::FiniteLTISDE,
    y::AbstractVector,
    z_r::AbstractVector,
    x_r::AbstractVector,
    t::Int,
)
    ts = get_time(fx.x)
    if t < 1 || t > length(ts)
        throw(error("t = $t must be between 1 and length(ts) = $(length(ts))."))
    end

    # Compute approximate posterior LGSSM.
    lgssm = build_lgssm(dtcify(z_r, fx))
    fx_post = posterior(lgssm, restructure(y, lgssm.emissions))

    # Prep prediction locations.
    x_other = x_r[1:1]
    x_rs = fill(x_other, length(ts))
    x_rs[t] = x_r
    x_pr = RegularInTime(ts, x_rs)

    # Compute the new emission distributions + approx posterior model.
    k_dtc = dtcify(z_r, fx.f.f.kernel)
    new_proj, Σs = dtc_post_emissions(k_dtc, x_pr, fx.f.storage)
    new_fx_post = LGSSM(fx_post.transitions, build_emissions(new_proj, Σs))

    # Compute marginals under modified posterior.
    return marginals(marginals_diag(new_fx_post)[t])
end

function approx_posterior_marginals(
    ::typeof(dtc),
    fx::FiniteLTISDE,
    y::AbstractVector,
    z_r::AbstractVector,
    x_pr::RegularInTime,
)
    ts = get_time(fx.x)
    if ts != get_time(x_pr)
        throw(error("Times don't match."))
    end

    # Compute approximate posterior LGSSM.
    lgssm = build_lgssm(dtcify(z_r, fx))
    fx_post = posterior(lgssm, restructure(y, lgssm.emissions))

    # Compute the new emission distributions + approx posterior model.
    k_dtc = dtcify(z_r, fx.f.f.kernel)
    new_proj, Σs = dtc_post_emissions(k_dtc, x_pr, fx.f.storage)
    new_fx_post = LGSSM(fx_post.transitions, build_emissions(new_proj, Σs))

    # Compute marginals under modified posterior.
    return vcat(map(marginals, marginals_diag(new_fx_post))...)
end

function build_emission_covs(k::DTCSeparable, x_new::RectilinearGrid)
    space_kernel = k.k.l
    z_r = k.z
    C_fp_u = kernelmatrix(space_kernel, get_space(x_new), z_r)
    C_u = cholesky(Symmetric(kernelmatrix(space_kernel, z_r) + ident_eps(z_r, 1e-9)))
    Cr_rpred_diag = kernelmatrix_diag(space_kernel, get_space(x_new))
    spatial_Q_diag = Cr_rpred_diag - diag_Xt_invA_X(C_u, C_fp_u')

    time_kernel = k.k.r
    time_vars = kernelmatrix_diag(time_kernel, get_time(x_new))
    return map(s_t -> Diagonal(spatial_Q_diag * s_t), time_vars)
end

function build_emission_covs(k::DTCSeparable, x_new::RegularInTime)
    space_kernel = k.k.l
    z_r = k.z
    C_u = cholesky(Symmetric(kernelmatrix(space_kernel, z_r) + ident_eps(z_r, 1e-9)))

    time_kernel = k.k.r
    time_vars = kernelmatrix_diag(time_kernel, get_time(x_new))
    return map(zip(time_vars, x_new.vs)) do ((time_var, x_r))
        C_fp_u = kernelmatrix(space_kernel, x_r, z_r)
        Cr_rpred_diag = kernelmatrix_diag(space_kernel, x_r)
        spatial_Q_diag = Cr_rpred_diag - diag_Xt_invA_X(C_u, C_fp_u')
        return Diagonal(spatial_Q_diag * time_var)
    end
end

function dtc_post_emissions(k::DTCSeparable, x_new::AbstractVector, storage::StorageType)
    _, _, _, new_proj, _ = lgssm_components(k, x_new, storage)
    new_Σs = build_emission_covs(k, x_new)
    return new_proj, new_Σs
end

function dtc_post_emissions(k::ScaledKernel, x_new::AbstractVector, storage::StorageType)
    (Cs, cs, Hs, hs), Σs = dtc_post_emissions(k.kernel, x_new, storage)
    σ = sqrt(convert(eltype(storage_type), only(k.σ²)))
    return (Cs, cs, map(H->σ * H, Hs), map(h->σ * h, hs)), map(Σ->σ^2 * Σ, Σs)
end

function dtc_post_emissions(k::KernelSum, x_new::AbstractVector, storage::StorageType)
    (Cs_l, cs_l, Hs_l, hs_l), Σs_l = dtc_post_emissions(k.kernels[1], x_new, storage)
    (Cs_r, cs_r, Hs_r, hs_r), Σs_r = dtc_post_emissions(k.kernels[2], x_new, storage)
    Cs = map(vcat, Cs_l, Cs_r)
    cs = cs_l + cs_r
    Hs = map(blk_diag, Hs_l, Hs_r)
    hs = map(vcat, hs_l, hs_r)
    return (Cs, cs, Hs, hs), map(+, Σs_l, Σs_r)
end
