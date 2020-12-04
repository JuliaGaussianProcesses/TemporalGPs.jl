struct CheckpointedLGSSM{Tmodel<:LGSSM} <: AbstractSSM
    model::Tmodel
end


#
# Implement the AbstractSSM interface.
#

Base.:(==)(x::CheckpointedLGSSM, y::CheckpointedLGSSM) = x.model == y.model

Base.length(ft::CheckpointedLGSSM) = length(ft.model)

dim_obs(ft::CheckpointedLGSSM) = dim_obs(ft.model)

dim_latent(ft::CheckpointedLGSSM) = dim_latent(ft.model)

Base.eltype(ft::CheckpointedLGSSM) = eltype(ft.model)

storage_type(ft::CheckpointedLGSSM) = storage_type(ft.model)

is_of_storage_type(ft::CheckpointedLGSSM, s::StorageType) = is_of_storage_type(ft.model, s)

is_time_invariant(model::CheckpointedLGSSM) = is_time_invariant(model.model)

Base.getindex(model::CheckpointedLGSSM, n::Int) = getindex(model.model, n)

mean(model::CheckpointedLGSSM) = mean(model.model)

cov(model::CheckpointedLGSSM) = cov(model.model)

function decorrelate(model::CheckpointedLGSSM, ys::AV{<:AV{<:Real}}, f)
    return decorrelate(model.model, ys, f)
end

function correlate(model::CheckpointedLGSSM, ys::AV{<:AV{<:Real}}, f)
    return correlate(model.model, ys, f)
end

#
# Checkpointed pullbacks.
#

for (foo, step_foo, foo_pullback) in [
    (:correlate, :step_correlate, :correlate_pullback),
    (:decorrelate, :step_decorrelate, :decorrelate_pullback),
]
    @eval @adjoint function $foo(model::CheckpointedLGSSM, ys::AV{<:AV{<:Real}}, f)
        return $foo_pullback(model, ys, f)
    end

    # Standard rrule a la ZygoteRules.
    @eval function $foo_pullback(
        model_checkpointed::CheckpointedLGSSM, ys::AV{<:AV{<:Real}}, f,
    )
        model = model_checkpointed.model
        @assert length(model) == length(ys)
        T = length(model)

        # Determine the number of checkpoints `B` to use. `B` is also the number of time
        # steps covered between each checkpoint, except the last which may contain less.
        B = ceil(Int, sqrt(T))

        # Pre-allocate for filtering distributions. The indexing is slightly different for
        # these than for other quantities. In particular, xs[t] := x_{t-1}.
        x0 = model.gmm.x0
        xs = Vector{typeof(x0)}(undef, B + 2)
        xs[1] = x0 # the filtering distribution at t = 0

        # Intermediate storage for filtering distributions during the block.
        xs_block = Vector{eltype(xs)}(undef, B + 1)
        xs_block[1] = x0

        # Simulate running the first iteration to determine type of vs.
        v_dummy = f(ys[1], xs[1])
        vs = Vector{typeof(v_dummy)}(undef, T)

        # Run first block to obtain type
        lml = 0.0
        for b in 1:B
            for c in 1:min(B, T - (b - 1) * B)
                t = (b - 1) * B + c

                # This is a hack to store the penultimate filtering distriubution.
                if t == T
                    xs[end] = xs_block[c]
                end

                lml_, α, x_ = $step_foo(model[t], xs_block[c], ys[t])
                xs_block[c + 1] = x_
                lml += lml_
                vs[t] = f(α, x_)
            end

            # Capture state at end of block.
            xs[b + 1] = xs_block[end]
            xs_block[1] = xs_block[end]
        end

        function foo_pullback(Δ::Tuple{Any, Union{Nothing, AbstractVector}})

            Δlml = Δ[1]
            Δvs = Δ[2] isa Nothing ? Fill(nothing, T) : Δ[2]

            # Compute the pullback through the last element of the chain to get
            # initialisations for cotangents to accumulate.
            # Grabs the penultimate filtering distribution, xs[end].
            Δys = Vector{eltype(ys)}(undef, T)
            (Δα, Δx__) = get_pb(f)(last(Δvs))
            _, pullback_last = _pullback(NoContext(), $step_foo, model[T], xs[end], ys[T])
            _, Δmodel_at_T, Δx, Δy = pullback_last((Δlml, Δα, Δx__))
            Δmodel = get_adjoint_storage(model, Δmodel_at_T)
            Δys[T] = Δy

            for b in reverse(1:B)
                xs_block[1] = xs[b]
                block_length = min(B, T - (b - 1) * B)

                # Redo forwards pass, storing all intermediate quantities.
                for c in 1:block_length
                    t = (b - 1) * B + c

                    _, _, x = $step_foo(model[t], xs_block[c], ys[t])
                    xs_block[c + 1] = x
                end

                # Perform reverse-pass through the current block.
                for c in reverse(1:block_length)
                    t = (b - 1) * B + c
                    if t != T
                        Δα, Δx__ = get_pb(f)(Δvs[t])
                        Δx_ = Zygote.accum(Δx, Δx__)
                        _, pullback_t = _pullback(
                            NoContext(), $step_foo, model[t], xs_block[c], ys[t],
                        )
                        _, Δmodel_at_t, Δx, Δy = pullback_t((Δlml, Δα, Δx_))
                        Δmodel = _accum_at(Δmodel, t, Δmodel_at_t)
                        Δys[t] = Δy
                    end
                end
            end

            # Merge all gradient info associated with the model into the same place.
            Δmodel_ = (
                gmm = merge(Δmodel.gmm, (x0=Δx,)),
                Σ = Δmodel.Σ,
            )

            return (model = Δmodel_, ), Δys, nothing
        end

        return (lml, vs), foo_pullback
    end
end

checkpointed(model::LGSSM) = CheckpointedLGSSM(model)

# Adapt interface to work with checkpointed LGSSM.
rand_αs(rng::AbstractRNG, model::CheckpointedLGSSM, D) = rand_αs(rng, model.model, D)

function to_observed_diag(H, h, x)
    f = to_observed(H, h, x)
    return (m=f.m, s=diag(f.P))
end

"""
    smooth(model_checkpointed::CheckpointedLGSSM, ys::AbstractVector)

Filter, smooth, and compute the log marginal likelihood of the data. Returns the marginals
of the filtering and smoothing distributions. Employs a single step of binomial
checkpointing to ammeliorate memory usage issues.
"""
function smooth(model_checkpointed::CheckpointedLGSSM, ys::AbstractVector)

    model = model_checkpointed.model
    @assert length(model) == length(ys)
    T = length(model)
    Hs = model.gmm.H
    hs = model.gmm.h

    # Determine the number of checkpoints `B` to use. `B` is also the number of time
    # steps covered between each checkpoint, except the last which may contain less.
    B = ceil(Int, sqrt(T))

    # Pre-allocate for filtering distributions. The indexing is slightly different for
    # these than for other quantities. In particular, xs[t] := x_{t-1}.
    xs = Vector{typeof(model.gmm.x0)}(undef, B + 2)
    xs[1] = model.gmm.x0 # the filtering distribution at t = 0

    # Simulate running the first iteration to determine type of xs_filter.
    xs_filter_dummy = to_observed_diag(Hs[1], hs[1], xs[1])
    xs_filter = Vector{typeof(xs_filter_dummy)}(undef, T)

    # Run first block to obtain type
    lml = 0.0
    x = xs[1]
    for b in 1:B
        for c in 1:min(B, T - (b - 1) * B)
            t = (b - 1) * B + c

            # This is a hack to store the penultimate filtering distriubution.
            if t == T
                xs[end] = x
            end

            lml_, α, x = step_decorrelate(model[t], x, ys[t])
            lml += lml_
            xs_filter[t] = to_observed_diag(Hs[t], hs[t], x)
        end

        # Capture state at end of block.
        xs[b + 1] = x
    end

    # Perform the reverse-pass.
    ε = convert(eltype(model), 1e-12)

    x_smooth = xs[end - 1]

    xs_block = Vector{eltype(xs)}(undef, B + 1)

    xs_smooth = Vector{eltype(xs_filter)}(undef, T)
    xs_smooth[end] = to_observed_diag(Hs[end], hs[end], x_smooth)

    for b in reverse(1:B)
        xs_block[1] = xs[b]
        block_length = min(B, T - (b - 1) * B)

        # Redo forwards pass, storing all intermediate quantities.
        for c in 1:block_length
            t = (b - 1) * B + c

            _, _, x = step_decorrelate(model[t], xs_block[c], ys[t])
            xs_block[c + 1] = x
        end

        # Perform reverse-pass through the current block.
        for c in reverse(1:block_length)
            t = (b - 1) * B + c

            if t != T
                xf = xs_block[c + 1]
                xp = predict(model[t + 1], xf)

                U = cholesky(Symmetric(xp.P + ε * I)).U
                Gt = U \ (U' \ (model.gmm.A[t + 1] * xf.P))
                x_smooth = Gaussian(
                    _compute_ms(xf.m, Gt, x_smooth.m, xp.m),
                    _compute_Ps(xf.P, Gt, x_smooth.P, xp.P),
                )
                xs_smooth[t] = to_observed_diag(Hs[t], hs[t], x_smooth)
            end
        end

        # Transfer final smoothing distribution over to the next block.
        xs_block[end] = xs_block[1]
    end

    return xs_filter, xs_smooth, lml
end
