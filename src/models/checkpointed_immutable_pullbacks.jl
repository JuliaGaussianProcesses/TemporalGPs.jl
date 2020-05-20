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

function decorrelate(mut, model::CheckpointedLGSSM, ys::AV{<:AV{<:Real}}, f=copy_first)
    return decorrelate(mut, model.model, ys, f)
end

function correlate(mut, model::CheckpointedLGSSM, ys::AV{<:AV{<:Real}}, f=copy_first)
    return correlate(mut, model.model, ys, f)
end

#
# Checkpointed pullbacks.
#

for (foo, step_foo, foo_pullback, step_foo_pullback) in [
    (:correlate, :step_correlate, :correlate_pullback, :step_correlate_pullback),
    (:decorrelate, :step_decorrelate, :decorrelate_pullback, :step_decorrelate_pullback),
]
    @eval @adjoint function $foo(
        ::Immutable,
        model::CheckpointedLGSSM,
        ys::AV{<:AV{<:Real}},
        f=copy_first,
    )
        return $foo_pullback(Immutable(), model, ys, f)
    end

    # Standard rrule a la ZygoteRules.
    @eval function $foo_pullback(
        ::Immutable,
        model_checkpointed::CheckpointedLGSSM,
        ys::AV{<:AV{<:Real}},
        f,
    )
        model = model_checkpointed.model
        @assert length(model) == length(ys)
        T = length(model)

        # Determine the number of checkpoints `B` to use. `B` is also the number of time
        # steps covered between each checkpoint, except the last which may contain less.
        B = ceil(Int, sqrt(T))

        # Pre-allocate for filtering distributions. The indexing is slightly different for
        # these than for other quantities. In particular, xs[t] := x_{t-1}.
        xs = Vector{typeof(model.gmm.x0)}(undef, B + 2)
        xs[1] = model.gmm.x0 # the filtering distribution at t = 0

        # Simulate running the first iteration to determine type of vs.
        v_dummy = f(ys[1], xs[1])
        vs = Vector{typeof(v_dummy)}(undef, T)

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

                lml_, α, x = $step_foo(model[t], x, ys[t])
                lml += lml_
                vs[t] = f(α, x)
            end

            # Capture state at end of block.
            xs[b + 1] = x
        end

        function foo_pullback(Δ::Tuple{Any, Nothing})
            return foo_pullback((first(Δ), Fill(nothing, T)))
        end

        function foo_pullback(Δ::Tuple{Any, AbstractVector})

            Δlml = Δ[1]
            Δvs = Δ[2]

            # Intermediate storage for filtering distributions during the block.
            xs_block = Vector{eltype(xs)}(undef, B + 1)

            # Compute the pullback through the last element of the chain to get
            # initialisations for cotangents to accumulate.
            # Grabs the penultimate filtering distribution, xs[end].
            Δys = Vector{eltype(ys)}(undef, T)
            (Δα, Δx__) = get_pb(f)(last(Δvs))
            _, pullback_last = $step_foo_pullback(model[T], xs[end], ys[T])
            Δmodel_at_T, Δx, Δy = pullback_last((Δlml, Δα, Δx__))
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
                        _, pullback_t = $step_foo_pullback(model[t], xs_block[c], ys[t])
                        Δmodel_at_t, Δx, Δy = pullback_t((Δlml, Δα, Δx_))
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

            return nothing, (model = Δmodel_, ), Δys, nothing
        end

        return (lml, vs), foo_pullback
    end
end

checkpointed(model::LGSSM) = CheckpointedLGSSM(model)

# Adapt interface to work with checkpointed LGSSM.
rand_αs(rng::AbstractRNG, model::CheckpointedLGSSM, D) = rand_αs(rng, model.model, D)
