#
# This file contains pullbacks for stuff in generic.jl. These are purely performance
# optimisations for algorithmic differentiation, and in no way important for understanding
# the structure of the package, or its functionality.
#

function Zygote.accum(a::UpperTriangular, b::UpperTriangular)
    return UpperTriangular(Zygote.accum(a.data, b.data))
end

function Zygote.accum(D::Diagonal{<:Real}, U::UpperTriangular{<:Real, <:SMatrix})
    return UpperTriangular(D + U.data)
end

#
# Objects in which to storage / accumulate the adjoint w.r.t. the hypers.
#

function get_adjoint_storage(x::Vector, init::T) where {T<:AbstractVecOrMat{<:Real}}
    Δx = Vector{T}(undef, length(x))
    Δx[end] = init
    return Δx
end

get_adjoint_storage(x::Fill, init) = (value=init,)

# This is a slightly weird adjoint. The fields don't directly correspond to fields of the
# LGSSM. This is because this object is always accessed via `getindex` in this
# functionality. To make life simple, this somewhat unintuitive hack was necessary.
function get_adjoint_storage(x::LGSSM, Δx::NamedTuple{(:A, :a, :Q, :H, :h, :Σ)})
    return (
        gmm = get_adjoint_storage(x.gmm, (A=Δx.A, a=Δx.a, Q=Δx.Q, H=Δx.H, h=Δx.h)),
        Σ = get_adjoint_storage(x.Σ, Δx.Σ),
    )
end

function get_adjoint_storage(x::GaussMarkovModel, Δx::NamedTuple{(:A, :a, :Q, :H, :h)})
    return (
        A = get_adjoint_storage(x.A, Δx.A),
        a = get_adjoint_storage(x.a, Δx.a),
        Q = get_adjoint_storage(x.Q, Δx.Q),
        H = get_adjoint_storage(x.H, Δx.H),
        h = get_adjoint_storage(x.h, Δx.h),
    )
end

function _accum_at(Δxs::Vector, n::Int, Δx)
    Δxs[n] = Δx
    return Δxs
end

_accum_at(Δxs::NamedTuple{(:value,)}, n::Int, Δx) = (value=Δxs.value + Δx,)

function _accum_at(Δxs::NamedTuple{(:A, :a, :Q, :H, :h)}, n::Int, Δx)
    return (
        A = _accum_at(Δxs.A, n, Δx.A),
        a = _accum_at(Δxs.a, n, Δx.a),
        Q = _accum_at(Δxs.Q, n, Δx.Q),
        H = _accum_at(Δxs.H, n, Δx.H),
        h = _accum_at(Δxs.h, n, Δx.h),
    )
end

function _accum_at(Δxs::NamedTuple{(:gmm, :Σ)}, n::Int, Δx)
    return(
        gmm = _accum_at(Δxs.gmm, n, (A=Δx.A, a=Δx.a, Q=Δx.Q, H=Δx.H, h=Δx.h)),
        Σ = _accum_at(Δxs.Σ, n, Δx.Σ),
    )
end

function get_pb(::typeof(copy_first))
    copy_first_pullback(Δ) = (copy(Δ), nothing)
    copy_first_pullback(Δ::Nothing) = (nothing, nothing)
    return copy_first_pullback
end

get_pb(::typeof(pick_last)) = Δ->(nothing, Δ)

for (foo, step_foo, step_foo_pullback) in [
    (:correlate, :step_correlate, :step_correlate_pullback),
    (:decorrelate, :step_decorrelate, :step_decorrelate_pullback),
]
    # Standard rrule a la ChainRulesCore.
    @eval @adjoint function $foo(
        ::Immutable,
        model::LGSSM,
        ys::AV{<:AV{<:Real}},
        f=copy_first,
    )
        @assert length(model) == length(ys)

        # Process first observation.
        (lml, α, x), pb = $step_foo_pullback(model[1], model.gmm.x0, first(ys))
        v = f(α, x)

        # Allocate for remainder of operation.
        vs = Vector{typeof(v)}(undef, length(model))
        pullbacks = Vector{typeof(pb)}(undef, length(ys))
        vs[1] = v
        pullbacks[1] = pb

        # Process remaining observations.
        for t in 2:length(ys)
            (lml_, α, x), pb = $step_foo_pullback(model[t], x, ys[t])
            lml += lml_
            vs[t] = f(α, x)
            pullbacks[t] = pb
        end

        return (lml, vs), function(Δ)

            Δlml = Δ[1]
            Δvs = Δ[2] === nothing ? Fill(nothing, length(vs)) : Δ[2]

            # Compute the pullback through the last element of the chain.
            Δys = Vector{eltype(ys)}(undef, length(ys))
            (Δα, Δx) = get_pb(f)(last(Δvs))
            Δmodel_at_T, Δx, Δy = last(pullbacks)((Δlml, Δα, Δx))
            Δmodel = get_adjoint_storage(model, Δmodel_at_T)
            Δys[end] = Δy

            # Work backwards through the chain.
            for t in reverse(1:length(ys)-1)
                Δα, Δx_ = get_pb(f)(Δvs[t])
                Δx = Zygote.accum(Δx, Δx_)
                Δmodel_at_t, Δx, Δy = pullbacks[t]((Δlml, Δα, Δx))
                Δmodel = _accum_at(Δmodel, t, Δmodel_at_t)
                Δys[t] = Δy
            end

            # Merge all gradient info associated with the model into the same place.
            Δmodel = (
                gmm = merge(Δmodel.gmm, (x0=Δx,)),
                Σ = Δmodel.Σ,
            )

            return nothing, Δmodel, Δys, nothing
        end
    end
end



#
# AD-free pullbacks for a few things. These are primitives that will be used to write the
# gradients.
#

function cholesky_pullback(Σ::Symmetric{<:Real, <:StridedMatrix})
    C = cholesky(Σ)
    return C, function(Δ::NamedTuple)
        U, Ū = C.U, Δ.factors
        Σ̄ = Ū * U'
        Σ̄ = LinearAlgebra.copytri!(Σ̄, 'U')
        Σ̄ = ldiv!(U, Σ̄)
        BLAS.trsm!('R', 'U', 'T', 'N', one(eltype(Σ)), U.data, Σ̄)

        @inbounds for n in diagind(Σ̄)
            Σ̄[n] /= 2
        end
        return (UpperTriangular(Σ̄),)
    end
end

function cholesky_pullback(S::Symmetric{<:Real, <:StaticMatrix{N, N}}) where {N}
    C = cholesky(S)
    return C, function(Δ::NamedTuple)
        U, ΔU = C.U, Δ.factors
        ΔS = U \ (U \ SMatrix{N, N}(Symmetric(ΔU * U')))'
        ΔS = ΔS - Diagonal(ΔS ./ 2)
        return (UpperTriangular(ΔS),)
    end
end

function logdet_pullback(C::Cholesky)
    return logdet(C), function(Δ)
        return ((uplo=nothing, info=nothing, factors=Diagonal(2 .* Δ ./ diag(C.factors))),)
    end
end

AtA_pullback(A::AbstractMatrix{<:Real}) = A'A, Δ->(A * (Δ + Δ'),)



#
# substantial pullbacks
#

@adjoint function predict(m::AV, P::AM, A::AM, a::AV, Q::AM)
    return predict_pullback(m, P, A, a, Q)
end

function predict_pullback(m::AV, P::AM, A::AM, a::AV, Q::AM)
    mp = A * m + a # 1
    T = A * P # 2
    Pp = T * A' + Q # 3
    return (mp, Pp), function(Δ)
        Δmp = Δ[1]
        ΔPp = Δ[2]

        # 3
        ΔQ = ΔPp
        ΔA = ΔPp' * T
        ΔT = ΔPp * A

        # 2
        ΔA += ΔT * P'
        ΔP = A'ΔT

        # 1
        ΔA += Δmp * m'
        Δm = A'Δmp
        Δa = Δmp

        return Δm, ΔP, ΔA, Δa, ΔQ
    end
end

@adjoint function step_decorrelate(model, x::Gaussian, y::AV{<:Real})
    return step_decorrelate_pullback(model, x, y)
end

function step_decorrelate_pullback(model, x::Gaussian, y::AV{<:Real})

    # Evaluate function, keeping track of derivatives.
    (mp, Pp), predict_pb = predict_pullback(x.m, x.P, model.A, model.a, model.Q)
    (mf, Pf, lml, α), update_decorrelate_pb = 
        update_decorrelate_pullback(mp, Pp, model.H, model.h, model.Σ, y)

    return (lml, α, Gaussian(mf, Pf)), function(Δ)

        # Unpack stuff.
        Δlml, Δα, Δx = Δ
        Δmf = Δx === nothing ? zero(mp) : Δx.m
        ΔPf = Δx === nothing ? zero(Pp) : Δx.P

        # Backprop through stuff.
        Δmp, ΔPp, ΔH, Δh, ΔΣ, Δy = update_decorrelate_pb((Δmf, ΔPf, Δlml, Δα))
        Δmf, ΔPf, ΔA, Δa, ΔQ = predict_pb((Δmp, ΔPp))

        Δx = (m=Δmf, P=ΔPf)
        Δmodel = (A=ΔA, a=Δa, Q=ΔQ, H=ΔH, h=Δh, Σ=ΔΣ)
        return Δmodel, Δx, Δy
    end
end

@adjoint function update_decorrelate(m, P, H, h, Σ, y)
    return update_decorrelate_pullback(m, P, H, h, Σ, y)
end

function update_decorrelate_pullback(mp, Pp, H::AM, h::AV, Σ::AM, y::AV{<:Real})

    V = H * Pp # 1
    S_1 = V * H' + Σ # 2
    S, S_pb = cholesky_pullback(Symmetric(S_1)) # 2.1
    U = S.U # 3
    B = U' \ V # 4
    η = y - H * mp - h # 5
    α = U' \ η # 6

    mf = mp + B'α # 7
    BtB, BtB_pb = AtA_pullback(B) # 8
    Pf = Pp - BtB # 9

    logdet_S, logdet_S_pb = logdet_pullback(S) # 10
    lml = -(length(y) * log(2π) + logdet_S + α'α) / 2 # 11

    return (mf, Pf, lml, α), function(Δ)
        Δmf, ΔPf, Δlml, Δα = Δ

        Δlml = Δlml === nothing ? zero(lml) : Δlml
        Δα = Δα === nothing ? zero(α) : Δα

        # 11
        Δα = Δα .- Δlml * α
        Δlogdet_S = -Δlml / 2

        # 10
        ΔS = first(logdet_S_pb(Δlogdet_S))

        # 9
        ΔPp = ΔPf
        ΔBtB = -ΔPf

        # 8
        ΔB = first(BtB_pb(ΔBtB))

        # 7
        Δmp = Δmf
        Δα += B * Δmf
        ΔB += α * Δmf'

        # 6
        Δη = U \ Δα
        ΔU = -α * Δη'

        # 5
        Δy = Δη
        ΔH = -Δη * mp'
        Δmp += -H'Δη
        Δh = -Δη

        # 4
        ΔV = U \ ΔB
        ΔU += -B * ΔV'

        # 3
        ΔS = (uplo=nothing, info=nothing, factors=get_ΔS(ΔS.factors, UpperTriangular(ΔU)))

        # 2.1
        ΔS_1 = first(S_pb(ΔS))

        # 2
        ΔV += ΔS_1 * H
        ΔH += ΔS_1'V
        ΔΣ = my_collect(ΔS_1)

        # 1
        ΔH += ΔV * Pp'
        ΔPp += H'ΔV

        return Δmp, ΔPp, ΔH, Δh, ΔΣ, Δy
    end
end

get_ΔS(A, B) = A + B

function get_ΔS(
    A::Diagonal{<:Any, <:SVector{D}},
    B::UpperTriangular{<:Any, <:SMatrix{D, D}},
) where {D}
    return SMatrix{D, D}(A) + SMatrix{D, D}(B)
end

@adjoint function step_correlate(model, x::Gaussian, α::AV{<:Real})
    return step_correlate_pullback(model, x, α)
end

function step_correlate_pullback(model, x::Gaussian, α::AV{<:Real})

    # Evaluate function, keeping track of derivatives.
    (mp, Pp), predict_pb = predict_pullback(x.m, x.P, model.A, model.a, model.Q)
    (mf, Pf, lml, y), update_decorrelate_pb = 
        update_correlate_pullback(mp, Pp, model.H, model.h, model.Σ, α)

    return (lml, y, Gaussian(mf, Pf)), function(Δ)

        # Unpack stuff.
        Δlml, Δy, Δx = Δ
        Δmf = Δx === nothing ? zeros(size(mp)) : Δx.m
        ΔPf = Δx === nothing ? zeros(size(Pp)) : Δx.P

        # Backprop through stuff.
        Δmp, ΔPp, ΔH, Δh, ΔΣ, Δα = update_decorrelate_pb((Δmf, ΔPf, Δlml, Δy))
        Δmf, ΔPf, ΔA, Δa, ΔQ = predict_pb((Δmp, ΔPp))

        Δx = (m=Δmf, P=ΔPf)
        Δmodel = (A=ΔA, a=Δa, Q=ΔQ, H=ΔH, h=Δh, Σ=ΔΣ)
        return Δmodel, Δx, Δα
    end
end

@adjoint function update_correlate(mp, Pp, H, h, Σ, α)
    return update_correlate_pullback(mp, Pp, H, h, Σ, α)
end

function update_correlate_pullback(mp, Pp, H::AM, h::AV, Σ::AM, α::AV{<:Real})

    V = H * Pp # 1
    S_1 = V * H' + Σ # 2
    S, S_pb = cholesky_pullback(Symmetric(S_1)) # 2.1
    U = S.U # 3
    B = U' \ V # 4
    y = U'α + H * mp + h # 5

    mf = mp + B'α # 6
    BtB, BtB_pb = AtA_pullback(B) # 7
    Pf = Pp - BtB # 8

    logdet_S, logdet_S_pb = logdet_pullback(S) # 9
    lml = -(length(y) * log(2π) + logdet_S + α'α) / 2 # 10

    return (mf, Pf, lml, y), function(Δ)
        Δmf, ΔPf, Δlml, Δy = Δ

        Δlml = Δlml === nothing ? zero(lml) : Δlml
        Δy = Δy === nothing ? zero(y) : Δy

        # 10
        Δα = (-Δlml) * α
        Δlogdet_S = -Δlml / 2

        # 9
        ΔS = first(logdet_S_pb(Δlogdet_S))

        # 8
        ΔPp = ΔPf
        ΔBtB = -ΔPf

        # 7
        ΔB = first(BtB_pb(ΔBtB))

        # 6
        Δmp = Δmf
        Δα += B * Δmf
        ΔB += α * Δmf'

        # 5
        Δα += U * Δy
        ΔU = α * Δy'
        ΔH = Δy * mp'
        Δmp += H'Δy
        Δh = Δy

        # 4
        ΔV = U \ ΔB
        ΔU += -B * ΔV'

        # 3
        ΔS = Zygote.accum(ΔS, (uplo=nothing, info=nothing, factors=UpperTriangular(ΔU),))

        # 2.1
        ΔS_1 = my_collect(first(S_pb(ΔS)))

        # 2
        ΔV += ΔS_1 * H
        ΔH += ΔS_1'V
        ΔΣ = ΔS_1

        # 1
        ΔH += ΔV * Pp'
        ΔPp += H'ΔV

        return Δmp, ΔPp, ΔH, Δh, ΔΣ, Δα
    end
end

my_collect(A::AbstractMatrix) = collect(A)
function my_collect(A::UpperTriangular{T, <:SMatrix{D, D, T}}) where {T<:Real, D}
    return SMatrix{D, D}(A)
end
