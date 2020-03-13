#
# This file contains pullbacks for stuff in generic.jl. These are purely performance
# optimisations for algorithmic differentiation, and in no way important for understanding
# the structure of the package, or its functionality.
#

Zygote.@adjoint function LGSSM(A, Q, H, Σ, x₀, T)
    return LGSSM(A, Q, H, Σ, x₀, T), function(Δ)
        return (
            maybegetfield(Δ, Val(:A)),
            maybegetfield(Δ, Val(:Q)),
            maybegetfield(Δ, Val(:H)),
            maybegetfield(Δ, Val(:Σ)),
            maybegetfield(Δ, Val(:x₀)),
            nothing,
        )
    end 
end

function Zygote.accum(a::UpperTriangular, b::UpperTriangular)
    return UpperTriangular(Zygote.accum(a.data, b.data))
end

function Zygote.accum(D::Diagonal{<:Real}, U::UpperTriangular{<:Real, <:SMatrix})
    return UpperTriangular(D + U.data)
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


@adjoint function step_decorrelate(model, x::Gaussian, y::AV{<:Real})
    return step_decorrelate_pullback(model, x, y)
end

function step_decorrelate_pullback(model, x::Gaussian, y::AV{<:Real})

    # Evaluate function, keeping track of derivatives.
    (mp, Pp), _predict_pb = _predict_pullback(x.m, x.P, model.A, model.Q)
    (mf, Pf, lml, α), update_decorrelate_pb = 
        update_decorrelate_pullback(mp, Pp, model.H, model.Σ, y)

    return (lml, α, Gaussian(mf, Pf)), function(Δ)

        # Unpack stuff.
        Δlml, Δα, Δx = Δ
        Δmf = Δx === nothing ? zero(mp) : Δx.m
        ΔPf = Δx === nothing ? zero(Pp) : Δx.P

        # Backprop through stuff.
        Δmp, ΔPp, ΔH, ΔΣ, Δy = update_decorrelate_pb((Δmf, ΔPf, Δlml, Δα))
        Δmf, ΔPf, ΔA, ΔQ = _predict_pb((Δmp, ΔPp))

        Δx = (m=Δmf, P=ΔPf)
        Δmodel = (A=ΔA, Q=ΔQ, H=ΔH, Σ=ΔΣ)
        return Δmodel, Δx, Δy
    end
end

@adjoint _predict(m::AV, P::AM, A::AM, Q::AM) = _predict_pullback(m, P, A, Q)

function _predict_pullback(m::AV, P::AM, A::AM, Q::AM)
    mp = A * m # 1
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

        return Δm, ΔP, ΔA, ΔQ
    end
end

@adjoint function update_decorrelate(m, P, h, σ², y)
    return update_decorrelate_pullback(m, P, h, σ², y)
end

function update_decorrelate_pullback(mp, Pp, H::AM, Σ::AM, y::AV{<:Real})

    V = H * Pp # 1
    S_1 = V * H' + Σ # 2
    S, S_pb = cholesky_pullback(Symmetric(S_1)) # 2.1
    U = S.U # 3
    B = U' \ V # 4
    η = y - H * mp # 5
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
        ΔΣ = ΔS_1

        # 1
        ΔH += ΔV * Pp'
        ΔPp += H'ΔV

        return Δmp, ΔPp, ΔH, ΔΣ, Δy
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
    (mp, Pp), _predict_pb = _predict_pullback(x.m, x.P, model.A, model.Q)
    (mf, Pf, lml, y), update_decorrelate_pb = 
        update_correlate_pullback(mp, Pp, model.H, model.Σ, α)

    return (lml, y, Gaussian(mf, Pf)), function(Δ)

        # Unpack stuff.
        Δlml, Δy, Δx = Δ
        Δmf = Δx === nothing ? zeros(size(mp)) : Δx.m
        ΔPf = Δx === nothing ? zeros(size(Pp)) : Δx.P

        # Backprop through stuff.
        Δmp, ΔPp, ΔH, ΔΣ, Δα = update_decorrelate_pb((Δmf, ΔPf, Δlml, Δy))
        Δmf, ΔPf, ΔA, ΔQ = _predict_pb((Δmp, ΔPp))

        Δx = (m=Δmf, P=ΔPf)
        Δmodel = (A=ΔA, Q=ΔQ, H=ΔH, Σ=ΔΣ)
        return Δmodel, Δx, Δα
    end
end

@adjoint update_correlate(mp, Pp, h, σ², α) = update_correlate_pullback(mp, Pp, h, σ², α)

function update_correlate_pullback(mp, Pp, H::AM, Σ::AM, α::AV{<:Real})

    V = H * Pp # 1
    S_1 = V * H' + Σ # 2
    S, S_pb = cholesky_pullback(Symmetric(S_1)) # 2.1
    U = S.U # 3
    B = U' \ V # 4
    y = U'α + H * mp # 5

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

        return Δmp, ΔPp, ΔH, ΔΣ, Δα
    end
end

my_collect(A::AbstractMatrix) = collect(A)
function my_collect(A::UpperTriangular{T, <:SMatrix{D, D, T}}) where {T<:Real, D}
    return SMatrix{D, D}(A)
end
