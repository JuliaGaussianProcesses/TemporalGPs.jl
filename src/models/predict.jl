#
# Generic implementation. Good for StaticArrays.
#

@inline function predict(mf::AV, Pf::AM, A::AM, a::AV, Q::AM)
    return A * mf + a, (A * Pf) * A' + Q
end

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


#
# `A <: Matrix{<:Real}`.
#

function predict(
    mf::Vector{T},
    Pf::Symmetric{T, Matrix{T}},
    A::AM{T},
    a::Vector{T},
    Q::Matrix{T},
) where {T<:Real}
    mp = Vector{T}(undef, size(mf))
    Pp = Matrix{T}(undef, size(Pf))
    return predict!(mp, Pp, mf, Pf, A, a, Q)
end

function predict!(
    mp::Vector{T},
    Pp::Matrix{T},
    mf::Vector{T},
    Pf::Symmetric{T, Matrix{T}},
    A::AM{T},
    a::Vector{T},
    Q::Matrix{T},
) where {T<:Real}

    # Compute predictive mean.
    mp = mul!(copy!(mp, a), A, mf, one(T), one(T))

    # Compute predictive covariance.
    APf = mul!(Matrix{T}(undef, size(Pf)), A, Pf, one(T), zero(T))
    Pp = mul!(copy!(Pp, Q), APf, A', one(T), one(T))

    return mp, Pp
end

function predict_pullback(
    mf::Vector{T},
    Pf::Symmetric{T, Matrix{T}},
    A::Matrix{T},
    a::Vector{T},
    Q::Matrix{T},
) where {T<:Real}

    # Pre-allocate for output.
    mp = Vector{T}(undef, size(mf))
    Pp = Matrix{T}(undef, size(Pf))

    # 1: Compute predictive mean.
    mp = mul!(copy!(mp, a), A, mf, one(T), one(T))

    # 2: compute A * Pf
    APf = mul!(Matrix{T}(undef, size(Pf)), A, Pf, one(T), zero(T))

    # 3: compute APf * A' + Q
    Pp = mul!(copy!(Pp, Q), APf, A', one(T), one(T))

    return (mp, Pp), function(Δ)
        Δmp = Δ[1]
        ΔPp = Δ[2]

        # Pre-allocate for cotangents.
        Δmf = Vector{T}(undef, size(mf))
        ΔPf = Matrix{T}(undef, size(Pf))
        ΔAPf = Matrix{T}(undef, size(APf))
        ΔA = Matrix{T}(undef, size(A))

        # 3
        ΔQ = ΔPp
        ΔA = mul!(ΔA, ΔPp', APf)
        ΔAPf = mul!(ΔAPf, ΔPp, A)

        # 2
        ΔA = mul!(ΔA, ΔAPf, Pf', one(T), one(T))
        ΔPf = mul!(ΔPf, A', ΔAPf)

        # 1
        ΔA = mul!(ΔA, Δmp, mf', one(T), one(T))
        Δmf = mul!(Δmf, A', Δmp)
        Δa = Δmp

        return Δmf, ΔPf, ΔA, Δa, ΔQ
    end
end



#
# A <: BlockDiagonal{<:Real}
#

function predict(
    mf::Vector{T},
    Pf::Symmetric{T, Matrix{T}},
    A::BlockDiagonal{T, TM},
    a::Vector{T},
    Q::BlockDiagonal{T, TM},
) where {T<:Real, TM<:AbstractMatrix{T}}
    return nothing
end
