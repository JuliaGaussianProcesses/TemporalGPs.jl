struct LinearGaussianDynamics{TA, Ta, TQ}
    A::TA
    a::Ta
    Q::TQ
end

function Base.:(==)(x::LinearGaussianDynamics, y::LinearGaussianDynamics)
    return (x.A == y.A) && (x.a == y.a) && (x.Q == y.Q)
end

function predict(x::Gaussian, f::LinearGaussianDynamics)
    return Gaussian(f.A * x.m + f.a, f.A * x.P * f.A' + f.Q)
end

function correlate(x::Gaussian, f::LinearGaussianDynamics, α)
    A = f.A
    V = A * x.P
    S = cholesky(Symmetric(V * A' + f.Q))
    B = S.U' \ V
    y = S.U'α + (A * x.m + f.a)

    lml = -(length(y) * convert(scalar_type(α), log(2π)) + logdet(S) + α'α) / 2
    return Gaussian(x.m + B'α, x.P - B'B), lml, y
end

function decorrelate(x::Gaussian, f::LinearGaussianDynamics, y)
    A = f.A
    V = A * x.P
    S = cholesky(Symmetric(V * A' + f.Q))
    B = S.U' \ V
    α = S.U' \ (y - (A * x.m - f.a))

    lml = -(length(y) * convert(scalar_type(y), log(2π)) + logdet(S) + α'α) / 2
    return Gaussian(x.m + B'α, x.P - B'B), lml, α
end

scalar_type(x::AbstractVector{T}) where {T} = T
scalar_type(x::T) where {T<:Real} = T

Zygote.@nograd scalar_type
