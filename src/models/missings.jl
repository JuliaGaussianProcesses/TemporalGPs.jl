# Several strategies for missing data handling were attempted.
# 1. Use `missing`s as expected. This turned out to be problematic for type-stability.
# 2. Sentinel values (NaNs). Also problematic for type-stability because Zygote.
# 3. (The adopted strategy) - replace missings with arbitrary observations and _large_
#   observation noises. While not optimal, type-stability is preserved inside the
#   performance-sensitive code.
#
# In an ideal world, strategy 1 would work. Unfortunately Zygote isn't up to it yet.

function decorrelate(
    model::LGSSM, ys::AbstractVector{<:Union{T, Missing}},
) where {T<:AbstractVector{<:Real}}
    Σs_filled_in, ys_filled_in = fill_in_missings(model.Σ, ys)
    model_with_missings = LGSSM(model.gmm, Σs_filled_in)
    uncorrected_lml, αs, xs = decorrelate(model_with_missings, ys_filled_in)
    lml = uncorrected_lml - _logpdf_volume_compensation(ys)
    return lml, αs, xs
end

function fill_in_missings(
    Σs::AbstractVector{<:AbstractMatrix}, y::AbstractVector{Union{T, Missing}},
) where {T<:AbstractVector{<:Real}}

    # Fill in observation covariance matrices with very large values.
    Σs_filled_in = map(n -> y[n] isa Missing ? build_large_var(Σs[n]) : Σs[n], eachindex(y))

    # Fill in missing y's with zero vectors (any other choice would also have been fine).
    y_filled_in = Vector{T}(undef, length(y))
    map!(
        n -> y[n] isa Missing ? get_zero(size(Σs[n], 1), T) : y[n],
        y_filled_in,
        eachindex(y),
    )

    return Σs_filled_in, y_filled_in
end

function _logpdf_volume_compensation(y)
    N_missing = count(ismissing, y)
    return N_missing * length(y) * log(2π * 1e9)
end

Zygote.@nograd _logpdf_volume_compensation

function make_missing(αs_filled::AbstractVector, ys::AbstractVector)
    return map((α, y) -> y isa Missing ? missing : α, αs_filled, ys)
end

@adjoint function fill_in_missings(
    Σs::AbstractVector{<:AbstractMatrix}, y::AbstractVector{Union{T, Missing}},
) where {T}
    function pullback_fill_in_missings(Δ)
        ΔΣs_filled_in = Δ[1]
        Δy_filled_in = Δ[2]

        # The cotangent of a `Missing` doesn't make sense, so should be a `DoesNotExist`.
        Δy = Vector{Union{eltype(Δy_filled_in), DoesNotExist}}(undef, length(y))
        map!(
            n -> y[n] isa Missing ? DoesNotExist() : Δy_filled_in[n],
            Δy,
            eachindex(y),
        )

        # Fill in missing locations with zeros. Opting for type-stability to keep things
        # simple.
        ΔΣs = map(
            n -> y[n] isa Missing ? zero(Σs[n]) : ΔΣs_filled_in[n],
            eachindex(y),
        )

        return ΔΣs, Δy
    end
    return fill_in_missings(Σs, y), pullback_fill_in_missings
end

get_zero(D::Int, ::Type{Vector{T}}) where {T} = zeros(T, D)

get_zero(::Int, ::Type{T}) where {T<:SVector} = zeros(T)

build_large_var(S::T) where {T<:Matrix} = T(1e9I, size(S))

build_large_var(S::T) where {T<:SMatrix} = T(1e9I)

ChainRulesCore.@non_differentiable build_large_var(::AbstractMatrix)
