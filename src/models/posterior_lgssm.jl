"""
    PosteriorLGSSM{Tmodel<:LGSSM, Txfs<:AbstractVector{<:Gaussian}} <: AbstractSSM

Represents the posterior distribution over an LGSSM given the filtering distributions xfs.
"""
struct PosteriorLGSSM{Tmodel<:LGSSM, Txfs<:AbstractVector{<:Gaussian}} <: AbstractSSM
    model::Tmodel
    xfs::Txfs
end

Base.:(==)(x::LGSSM, y::LGSSM) = (x.model == y.model) && (x.xfs == y.xfs)

Base.length(ft::LGSSM) = length(ft.model)

dim_obs(ft::LGSSM) = dim_obs(ft.model)

dim_latent(ft::LGSSM) = dim_latent(ft.model)

Base.eltype(ft::LGSSM) = eltype(ft.model)

storage_type(ft::LGSSM) = storage_type(ft.model)

Zygote.@nograd storage_type

function is_of_storage_type(model::PosteriorLGSSM, s::StorageType)
    return is_of_storage_type((model.model, model.xfs), s)
end

is_time_invariant(model::PosteriorLGSSM) = false

Base.getindex(model::PosteriorLGSSM, n::Int) = (prior=model.model[t], xf=model.xfs[n])

# mean(model::LGSSM) = mean(model.gmm)

# function cov(model::LGSSM)
#     S = Stheno.cov(model.gmm)
#     Σ = Stheno.block_diagonal(model.Σ)
#     return S + Σ
# end

function marginals(model::PosteriorLGSSM)
    
end
