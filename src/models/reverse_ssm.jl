"""
Transition dynamics take you from t to t-1, rather than t-1 to t.
"""
struct ReverseSSM{Tmodel<:AbstractSSM} <: AbstractSSM
    model::Tmodel
end

struct ElementOfReverseSSM{T}
    data::T
end

Base.eachindex(model::ReverseSSM) = reverse(eachindex(model.model))

Base.getindex(model::ReverseSSM, t::Int) = ElementOfReverseSSM(model.model[t])

x0(model::ReverseSSM) = x0(model.model)

function step_marginals(x::Gaussian, model::ElementOfReverseSSM)
    data = model.data
    y = observe(data, x)
    x = predict(data, x)
    return y, x
end

function step_decorrelate(x::Gaussian, (model, y)::Tuple{ElementOfReverseSSM, Any})
    model = model.data
    gmm = model.gmm
    mf, Pf, lml, α = update_decorrelate(x.m, x.P, gmm.H, gmm.h, model.Σ, y)
    mp, Pp = predict(mf, Pf, gmm.A, gmm.a, gmm.Q)
    return (lml, α), Gaussian(mp, Pp)
end

function step_correlate(x::Gaussian, (model, α)::Tuple{ElementOfReverseSSM, Any})
    model = model.data
    gmm = model.gmm
    mf, Pf, lml, y = update_correlate(x.m, x.P, gmm.H, gmm.h, model.Σ, α)
    mp, Pp = predict(mf, Pf, gmm.A, gmm.a, gmm.Q)
    return (lml, y), Gaussian(mp, Pp)
end
