# #
# # This file contains pullbacks for stuff in immutable_inference.jl. There is no good reason
# # to understand what's going on here unless you're working on AD and this package.
# #

# #
# # Objects in which to storage / accumulate the adjoint w.r.t. the hypers.
# #

# zeroed_adjoint(x::Array{<:Real}) = zero(x)
# zeroed_adjoint(x::SArray{<:Any, <:Real}) = zero(x)
# zeroed_adjoint(::Type{T}) where {T<:Real} = zero(T)
# zeroed_adjoint(::Type{Union{Nothing, T}}) where {T<:Real} = nothing

# zeroed_adjoint(x::Gaussian) = (m=zeroed_adjoint(x.m), P=zeroed_adjoint(x.P))

# get_adjoint(Δx::AbstractVector, n::Int) = Δx[n]
# get_adjoint(::Nothing, ::Int) = nothing

# get_adjoint(x, Δx::AbstractVector, n::Int) = Δx[n]
# get_adjoint(x, ::Nothing, ::Int) = zeroed_adjoint(x)

# for (foo, step_foo, foo_pullback) in [
#     (:correlate, :step_correlate, :correlate_pullback),
#     (:decorrelate, :step_decorrelate, :decorrelate_pullback),
# ]
#     @eval function Zygote._pullback(
#         ::AContext, ::typeof($foo), model::LGSSM, ys::AbstractVector{<:AbstractVector},
#     )
#         lml, αs, xs = $foo(model, ys)

#         function step_pullback(t::Int, Δlml, Δαs, Δx, model, ys)
#             x = t > 1 ? xs[t - 1] : model.gmm.x0
#             _, pb = _pullback(NoContext(), $step_foo, model[t], x, ys[t])
#             Δ_t = (Δlml, get_adjoint(αs[t], Δαs, t), Δx)
#             _, Δmodel_t, Δx, Δy = pb(Δ_t)
#             return Δmodel_t, Δx, Δy
#         end

#         function $foo_pullback(
#             Δ::Union{
#                 Nothing,
#                 Tuple{Any, Union{AbstractVector, Nothing}, Union{AbstractVector, Nothing}},
#             },
#         )

#             Δ = Δ === nothing ? (nothing, nothing, nothing) : Δ
#             Δlml = Δ[1]
#             Δαs = Δ[2]
#             Δxs = Δ[3]

#             # Get model adjoint type by performing 1st iteration of reverse-pass.
#             T = length(model)
#             Δmodel_T, Δx, Δy = step_pullback(T, Δlml, Δαs, get_adjoint(Δxs, T), model, ys)
#             Δmodel = get_adjoint_storage(model, Δmodel_T)
#             Δys = get_adjoint_storage(ys, Δy)

#             # Iterate backwards through the data.
#             for t in reverse(1:(T-1))
#                 Δx = accum(get_adjoint(Δxs, t), Δx)
#                 Δmodel_t, Δx, Δy = step_pullback(t, Δlml, Δαs, Δx, model, ys)
#                 Δmodel = _accum_at(Δmodel, t, Δmodel_t)
#                 Δys[t] = Δy
#             end

#             return nothing, (gmm=merge(Δmodel.gmm, (x0=Δx,)), Σ=Δmodel.Σ), Δys
#         end

#         return (lml, αs, xs), $foo_pullback
#     end
# end
