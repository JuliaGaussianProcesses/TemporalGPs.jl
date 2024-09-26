module TemporalGPsMooncakeExt

using Mooncake, TemporalGPs
import Mooncake: rrule!!, CoDual, primal, @is_primitive, zero_fcodual, MinimalCtx

@is_primitive MinimalCtx Tuple{typeof(TemporalGPs.time_exp), AbstractMatrix{<:Real}, Real}
function rrule!!(::CoDual{typeof(TemporalGPs.time_exp)}, A::CoDual, t::CoDual{Float64})
    B_dB = zero_fcodual(TemporalGPs.time_exp(primal(A), primal(t)))
    B = primal(B_dB)
    dB = tangent(B_dB)
    time_exp_pb(::NoRData) = NoRData(), NoRData(), sum(dB .* (primal(A) * B))
    return B_dB, time_exp_pb
end

end
