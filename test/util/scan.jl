using Test
using Zygote: ZygoteRuleConfig
using TemporalGPs: scan_emit

@testset "scan" begin

    # Run forwards.
    x = StructArray([(a=randn(), b=randn()) for _ in 1:10])
    stepper = (x_, y_) -> (x_ + y_.a * y_.b * x_, x_ + y_.b)
    test_rrule(scan_emit, stepper, x, 0.0, eachindex(x))

    # Run in reverse.
    test_rrule(scan_emit, stepper, x, 0.0, reverse(eachindex(x)))
end
