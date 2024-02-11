using Test
using TemporalGPs: scan_emit
using StructArrays

@testset "scan" begin
    # Run forwards.
    x = StructArray([(a=randn(), b=randn()) for _ in 1:10])
    stepper = (x_, y_) -> (x_ + y_.a * y_.b * x_, x_ + y_.b)
end
