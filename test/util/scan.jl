using TemporalGPs: scan_emit

@testset "scan" begin

    # Run forwards.
    x = StructArray([(a=randn(), b=randn()) for _ in 1:100])
    stepper = (x_, y_) -> (x_ + y_.a * y_.b * x_, x_ + y_.b)
    adjoint_test((init, x) -> scan_emit(stepper, x, init, eachindex(x)), (0.0, x))

    # Run in reverse.
    adjoint_test((init, x) -> scan_emit(stepper, x, init, reverse(eachindex(x))), (0.0, x))
end
