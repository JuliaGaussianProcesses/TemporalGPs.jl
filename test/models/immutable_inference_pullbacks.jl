# This file contains a collection of optimisations for use with reveerse-mode AD. 
# Consequently, it is not necessary to understand the contents of this file to understand
# the package as a whole.

using TemporalGPs:
    is_of_storage_type,
    Gaussian,
    cholesky_pullback,
    logdet_pullback,
    update_correlate,
    step_correlate,
    correlate,
    update_decorrelate,
    step_decorrelate,
    decorrelate


function verify_pullback(f_pullback, input, Δoutput, storage)
    output, _pb = f_pullback(input...)
    Δinput = _pb(Δoutput)

    @test is_of_storage_type(input, storage.val)
    @test is_of_storage_type(output, storage.val)
    @test is_of_storage_type(Δinput, storage.val)
    @test is_of_storage_type(Δoutput, storage.val)

    if storage.val isa SArrayStorage
        @test allocs(@benchmark $f_pullback($input...)) == 0
        @test allocs(@benchmark $_pb($Δoutput)) == 0
    end
end

@testset "immutable_inference_pullbacks" begin

end
