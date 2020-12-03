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
    @testset "$N, $T" for N in [1, 2, 3], T in [Float32, Float64]

        rng = MersenneTwister(123456)

        # Do dense stuff.
        S_ = randn(rng, T, N, N)
        S = S_ * S_' + I
        C = cholesky(S)
        Ss = SMatrix{N, N, T}(S)
        Cs = cholesky(Ss)

        @testset "cholesky" begin
            C_fwd, pb = cholesky_pullback(Symmetric(S))
            Cs_fwd, pbs = cholesky_pullback(Symmetric(Ss))

            @test eltype(C_fwd) == T
            @test eltype(Cs_fwd) == T

            ΔC = randn(rng, T, N, N)
            ΔCs = SMatrix{N, N, T}(ΔC)

            @test C.U ≈ Cs.U
            @test Cs.U ≈ Cs_fwd.U

            ΔS, = pb((factors=ΔC, ))
            ΔSs, = pbs((factors=ΔCs, ))

            @test ΔS ≈ ΔSs
            @test eltype(ΔS) == T
            @test eltype(ΔSs) == T

            @test allocs(@benchmark cholesky(Symmetric($Ss))) == 0
            @test allocs(@benchmark cholesky_pullback(Symmetric($Ss))) == 0
            @test allocs(@benchmark $pbs((factors=$ΔCs,))) == 0
        end
        @testset "logdet" begin
            @test logdet(Cs) ≈ logdet(C)
            C_fwd, pb = logdet_pullback(C)
            Cs_fwd, pbs = logdet_pullback(Cs)

            @test eltype(C_fwd) == T
            @test eltype(Cs_fwd) == T

            @test logdet(Cs) ≈ Cs_fwd

            Δ = randn(rng, T)
            ΔC = first(pb(Δ)).factors
            ΔCs = first(pbs(Δ)).factors

            @test ΔC ≈ ΔCs
            @test eltype(ΔC) == T
            @test eltype(ΔCs) == T

            @test allocs(@benchmark logdet($Cs)) == 0
            @test allocs(@benchmark logdet_pullback($Cs)) == 0
            @test allocs(@benchmark $pbs($Δ)) == 0
        end
    end
end
