using TemporalGPs: UniLTISDE, StaticStorage, DenseStorage, to_sde

@testset "gp_to_sde" begin
    t0, Δt, T = 0.0, 0.3, 101
    t_range = range(t0; step=Δt, length=T)
    t = collect(t_range)
    σ²_n = 0.3

    base_kernels = [Matern12(), Matern32(), Matern52()]
    ls = [1e-4, 0.1, 1.0, 10.0, 100.0]
    σ²s = [1e-1, 1.0, 10.0, 100.0]

    @testset "LTISDE from $kernel" for kernel in base_kernels

        # Storage types represent the same things.
        @test to_sde(kernel, DenseStorage()) == to_sde(kernel, DenseStorage())
        @test to_sde(kernel, StaticStorage()) == to_sde(kernel, StaticStorage())
        @test to_sde(kernel, DenseStorage()) == to_sde(kernel, StaticStorage())

        # StaticStorage constructors don't allocate.
        @test (@ballocated to_sde($kernel, StaticStorage())) == 0
        @test (@ballocated to_sde($(5.0 * kernel), StaticStorage())) == 0
        @test (@ballocated to_sde($(stretch(kernel, 1.0)), StaticStorage())) == 0

        # single-output kernels produce single-output SDEs.
        @test to_sde(kernel, DenseStorage()) isa UniLTISDE
        @test to_sde(kernel, StaticStorage()) isa UniLTISDE

        # GPs with either storage type produce the same SDE. (DenseStorage default).
        @test to_sde(GP(kernel, GPC())) == to_sde(GP(kernel, GPC()), StaticStorage())

        # Converting a GP to an SDE with static storage produces no extra allocs.
        @test (@ballocated to_sde($(GP(kernel, GPC())), StaticStorage())) == 0

        # Singe-output GPs produce single-output SDEs.
        @test to_sde(GP(kernel, GPC())) isa UniLTISDE
        @test to_sde(GP(kernel, GPC()), StaticStorage()) isa UniLTISDE
    end
end
