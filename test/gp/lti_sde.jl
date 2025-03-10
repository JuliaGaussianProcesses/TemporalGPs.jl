using KernelFunctions
using KernelFunctions: kappa
using TemporalGPs: build_lgssm, StorageType, is_of_storage_type, lgssm_components
using Test

# Everything is tested once the LGSSM is constructed, so the logpdf bit of this test
# function is probably redundant. It is good to do a little bit of integration testing
# though.
function _logpdf_tester(f_naive::GP, y, storage::StorageType, σ², t::AbstractVector)
    f = to_sde(f_naive, storage)
    return logpdf(f(t, σ²...), y)
end

println("lti_sde:")
@testset "lti_sde" begin
    @testset "ApproxPeriodicKernel" begin
        k = ApproxPeriodicKernel()
        @test k isa ApproxPeriodicKernel{7}
        # Test that it behaves like a normal PeriodicKernel
        k_base = PeriodicKernel()
        x = rand()
        @test kappa(k, x) == kappa(k_base, x)
        x = rand(3)
        @test kernelmatrix(k, x) ≈ kernelmatrix(k_base, x)
        # Test dimensionality of LGSSM components
        Nt = 10
        @testset "$(typeof(t)), $storage, $N" for t in (
                sort(rand(Nt)), RegularSpacing(0.0, 0.1, Nt)
            ),
            storage in (ArrayStorage{Float64}(), ),
            N in (5, 8)

            k = ApproxPeriodicKernel{N}()
            As, as, Qs, emission_projections, x0 = lgssm_components(k, t, storage)
            @test length(As) == Nt
            @test all(x -> size(x) == (N * 2, N * 2), As)
            @test length(as) == Nt
            @test all(x -> size(x) == (N * 2,), as)
            @test length(Qs) == Nt
            @test all(x -> size(x) == (N * 2, N * 2), Qs)
        end
    end

    @testset "block_diagonal" begin
        rng = Xoshiro(123)
        A = randn(2, 2)
        B = randn(3, 3)
        C = randn(5, 5)
        test_rule(rng, TemporalGPs.block_diagonal, A, B, C; is_primitive=false)
        test_rule(
            rng,
            TemporalGPs.block_diagonal,
            SMatrix{2,2}(A),
            SMatrix{3,3}(B),
            SMatrix{5,5}(C);
            is_primitive=false,
        )
    end

    @testset "SimpleKernel parameter types" begin
        storages = (
            (name="dense storage Float64", val=ArrayStorage(Float64)),
            (name="static storage Float64", val=SArrayStorage(Float64)),
            # (name="dense storage Float32", val=ArrayStorage(Float32)),
            # (name="static storage Float32", val=SArrayStorage(Float32)),
        )

        kernels = [
            Matern12Kernel(),
            Matern32Kernel(),
            Matern52Kernel(),
            ConstantKernel(; c=1.5),
            CosineKernel(),
        ]

        @testset "$kernel, $(storage.name)" for kernel in kernels, storage in storages
            F, q, H = TemporalGPs.to_sde(kernel, storage.val)
            @test is_of_storage_type(F, storage.val)
            @test is_of_storage_type(q, storage.val)
            @test is_of_storage_type(H, storage.val)

            x = TemporalGPs.stationary_distribution(kernel, storage.val)
            @test is_of_storage_type(x, storage.val)
        end
    end

    @testset "lgssm_components" begin
        rng = MersenneTwister(123456)
        N = 13
        kernels = vcat(
            # Base kernels.
            (name="base-Matern12Kernel", val=Matern12Kernel()),
            map([Matern32Kernel, Matern52Kernel]) do k
                (; name="base-$k", val=k())
            end,

            # Scaled kernels.
            map([1e-1, 1.0, 10.0, 100.0]) do σ²
                (; name="scaled-σ²=$σ²", val=σ² * Matern32Kernel())
            end,

            # Stretched kernels.
            map([1e-2, 0.1, 1.0, 10.0, 100.0]) do λ
                (; name="stretched-λ=$λ", val=Matern32Kernel() ∘ ScaleTransform(λ))
            end,

            # Approx periodic kernels
            map([7, 11]) do N
                (name="approx-periodic-N=$N", val=ApproxPeriodicKernel{N}(; r=1.0))
            end,

            # Product kernels
            (
                name="prod-Matern52Kernel-Matern32Kernel",
                val=(1.5 * Matern52Kernel() * Matern32Kernel()) ∘ ScaleTransform(0.01),
            ),
            (
                name="prod-Matern32Kernel-Matern52Kernel-ConstantKernel",
                val=3.0 * Matern32Kernel() * Matern52Kernel() * ConstantKernel(),
            ),
            # This is known not to work at all (not a gradient problem).
            # (
            #     name="prod-(Matern32Kernel + ConstantKernel) * Matern52Kernel",
            #     val=(Matern32Kernel() + ConstantKernel()) * Matern52Kernel(),
            # ),

            # Summed kernels.
            (
                name="sum-Matern12Kernel-Matern32Kernel",
                val=1.5 * Matern12Kernel() ∘ ScaleTransform(0.1) +
                    0.3 * Matern32Kernel() ∘ ScaleTransform(1.1),
            ),
            (
                name="sum-Matern32Kernel-Matern52Kernel-ConstantKernel",
                val=2.0 * Matern32Kernel() +
                    0.5 * Matern52Kernel() +
                    1.0 * ConstantKernel(),
            ),
        )

        # Construct a Gauss-Markov model with either dense storage or static storage.
        storages = (
            (name="dense storage Float64", val=ArrayStorage(Float64)),
            (name="static storage Float64", val=SArrayStorage(Float64)),
        )

        # Either regular spacing or irregular spacing in time.
        ts = (
            (name="irregular spacing", val=collect(RegularSpacing(0.0, 0.3, N))),
            (name="regular spacing", val=RegularSpacing(0.0, 0.3, N)),
        )

        σ²s = (
            (name="homoscedastic noise", val=(0.1,)),
            (name="heteroscedastic noise", val=(rand(rng, N) .+ 1e-1, )),
        )

        means = (
            (name="Zero Mean", val=ZeroMean()),
            (name="Const Mean", val=ConstMean(3.0)),
            (name="Custom Mean", val=CustomMean(x -> 2x)),
        )

        @testset "$(kernel.name), $(m.name), $(storage.name), $(t.name), $(σ².name)" for
            kernel in kernels,
            m in means,
            storage in storages,
            t in ts,
            σ² in σ²s

            println("$(kernel.name), $(storage.name), $(m.name), $(t.name), $(σ².name)")

            if kernel.val isa TemporalGPs.ApproxPeriodicKernel &&
                storage.val isa SArrayStorage
                @info "skipping because ApproxPeriodicKernel not compatible with SArrayStorage"
                continue
            end

            # Construct Gauss-Markov model.
            f_naive = GP(m.val, kernel.val)
            fx_naive = f_naive(collect(t.val), σ².val...)

            f = to_sde(f_naive, storage.val)
            fx = f(t.val, σ².val...)
            model = build_lgssm(fx)

            # is_of_storage_type(fx, storage.val)
            validate_dims(model)

            y = rand(rng, fx)

            @testset "prior" begin
                @test mean(fx) ≈ mean(fx_naive)
                @test cov(fx) ≈ cov(fx_naive)
                @test mean.(marginals(fx)) ≈ mean.(marginals(fx_naive))
                @test std.(marginals(fx)) ≈ std.(marginals(fx_naive))
                m_and_v = mean_and_var(fx)
                @test first(m_and_v) ≈ mean(fx)
                @test last(m_and_v) ≈ var(fx)
                @test logpdf(fx, y) ≈ logpdf(fx_naive, y)
            end

            test_rule(
                rng, _logpdf_tester, f_naive, y, storage.val, σ².val, t.val;
                is_primitive=false,
            )
        end
    end
end
