using TemporalGPs: GaussMarkovModel

println("to_gauss_markov:")
@testset "to_gauss_markov" begin

    @testset "blk_diag" begin
        adjoint_test(TemporalGPs.blk_diag, randn(5, 5), randn(2, 2), randn(3, 3))
    end

    @testset "BaseKernel parameter types" begin

        storages = (
            (name="dense storage Float64", val=ArrayStorage(Float64)),
            (name="static storage Float64", val=SArrayStorage(Float64)),
            (name="dense storage Float32", val=ArrayStorage(Float32)),
            (name="static storage Float32", val=SArrayStorage(Float32)),
        )

        kernels = [Matern12(), Matern32(), Matern52()]

        @testset "$kernel, $(storage.name)" for kernel in kernels, storage in storages
            F, q, H = TemporalGPs.to_sde(kernel, storage.val)
            validate_types(F, storage.val)
            validate_types(q, storage.val)
            validate_types(H, storage.val)

            x = TemporalGPs.stationary_distribution(kernel, storage.val)
            validate_types(x, storage.val)
        end
    end

    @testset "GaussMarkovModel from kernel correctness" begin
        rng = MersenneTwister(123456)
        N = 5

        kernels_info = vcat(

            # Base kernels.
            map([Matern12, Matern32, Matern52]) do kernel
                (name="base-$kernel", ctor=()->kernel(), θ=())
            end,

            # Scaled kernels.
            map([1e-1, 1.0, 10.0, 100.0]) do σ²
                (name="scaled-σ²=$σ²", ctor=(σ->σ^2 * Matern32()), θ=(sqrt(σ²),))
            end,

            # Stretched kernels.
            map([1e-4, 0.1, 1.0, 10.0, 100.0]) do λ
                (name="stretched-λ=$λ", ctor=(λ->stretch(Matern32(), λ)), θ=(λ,))
            end,

            # Summed kernels.
            (
                name="sum-Matern12-Matern32",
                ctor=(λl, λr, σl, σr)->begin
                    k_l = σl^2 * stretch(Matern12(), λl)
                    k_r = σr^2 * stretch(Matern32(), λr)
                    return k_l + k_r
                end,
                θ=(0.1, 1.1, 1.5, 0.3),
            ),
        )

        # construct a Gauss-Markov model with either dense storage or static storage.
        storages = (
            (name="dense storage Float64", val=ArrayStorage(Float64)),
            (name="static storage Float64", val=SArrayStorage(Float64)),
            (name="dense storage Float32", val=ArrayStorage(Float32)),
            (name="static storage Float32", val=SArrayStorage(Float32)),
        )

        # Either regular spacing or irregular spacing in time.
        ts = (
            (name="irregular spacing", val=sort(rand(rng, N))),
            (name="regular spacing", val=range(0.0; step=0.3, length=N)),
        )

        @testset "$(kernel_info.name), $(storage.name), $(t.name)" for
                kernel_info in kernels_info,
                storage in storages,
                t in ts

            # Convert all parameters to appropriate element type.
            θ = map(eltype(storage.val), kernel_info.θ)

            # Construct Gauss-Markov model.
            k = kernel_info.ctor(θ...)
            ft = GaussMarkovModel(k, t.val, storage.val)

            validate_types(ft, storage.val)
            validate_dims(ft)

            # Check that the covariances agree, only for high-ish precision.
            if eltype(storage.val) == Float64
                @test cov(ft) ≈ pw(k, t.val, t.val)

                # Ensure that it's possible to backprop through construction.
                if length(kernel_info.θ) > 0 && t.val isa Vector
                    N = length(ft)
                    Dobs = size(first(ft.H), 1)
                    Dlat = size(first(ft.H), 2)
                    ΔA = map(_ -> randn(rng, Dlat, Dlat), 1:N)
                    ΔQ = map(_ -> random_nice_psd_matrix(rng, Dlat, storage.val), 1:N)
                    ΔH = map(_ -> randn(rng, Dobs, Dlat), 1:N)
                    ΔH_sum = randn(rng, Dobs, Dlat)
                    Δm = randn(rng, size(ft.x0.m))
                    ΔP = random_nice_psd_matrix(rng, Dlat, storage.val)

                    adjoint_test(
                        (θ) -> begin
                            k = kernel_info.ctor(θ...)
                            ft = GaussMarkovModel(k, t.val, storage.val)
                            return (ft.A, ft.Q, sum(ft.H), ft.x0.m, ft.x0.P)
                        end,
                        (ΔA, ΔQ, ΔH_sum, Δm, ΔP),
                        θ;
                    )
                end
            end
        end
    end

    @testset "static perf" begin
        k = Matern32()
        t = range(0.0; step=0.3, length=11)
        @test (@ballocated TemporalGPs.GaussMarkovModel($k, $t, SArrayStorage(Float64))) == 0
    end
end
