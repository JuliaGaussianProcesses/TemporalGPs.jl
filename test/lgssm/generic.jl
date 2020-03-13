using TemporalGPs: smooth, StaticStorage, DenseStorage, _predict,
    update_decorrelate, update_decorrelate_pullback, step_decorrelate,
    step_decorrelate_pullback, update_correlate, update_correlate_pullback,
    step_correlate, dim_obs, dim_latent, cholesky_pullback, logdet_pullback,
    ScalarLGSSM, LGSSM
using Stheno: GP, GPC
using Zygote, StaticArrays



function to_vec(x::LGSSM)
    vecs_and_backs = map(name->to_vec(getfield(x, name)), [:A, :b, :Q, :H, :c, :Σ, :x₀])
    vecs, backs = first.(vecs_and_backs), last.(vecs_and_backs)
    x_vec, back = to_vec(vecs)
    function lgssm_to_vec(x′_vec)
        vecs′ = back(x′_vec)
        return LGSSM(map((back, v)->back(v), backs, vecs′)...)
    end
    return x_vec, lgssm_to_vec
end

@testset "to_vec(::LGSSM)" begin
    N = 11
    A, Q, H, R = randn(2, 2), randn(2, 2), randn(3, 2), randn(3, 3)
    bs = Fill(zeros(2), N)
    cs = Fill(zeros(3), N)
    x = TemporalGPs.Gaussian(randn(3), randn(3, 3))

    model = LGSSM(Fill(A, N), bs, Fill(Q, N), Fill(H, N), cs, Fill(R, N), x)
    model_vec, back = to_vec(model)
    @test back(model_vec) == model
end


@testset "generic" begin
    @testset "Cholesky pullbacks - $N" for N in [1, 2, 3]

        rng = MersenneTwister(123456)

        # Do dense stuff
        S_ = randn(rng, N, N)
        S = S_ * S_' + I
        C = cholesky(S)
        Ss = SMatrix{N, N}(S)
        Cs = cholesky(Ss)

        @testset "cholesky" begin
            C_fwd, pb = cholesky_pullback(Symmetric(S))
            Cs_fwd, pbs = cholesky_pullback(Symmetric(Ss))

            ΔC = randn(rng, N, N)
            ΔCs = SMatrix{N, N}(ΔC)

            @test C.U ≈ Cs.U
            @test Cs.U ≈ Cs_fwd.U
            @test first(pb((factors=ΔC,))) ≈ first(pbs((factors=ΔCs,)))

            @test (@allocated cholesky(Symmetric(Ss))) == 0
            @test (@allocated cholesky_pullback(Symmetric(Ss))) == 0
            @test (@allocated pbs((factors=ΔCs,))) == 0
        end
        @testset "logdet" begin
            @test logdet(Cs) ≈ logdet(C)
            C_fwd, pb = logdet_pullback(C)
            Cs_fwd, pbs = logdet_pullback(Cs)

            @test logdet(Cs) ≈ Cs_fwd

            Δ = randn(rng)
            @test first(pb(Δ)).factors ≈ first(pbs(Δ)).factors

            @test (@allocated logdet(Cs)) == 0
            @test (@allocated logdet_pullback(Cs)) == 0
            @test (@allocated pbs(Δ)) == 0
        end
    end

    @testset "static perf" begin
        @testset "ssm perf" begin
            ft = GP(Matern32(), GPC())(range(0.0; step=0.3, length=11), 0.3)
            @test @allocated(ssm(ft, StaticStorage())) == 0
        end
    end

    @testset "GP correctness" begin

        N = 11
        rng = MersenneTwister(123546)

        # Specify test cases.
        kernels_info = vcat(

            # (name="base-Matern32", ctor=()->Matern32(), θ=()),

            # Base kernels.
            map([Matern12, Matern32, Matern52]) do kernel
                (name="base-$kernel", ctor=()->kernel(), θ=())
            end,

            # Scaled kernels.
            map([1e-1, 1.0, 10.0, 100.0]) do σ²
                (name="scaled-σ²=$σ²", ctor=(σ²->σ² * Matern32()), θ=(σ²,))
            end,

            # Stretched kernels.
            map([1e-4, 0.1, 1.0, 10.0, 100.0]) do λ
                (name="stretched-λ=$λ", ctor=(λ->stretch(Matern32(), λ)), θ=(λ,))
            end,
        )

        # construct an LGSSM with either dense storage or static storage.
        storages = (
            (name="dense storage", val=DenseStorage()),
            # (name="static storage", val=StaticStorage()),
        )

        # Either regular spacing or irregular spacing in time.
        ts = (
            (name="regular spacing", val=range(0.0; step=0.3, length=N)),
            # (name="irregular spacing", val=sort(rand(rng, N))),
        )

        σ²s = (
            (name="homoscedastic noise", val=0.1),
            # (name="heteroscedastic noise", val=1 ./ (1 .+ exp.(.-randn(rng, N))) .+ 1e-1),
        )

        @testset "$(kernel_info.name), $(storage.name), $(t.name), $(σ².name)" for
            kernel_info in kernels_info,
            storage in storages,
            t in ts,
            σ² in σ²s

            # Construct kernel for use in basic tests.
            θ = kernel_info.θ
            kernel = kernel_info.ctor(θ...)

            # Construct naively-implemented model and generate a sample.
            f = GP(kernel, GPC())
            ft = f(t.val, σ².val)
            y = rand(MersenneTwister(123456), ft)

            # Construct ssm.
            model = ssm(ft, storage.val)

            # Stheno and SSM produce the same samples given the same seed.
            @test y ≈ rand(MersenneTwister(123456), model)

            # Stheno and SSM logpdfs agree.
            @test logpdf(ft, y) ≈ logpdf(model, y)

            rng = MersenneTwister(123456)
            Dlat = dim_latent(model)

            adjoint_test(
                (σ²_n, y, θ...)->begin
                    f = GP(kernel_info.ctor(θ...), GPC())
                    ft = f(t.val, σ²_n)
                    a = ssm(ft, storage.val)
                    b = filter(a, y)
                    return first(b)
                end,
                randn(rng),
                σ².val, y, θ...; atol=1e-6, rtol=1e-6,
            )

            adjoint_test(
                (σ²_n, y, θ...)->begin
                    f = GP(kernel_info.ctor(θ...), GPC())
                    return logpdf(ssm(f(t.val, σ²_n), storage.val), y)
                end,
                randn(rng),
                σ².val, y, θ...; atol=1e-6, rtol=1e-6,
            )

            adjoint_test(
                (σ²_n, y, θ...)->begin
                    f = GP(kernel_info.ctor(θ...), GPC())
                    return whiten(ssm(f(t.val, σ²_n), storage.val), y)
                end,
                randn(rng, length(y)),
                σ².val, y, θ...,
            )

            _, y_smooth, _ = smooth(ssm(f(t.val, σ².val), storage.val), y)

            # Check posterior marginals
            m_ssm = [first(y.m) for y in y_smooth]
            σ²_ssm = [first(y.P) for y in y_smooth]

            f′ = f | (f(t.val, σ².val) ← y)
            f′_marginals = marginals(f′(t.val))
            m_exact = mean.(f′_marginals)
            σ²_exact = std.(f′_marginals).^2

            @test m_ssm ≈ m_exact
            @test σ²_ssm ≈ σ²_exact
        end
    end

    @testset "static filtering gradients" begin
        rng = MersenneTwister(123456)
        t = range(0.1; step=0.11, length=10_000)
        f = GP(Matern32(), GPC())
        σ²_n = 0.54
        model = ssm(f(t, 0.54), StaticStorage()).model
        x = model.x₀
        D = length(x.m)
        m, P = x.m, x.P
        A, Q = first(model.A), first(model.Q)
        h, σ² = first(model.H), first(model.Σ)
        b = first(model.b)
        c = first(model.c)

        @testset "_predict" begin

            # Pullback is correct.
            Δx′ = (
                SVector{D}(randn(rng, D)),
                SMatrix{D, D}(randn(rng, D, D)),
            )
            adjoint_test(_predict, Δx′, m, P, A, Q)

            # Nothing is heap-allocated.
            primal, fwd, rvs = benchmark_adjoint(_predict, Δx′, m, P, A, Q; disp=false)
            @test allocs(primal) == 0
            @test allocs(fwd) == 0
            @test allocs(rvs) == 0
        end
        @testset "update_decorrelate & step_decorrelate (scalar output)" begin

            # _update pullback is correct.
            Δx′ = (
                SVector{D}(randn(rng, D)),
                SMatrix{D, D}(randn(rng, D, D)),
                randn(rng),
                SVector{1}(randn(rng)),
            )
            y = SVector{1}(randn(rng))
            adjoint_test(update_decorrelate, Δx′, m, P, h, σ², y)

            # Nothing is heap-allocated by _update.
            @test allocs(@benchmark update_decorrelate($m, $P, $h, $σ², $y)) == 0
            @test allocs(@benchmark update_decorrelate_pullback($m, $P, $h, $σ², $y)) == 0

            _, _update_pb = update_decorrelate_pullback(m, P, h, σ², y)
            @test allocs(@benchmark $_update_pb($Δx′)) == 0

            # Nothing is heap-allocated by iterate_filter. This is harder to check for
            # correctness for practical reasons, so we don't.
            Δx′ = (
                randn(rng),
                SVector{1}(randn(rng)),
                (
                    m=SVector{D}(randn(rng, D)),
                    P=SMatrix{D, D}(randn(rng, D, D)),
                ),
            )
            adjoint_test(step_decorrelate, Δx′, model[1], x, y)
            primal, fwd, rvs = benchmark_adjoint(step_decorrelate, Δx′, model[1], x, y; disp=false)
            @test allocs(primal) == 0
            @test allocs(fwd) == 0
            @test allocs(rvs) == 0
        end
        @testset "update_decorrelate & step_decorrelate (vector output)" begin

            Dout = 1

            # _update pullback is correct.
            Δx′ = (
                SVector{D}(randn(rng, D)),
                SMatrix{D, D}(randn(rng, D, D)),
                randn(rng),
                SVector{Dout}(randn(rng, Dout)),
            )

            H = SMatrix{Dout, D}(randn(rng, Dout, D))
            Σ_ = randn(rng, Dout, Dout)
            Σ = SMatrix{Dout, Dout}(Σ_ * Σ_' + I)
            y = SVector{Dout}(randn(rng, Dout))

            adjoint_test(update_decorrelate, Δx′, m, P, H, Σ, y; atol=1e-8, rtol=1e-8)

            # Nothing is heap-allocated by _update.
            @test allocs(@benchmark update_decorrelate($m, $P, $H, $Σ, $y)) == 0
            @test allocs(@benchmark update_decorrelate_pullback($m, $P, $H, $Σ, $y)) == 0

            _, _update_pb = update_decorrelate_pullback(m, P, H, Σ, y)
            @test allocs(@benchmark $_update_pb($Δx′)) == 0

            # Nothing is heap-allocated by step_decorrelate. This is harder to check for
            # correctness for practical reasons, so we don't.
            Δx′ = (
                randn(rng),
                SVector{Dout}(randn(rng, Dout)),
                (
                    m=SVector{D}(randn(rng, D)),
                    P=SMatrix{D, D}(randn(rng, D, D)),
                ),
            )
            T = length(model)
            bs = Fill(b, T)
            cs = Fill(c, T)
            model = LGSSM(Fill(A, T), bs, Fill(Q, T), Fill(H, T), cs, Fill(Σ, T), x)
            adjoint_test(step_decorrelate, Δx′, model[1], x, y)
            primal, fwd, rvs = benchmark_adjoint(step_decorrelate, Δx′, model[1], x, y; disp=false)
            @test allocs(primal) == 0
            @test allocs(fwd) == 0
            @test allocs(rvs) == 0
        end
        @testset "logpdf performance" begin
            Δlml = randn(rng)
            t = range(0.1; step=0.11, length=1_000_000)
            model = ssm(f(t, 0.54), StaticStorage())
            y = collect(rand(rng, model))

            # Ensure that allocs is roughly independent of length(t).
            primal, fwd, rvs = benchmark_adjoint(logpdf, Δlml, model, y; disp=false)
            @test allocs(primal) < 100
            @test allocs(fwd) < 100
            @test allocs(rvs) < 100
        end
        @testset "update_correlate & step_correlate" begin

            # _update pullback is correct.
            Δx′ = (
                SVector{D}(randn(rng, D)),
                SMatrix{D, D}(randn(rng, D, D)),
                randn(rng),
                SVector{1}(randn(rng)),
            )
            α = SVector{1}(randn(rng))

            adjoint_test(update_correlate, Δx′, m, P, h, σ², α)

            # Nothing is heap-allocated by _update.
            @test allocs(@benchmark update_correlate($m, $P, $h, $σ², $α)) == 0
            @test allocs(@benchmark update_correlate_pullback($m, $P, $h, $σ², $α)) == 0

            _, _update_pb = update_correlate_pullback(m, P, h, σ², α)
            _update_pb(Δx′)
            @test allocs(@benchmark $_update_pb($Δx′)) == 0

            # Nothing is heap-allocated by iterate_filter. This is harder to check for
            # correctness for practical reasons, so we don't.
            Δx′ = (
                randn(rng),
                SVector{1}(randn(rng)),
                (
                    m=SVector{D}(randn(rng, D)),
                    P=SMatrix{D, D}(randn(rng, D, D)),
                ),
            )
            adjoint_test(step_correlate, Δx′, model.model[1], x, α)
            primal, fwd, rvs = benchmark_adjoint(step_correlate, Δx′, model.model[1], x, α; disp=false)
            @test allocs(primal) == 0
            @test allocs(fwd) == 0
            @test allocs(rvs) == 0
        end
        @testset "rand" begin
            t = range(0.1; step=0.11, length=1_000_000)
            Δy = randn(rng, length(t))
            model = ssm(f(t, 0.54), StaticStorage())

            # Ensure that allocs is roughly independent of length(t).
            primal, fwd, rvs = benchmark_adjoint(
                model->rand(MersenneTwister(123456), model), Δy, model;
                disp=false,
            )
            @test allocs(primal) < 100
            @test allocs(fwd) < 100
            @test allocs(rvs) < 100
        end
    end
    @testset "rand (statistical)" begin

        # Specify a GP through time.
        rng = MersenneTwister(123456)
        x = range(0.0; step=0.11, length=5)
        f = GP(Matern32(), GPC())
        fx = f(x, 0.54)

        # Sample from LGSSM equivalent of GP prior at x lots of times.
        model = ssm(fx)
        ys = [first.(rand(rng, fx)) for _ in 1:1_000_000]

        # Check the mean and covariance roughly agree.
        @test all(isapprox.(mean(ys), 0.0; atol=1e-2, rtol=1e-2))
        @test all(isapprox.(cov(ys), cov(fx); atol=1e-2, rtol=1e-2))

        # Check that StaticStorage produces the same numbers.
        y_dense = rand(MersenneTwister(123456), ssm(fx))
        y_static = rand(MersenneTwister(123456), ssm(fx, StaticStorage()))
        @test y_dense ≈ y_static
    end
    @testset "(de)correlate" begin
        rng = MersenneTwister(123456)

        x = range(0.0; step=0.11, length=50)
        f = GP(Matern52(), GPC())
        fx = f(x, 0.1)

        U = cholesky(cov(fx)).U
        y = rand(MersenneTwister(123456), fx)

        α = whiten(ssm(fx), y)
        y′ = TemporalGPs.unwhiten(ssm(fx), α)

        @test y ≈ y′

        v = randn(rng, length(x))
        @test whiten(ssm(fx), v) ≈ U' \ v
        @test TemporalGPs.unwhiten(ssm(fx), v) ≈ U' * v
    end
end
