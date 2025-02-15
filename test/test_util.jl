using TemporalGPs:
    AbstractLGSSM,
    ElementOfLGSSM,
    Gaussian,
    Forward,
    Reverse,
    GaussMarkovModel,
    LGSSM,
    ordering,
    SmallOutputLGC,
    posterior_and_lml,
    predict,
    conditional_rand,
    AbstractLGC,
    dim_out,
    dim_in,
    _filter,
    x0,
    scan_emit,
    ε_randn

function test_interface(
    rng::AbstractRNG,
    conditional::AbstractLGC,
    x::Gaussian;
    check_inferred=true,
    check_adjoints=true,
    check_allocs=true,
)
    x_val = rand(rng, x)
    y = conditional_rand(rng, conditional, x_val)
    perf_flag = check_allocs ? :allocs : :none

    is_primitive = false
    @testset "rand" begin
        @test length(y) == dim_out(conditional)
        args = (TemporalGPs.ε_randn(rng, conditional), conditional, x_val)
        check_inferred && @test_opt target_modules = [TemporalGPs] conditional_rand(args...)
        check_adjoints && test_rule(rng, conditional_rand, args...; perf_flag, is_primitive)
    end

    @testset "predict" begin
        @test predict(x, conditional) isa Gaussian
        check_inferred && @test_opt target_modules = [TemporalGPs] predict(x, conditional)
        check_adjoints && test_rule(rng, predict, x, conditional; perf_flag, is_primitive)
    end

    conditional isa ScalarOutputLGC || @testset "predict_marginals" begin
        @test predict_marginals(x, conditional) isa Gaussian
        pred = predict(x, conditional)
        pred_marg = predict_marginals(x, conditional)
        @test mean(pred_marg) ≈ mean(pred)
        @test diag(cov(pred_marg)) ≈ diag(cov(pred))
        @test cov(pred_marg) isa Diagonal
    end

    @testset "posterior_and_lml" begin
        args = (x, conditional, y)
        @test posterior_and_lml(args...) isa Tuple{Gaussian,Real}
        check_inferred &&
            @test_opt target_modules = [TemporalGPs] posterior_and_lml(args...)
        check_adjoints &&
            test_rule(rng, posterior_and_lml, args...; perf_flag, is_primitive)
    end
end

"""
    test_interface(
        rng::AbstractRNG, ssm::AbstractLGSSM;
        check_inferred=true, check_adjoints=true, check_allocs=true,
    )

Basic consistency tests that any LGSSM should be able to satisfy. The purpose of these tests
is not to ensure correctness of any given implementation, only to ensure that it is self-
consistent and implements the required interface.
"""
function test_interface(
    rng::AbstractRNG,
    ssm::AbstractLGSSM;
    check_inferred=true,
    check_adjoints=true,
    check_allocs=true,
)
    perf_flag = check_allocs ? :allocs : :none
    y_no_missing = rand(rng, ssm)
    @testset "LGSSM interface" begin
        @testset "rand" begin
            @test is_of_storage_type(y_no_missing[1], storage_type(ssm))
            @test y_no_missing isa AbstractVector
            @test length(y_no_missing) == length(ssm)
            check_inferred && @test_opt target_modules = [TemporalGPs] rand(rng, ssm)
            rng = MersenneTwister(123456)
            if check_adjoints
                test_rule(
                    rng, rand, rng, ssm; perf_flag, interface_only=true, is_primitive=false
                )
            end
        end

        @testset "basics" begin
            @test_opt target_modules = [TemporalGPs] storage_type(ssm)
            @test length(ssm) == length(y_no_missing)
        end

        @testset "marginals" begin
            xs = marginals(ssm)
            @test is_of_storage_type(xs, storage_type(ssm))
            @test xs isa AbstractVector{<:Gaussian}
            @test length(xs) == length(ssm)
            check_inferred && @test_opt target_modules = [TemporalGPs] marginals(ssm)
            if check_adjoints
                test_rule(
                    rng,
                    scan_emit,
                    step_marginals,
                    ssm,
                    x0(ssm),
                    eachindex(ssm);
                    perf_flag,
                    is_primitive=false,
                    interface_only=true,
                )
            end
        end

        @testset "$(data.name)" for data in [(name="no-missings", y=y_no_missing)]
            _check_inferred = data.name == "with-missings" ? false : check_inferred

            y = data.y
            @testset "logpdf" begin
                lml = logpdf(ssm, y)
                @test lml isa Real
                @test is_of_storage_type(lml, storage_type(ssm))
                _check_inferred && @test_opt target_modules = [TemporalGPs] logpdf(ssm, y)
                if check_adjoints
                    test_rule(
                        rng,
                        scan_emit,
                        step_logpdf,
                        zip(ssm, y),
                        x0(ssm),
                        eachindex(ssm);
                        perf_flag,
                        is_primitive=false,
                        interface_only=true,
                    )
                end
            end
            @testset "_filter" begin
                xs = _filter(ssm, y)
                @test is_of_storage_type(xs, storage_type(ssm))
                @test xs isa AbstractVector{<:Gaussian}
                @test length(xs) == length(ssm)
                _check_inferred && @test_opt target_modules = [TemporalGPs] _filter(ssm, y)
                if check_adjoints
                    test_rule(
                        rng,
                        scan_emit,
                        step_filter,
                        zip(ssm, y),
                        x0(ssm),
                        eachindex(ssm);
                        perf_flag,
                        is_primitive=false,
                        interface_only=true,
                    )
                end
            end
            @testset "posterior" begin
                posterior_ssm = posterior(ssm, y)
                @test length(posterior_ssm) == length(ssm)
                @test ordering(posterior_ssm) != ordering(ssm)
                _check_inferred &&
                    @test_opt target_modules = [TemporalGPs] posterior(ssm, y)
                if check_adjoints
                    test_rule(
                        rng,
                        posterior,
                        ssm,
                        y;
                        perf_flag,
                        is_primitive=false,
                        interface_only=true,
                    )
                end
            end
        end
    end
end
