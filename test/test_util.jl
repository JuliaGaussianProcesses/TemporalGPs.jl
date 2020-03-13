create_psd_matrix(A::AbstractMatrix) = A * A' + I

function create_psd_stable_matrix(A::AbstractMatrix)
    B = create_psd_matrix(A)
    λ, U = eigen(B)
    λ .+= 1
    λ ./= (maximum(λ) + 1e-1 * maximum(λ))
    return Matrix(Symmetric(U * Diagonal(λ) * U'))
end



# Make FiniteDifferences work with some of the types in this package. Shame this isn't
# automated...

import FiniteDifferences: to_vec

# Make FiniteDifferences work for any user-defined type that has fields. This assumes that
# the type passed in admits a constructor that just takes all of the fields and produces a
# new object. As such, this won't work in the general case, but should work for all of the
# stuff in this package.
function to_vec(x::T) where {T}
    isempty(fieldnames(T)) && throw(error("Expected some fields. None found."))
    vecs_and_backs = map(name->to_vec(getfield(x, name)), fieldnames(T))
    vecs, backs = first.(vecs_and_backs), last.(vecs_and_backs)
    x_vec, back = to_vec(vecs)
    function struct_to_vec(x′_vec)
        vecs′ = back(x′_vec)
        return T(map((back, vec)->back(vec), backs, vecs′)...)
    end
    return x_vec, struct_to_vec
end

function to_vec(x::Base.ReinterpretArray{<:Real})
    return to_vec(collect(x))
end

function to_vec(x::T) where {T<:NamedTuple}
    isempty(fieldnames(T)) && throw(error("Expected some fields. None found."))
    vecs_and_backs = map(name->to_vec(getfield(x, name)), fieldnames(T))
    vecs, backs = first.(vecs_and_backs), last.(vecs_and_backs)
    x_vec, back = to_vec(vecs)
    function namedtuple_to_vec(x′_vec)
        vecs′ = back(x′_vec)
        x′s = map((back, vec)->back(vec), backs, vecs′)
        return (; zip(fieldnames(T), x′s)...)
    end
    return x_vec, namedtuple_to_vec
end

# Ensure that to_vec works for the types that we care about in this package.
@testset "custom FiniteDifferences stuff" begin
    @testset "NamedTuple" begin
        a, b = 5.0, randn(2)
        t = (a=a, b=b)
        nt_vec, back = to_vec(t)
        @test nt_vec isa Vector
        @test back(nt_vec) == t
    end
end

# My version of isapprox
function fd_isapprox(x_ad::Nothing, x_fd, rtol, atol)
    return fd_isapprox(x_fd, zero(x_fd), rtol, atol)
end
function fd_isapprox(x_ad::AbstractArray, x_fd::AbstractArray, rtol, atol)
    return all(fd_isapprox.(x_ad, x_fd, rtol, atol))
end
function fd_isapprox(x_ad::Real, x_fd::Real, rtol, atol)
    return isapprox(x_ad, x_fd; rtol=rtol, atol=atol)
end
function fd_isapprox(x_ad::NamedTuple, x_fd, rtol, atol)
    f = (x_ad, x_fd)->fd_isapprox(x_ad, x_fd, rtol, atol)
    return all([f(getfield(x_ad, key), getfield(x_fd, key)) for key in keys(x_ad)])
end
function fd_isapprox(x_ad::Tuple, x_fd::Tuple, rtol, atol)
    return all(map((x, x′)->fd_isapprox(x, x′, rtol, atol), x_ad, x_fd))
end
function fd_isapprox(x_ad::Dict, x_fd::Dict, rtol, atol)
    return all([fd_isapprox(get(()->nothing, x_ad, key), x_fd[key], rtol, atol) for
        key in keys(x_fd)])
end

function adjoint_test(
    f, ȳ, x...;
    rtol=1e-9,
    atol=1e-9,
    fdm=FiniteDifferences.Central(5, 1),
    print_results=false,
    test=true,
)
    # Compute forwards-pass and j′vp.
    y, back = Zygote.pullback(f, x...)
    adj_ad = back(ȳ)
    adj_fd = j′vp(fdm, f, ȳ, x...)

    # Check that forwards-pass agrees with plain forwards-pass.
    test && @test fd_isapprox(y, f(x...), rtol, atol)

    # Check that ad and fd adjoints (approximately) agree.
    print_results && print_adjoints(adj_ad, adj_fd, rtol, atol)
    test && @test fd_isapprox(adj_ad, adj_fd, rtol, atol)

    return adj_ad, adj_fd
end

function print_adjoints(adjoint_ad, adjoint_fd, rtol, atol)
    @show typeof(adjoint_ad), typeof(adjoint_fd)
    adjoint_ad, adjoint_fd = to_vec(adjoint_ad)[1], to_vec(adjoint_fd)[1]
    println("atol is $atol, rtol is $rtol")
    println("ad, fd, abs, rel")
    abs_err = abs.(adjoint_ad .- adjoint_fd)
    rel_err = abs_err ./ adjoint_ad
    display([adjoint_ad adjoint_fd abs_err rel_err])
    println()
end

using BenchmarkTools

function benchmark_adjoint(f, ȳ, args...; disp=true)
    disp && println("primal")
    primal = @benchmark $f($args...)
    if disp
        display(primal)
        println()
    end

    disp && println("pullback generation")
    forward_pass = @benchmark Zygote.pullback($f, $args...)
    if disp
        display(forward_pass)
        println()
    end

    y, back = Zygote.pullback(f, args...)

    disp && println("pullback evaluation")
    reverse_pass = @benchmark $back($ȳ)
    if disp
        display(reverse_pass)
        println()
    end

    return primal, forward_pass, reverse_pass
end

function adjoint_allocs(f, ȳ, args...; disp=true)
    primal = @allocated f(args...)
    forward_pass = @allocated Zygote.pullback(f, args...)

    y, back = Zygote.pullback(f, args...)
    reverse_pass = @allocated back(ȳ)

    return primal, forward_pass, reverse_pass
end

"""
    standard_lgssm_tests(rng::AbstractRNG, ssm, y::AbstractVector)

Basic consistency tests that any ssm should be able to satisfy. The purpose of these tests
is not to ensure correctness of any given implementation, only to ensure that it is self-
consistent and implements the required interface.
"""
function standard_lgssm_tests(rng::AbstractRNG, ssm, y::AbstractVector)
    @test last(filter(ssm, y)) ≈ logpdf(ssm, y)
    @test last(smooth(ssm, y)) ≈ logpdf(ssm, y)
end
