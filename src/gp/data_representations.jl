"""
    get_times(x::AbstractVector{<:Real})

Get the times associated with the inputs.
"""
get_times(x::AbstractVector{<:Real}) = x

"""
    inputs_to_time_form(x::AbstractVector)

Get the time-form representation of a vector of inputs.
Outputs an `AbstractVector` whose length is `length(get_times(x))`,
and whose elements comprise all of those associated with each time point.

For single-output time series problems, `x` is usually returned as-is, however, for
multi-output and spatio-temporal problems there are usually multiple inputs associated with
each point in time.
"""
inputs_to_time_form(x::AbstractVector{<:Real}) = Fill(nothing, length(x))

"""
    merge_inputs(x1::AbstractVector, x2::AbstractVector)

Merge two collections of inputs `x1` and `x2` if possible.
A notable example of inputs which cannot be merged are two rectilinear grids whose
spatial components differ.
"""
merge_inputs(x1::AbstractVector{<:Real}, x2::AbstractVector{<:Real}) = vcat(x1, x2) 

"""
    sort_in_time(x::AbstractVector)

Sort the input `x` temporally.
This is different from sorting the input in general, but agrees for real-valued inputs.
Returns both the sorting indices and the sorted inputs.
"""
function sort_in_time(x::AbstractVector{<:Real})
    idx = sortperm(x)
    return idx, x[idx]
end

"""
    observations_to_time_form(x::AbstractVector, y::AbstractVector{<:Real})

Get the time-form representation of a vector of observations `y` of length
`length(time_points(x))`, which is useful for state-space models.
The precise arrangements of the elements of `y` in the output depends on `x`.

For example, single-output time-series problems typically just return `y`, but multi-output
and spatio-temporal problems typically have multiple elements of `y` associated with a
single element of `x`.
"""
function observations_to_time_form(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Union{Missing, Real}},
)
    return y
end

"""
    noise_var_to_time_form(x::AbstractVector, S::AbstractMatrix{<:Real})

Get the time-form representation of an `AbstractMatrix` of observation noise variances `S`,
based on the `AbstractVector` of inputs `x`.

For example, single-output time problems will typically require `S` to be `Diagonal`, and
will just return its diagonal.
Multi-output and spatio-temporal problems can potentially have a block-diagonal `S`, as
it's typically only necessary to require that the observation variance at each point in time
is independent of other times.
"""
noise_var_to_time_form(x::AbstractVector{<:Real}, S::Diagonal{<:Real}) = diag(S)


"""
    destructure(x::AbstractVector, ys::AbstractVector)

Construct a flattened representation of `ys` in accordance with the structure in `x`.
"""
destructure(x::AbstractVector{<:Real}, ys::AbstractVector) = ys
