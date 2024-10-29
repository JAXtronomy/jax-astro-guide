# Assorted JAX Tips

Below are some accumulated tips and tricks to get your JAX code running quickly 
and avoid pitfalls that might not appear in documentation.  This list is in no 
way comprehensive or official, and we welcome suggestions for refinement or 
for other tips.

## Speed and Performance
 - Converting a list to a `jnp.array` is slow, see [https://github.com/google/jax/issues/10662](https://github.com/google/jax/issues/10662)
 - Reshaping arrays with `.reshape` is slow, see [https://github.com/google/jax/discussions/11013](https://github.com/google/jax/discussions/11013)
 - `.set` is slow in some cases.  Only use it if you don't know the index at with the element will be set, otherwise you should be able to use `vmap`/`vectorize` instead.
 - `map_coordinates` seems to be slow, though at the moment good alternatives for 2D interpolation are not known by this author.

## Best Practices
 - Many JAX functions are already built to take vector inputs.  `jnp.interp`, for example, can interpolate over vector inputs--this behavior is documented, but new users often miss it!
 - `jit` the `vmap`'d version of function, rather than `vmap`ing a `jit`ed function (see [https://github.com/jax-ml/jax/issues/6312](https://github.com/jax-ml/jax/issues/6312)).  The latter introduces some minor overhead which may be noticeable for faster computations.
 - Feel free to add the `jit` decorator to inner functions, rather than just the top-level function.  The only drawback is that this might prevent the compiler from finding a more efficient fusion by combining pieces of multiple inner functions in some clever way, but this seems like an edge case.  `jit`ting an inner function means it only gets executed once and then the code calls the cached trace on subsequent evaluations, which can be really helpful if you're calling an inner function multiple times in some outer routine.

### Equinox and Diffrax
If you're a user of [equinox](https://docs.kidger.site/equinox/) and/or [diffrax](https://docs.kidger.site/diffrax/) (both are common in scientific computing applications of JAX), there are a couple of extra best-practice tips that may be helpful for you:
 - If you use equinox modules and `vmap` is suspiciously slow, try using `eqx.filter_vmap` instead.
 - Be wary of `eqx.filter_jit`.  If you're using equinox modules and seeing lots of cache misses, `eqx.filter_jit` could be to blame.  At least replacing `filter_jit` with `jax.jit` could make profiler output easier to interpret.
 - Often it's better to reduce the number of `diffrax` calls, i.e., try not to solve different differential equations across different regions if you can help is.  If the equation you're trying to solve is significantly more complicated in one regime vs another then you may find multiple diffrax calls could be helpful.

## Control Flows

## Debugging and Profiling
