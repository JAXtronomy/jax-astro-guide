# Assorted JAX Tips

Below are some accumulated tips and tricks to get your JAX code running quickly 
and avoid pitfalls that might not appear obviously in documentation.  This list 
is in no way comprehensive or official, and we welcome suggestions for refinement 
or for other tips.

## Speed and Performance
 - Converting a list to a `jnp.array` is slow, see [https://github.com/google/jax/issues/10662](https://github.com/google/jax/issues/10662)
 - Reshaping arrays with `.reshape` is slow, see [https://github.com/google/jax/discussions/11013](https://github.com/google/jax/discussions/11013)
 - `.set` is slow in some cases.  Only use it if you don't know the index at with the element will be set, otherwise you should be able to use `vmap`/`vectorize` instead.
 - `map_coordinates` seems to be slow, though at the moment good alternatives for 2D interpolation are not known by this author.

## Best Practices
 - Many JAX functions are already built to take vector inputs.  `jnp.interp`, for example, can interpolate over vector inputs--this behavior is documented, but new users often miss it!
 - `jit` the `vmap`'d version of function, rather than `vmap`ing a `jit`ed function (see [https://github.com/jax-ml/jax/issues/6312](https://github.com/jax-ml/jax/issues/6312)).  The latter introduces some minor overhead which may be noticeable for faster computations.
 - Use `jax.pure_callback` to call a python function in JAX and still get to use jit.  Unfortunately for autodifferentiation you will need to specify a `custom_jvp` or `custom_vjp` if you use this.

### Equinox and Diffrax
If you're a user of [equinox](https://docs.kidger.site/equinox/) and/or [diffrax](https://docs.kidger.site/diffrax/) (both are common in scientific computing applications of JAX), there are a couple of extra best-practice tips that may be helpful for you:
 - If you use equinox modules and `vmap` is suspiciously slow, try using `eqx.filter_vmap` instead.
 - Be wary of `eqx.filter_jit`.  If you're using equinox modules and seeing lots of cache misses when profiling, `eqx.filter_jit` could be to blame.  At least replacing `filter_jit` with `jax.jit` could make profiler output easier to interpret.
 - Often it's better to reduce the number of `diffrax` calls, i.e., try not to solve different differential equations across different regions if you can help is.  If the equation you're trying to solve is significantly more complicated in one regime vs another then you may find multiple diffrax calls could be helpful.

## Control Flows
 - `jnp.where` is a great way to replace `if`/`else` statements in your code, though you can still avoid conditionals in other ways.  For example, Python automatically converts the boolean array to values (`1` for `True`, `0` for `False`), which JAX works with as well.  You can therefore write lines like  `x = value * jnp.logical_or(cond1,cond2)`.
 - If you're doing logic on array indices that you wanted to ravel/unravel, you can just use the numpy versions, **as long as you're operating on static objects**.  Something like
   ```
   np.unravel_index(np.arange(x.shape[0]), (5,5))
   if x.shape[0]<5:
	     ...
   ```
   is perfectly fine in `jit`ed code.

## Debugging and Profiling
 - JAX now has many ways to print inside a `jit`ed function!  `jax.debug.print` is a great for simple inspection (but does not work with `grad`).  Calling, e.g.
   ```
    t = ...
    def printer(x):
	      print(x)
    jax.debug.callback(printer, t)
   ```
   works inside both `jit` and `grad`.  See [https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html](https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html) for more.
 - Always use `.block_until_ready()` or do something with the result of your calculation if you're doing a timing test, otherwise you might just get the dispatch time (see [asynchronous dispatch](https://jax.readthedocs.io/en/latest/async_dispatch.html)).
 - Don't be afraid of functions like `jax.lower(fun)(x)` or `jax.make_jaxpr(fun)(x)` to see what the compilers are doing to the function `fun` with input `x`.  The former is best for smaller, more contained functions.
