# BasisExpansions.jl

`BasisExpansions.jl` provides `BasisExpansion`, an `AbstractArray` wrapper for
representing a matrix-basis expansion with trainable coefficients. Optional
column-scaling and QR preconditioning are available, along with real and
complex polynomial constructors.

```julia
using BasisExpansions

x = range(0, 1; length = 20)
model = polynomial_representation(collect(x), 3)
values = model()
coefs = get_coefficients(model)
```
