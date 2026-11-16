# BasisExpansions.jl

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://ferreolS.github.io/BasisExpansions.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://ferreolS.github.io/BasisExpansions.jl/dev)
[![Test workflow status](https://github.com/ferreolS/BasisExpansions.jl/actions/workflows/Test.yml/badge.svg?branch=master)](https://github.com/ferreolS/BasisExpansions.jl/actions/workflows/Test.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/ferreolS/BasisExpansions.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ferreolS/BasisExpansions.jl)
[![Docs workflow Status](https://github.com/ferreolS/BasisExpansions.jl/actions/workflows/Docs.yml/badge.svg?branch=master)](https://github.com/ferreolS/BasisExpansions.jl/actions/workflows/Docs.yml?query=branch%3Amaster)


`BasisExpansions.jl` provides `BasisExpansion`, an `AbstractArray` wrapper for
representing a matrix-basis expansion with trainable coefficients. Optional
column-scaling and QR preconditioning are available, along with real and
complex polynomial constructors.

```julia
using BasisExpansions

x = range(0, 1; length = 20)
model = BasisExpansions.polynomial_representation(x, 3)
values = collect(model)
coefs = BasisExpansions.get_coefficients(model)
```
