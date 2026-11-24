"""
    BasisExpansions

Represent values as linear combinations of basis vectors with trainable
coefficients. [`BasisExpansion`](@ref) evaluates a basis matrix and its
coefficients as an `AbstractArray`, with optional column-scaling or QR
preconditioning.

Use [`get_coefficients`](@ref) and [`set_coefficients!`](@ref) to access or
update coefficients in the original basis. [`polynomial_representation`](@ref)
and [`complex_polynomial_representation`](@ref) construct polynomial
expansions evaluated at supplied points.
"""
module BasisExpansions

using LinearAlgebra
using Optimisers
using TypeUtils: get_precision
using Functors: @functor

export BasisExpansion

if VERSION >= v"1.11"
    Core.eval(
        @__MODULE__, Expr(
            :public,
            :polynomial_representation,
            :complex_polynomial_representation,
            :get_coefficients,
            :set_coefficients!

        )
    )
else
    @eval begin
        export polynomial_representation,
            complex_polynomial_representation,
            get_coefficients,
            set_coefficients!
    end
end

"""
    BasisExpansion(basis, coefs; precond = nothing)

Construct an array-like representation of `basis * coefs` with trainable
coefficients. The input `coefs` are interpreted in the original basis.

Set `precond` to `:column_scaling` to scale each basis column to unit
root-mean-square norm, or to `:qr` to use a thin QR factorization of a dense
copy of the basis. QR preconditioning requires full column rank. The default,
`nothing`, leaves the basis unchanged.

The resulting `BasisExpansion` behaves as an `AbstractArray` with the element
type of `basis * coefs` and the row axis of `basis`. Inputs must have one-based
indices, nonempty dimensions, and finite entries. Column scaling also requires
nonzero basis columns. Use [`get_coefficients`](@ref) to retrieve coefficients
in the original basis after optimization.
"""
struct BasisExpansion{T, N, B, C, P} <: AbstractArray{T, N}
    basis::B
    coefs::C
    transform::P

    function BasisExpansion{T, N, B, C, P}(
            basis, coefs, transform
        ) where {T, N, B, C, P}
        return new{T, N, B, C, P}(basis, coefs, transform)
    end

    function BasisExpansion(
            basis::B, coefs::C, transform::P
        ) where {B, C, P}
        T = promote_type(eltype(B), eltype(C))
        N = ndims(C)
        return BasisExpansion{T, N, B, C, P}(basis, coefs, transform)
    end
end

@functor BasisExpansion (coefs,)

Base.size(self::BasisExpansion) = (size(self.basis, 1), size(self.coefs)[2:end]...)
Base.axes(self::BasisExpansion) = (axes(self.basis, 1), axes(self.coefs)[2:end]...)
Base.getindex(self::BasisExpansion, indices...) =
    transpose(@view self.basis[indices[1], :]) * @view self.coefs[:, indices[2:end]...]
Base.IndexStyle(::Type{<:BasisExpansion}) = IndexCartesian()
Base.broadcastable(self::BasisExpansion) = self()

function BasisExpansion(
        basis::AbstractMatrix{T1}, coefs::AbstractVector{T2};
        precond = nothing
    ) where {T1, T2}
    Base.require_one_based_indexing(basis, coefs)
    m, n = size(basis)
    m > 0 && n > 0 || throw(ArgumentError("basis must have nonempty dimensions"))
    n == length(coefs) || throw(DimensionMismatch("basis columns must match the number of coefs"))
    precond in (:column_scaling, :qr, nothing) || throw(ArgumentError("unknown preconditioning precond: $precond"))
    T = get_precision(T1)
    coefs isa Vector{T} || (coefs = Vector{T}(coefs))
    all(isfinite, basis) && all(isfinite, coefs) || throw(ArgumentError("basis and coefs must be finite"))

    return BasisExpansion(Val(precond), basis, coefs)
end

BasisExpansion(::Val{nothing}, basis::AbstractMatrix, coefs::AbstractVector{T}) where {T <: AbstractFloat} =
    BasisExpansion(basis, coefs, nothing)

function BasisExpansion(::Val{:column_scaling}, basis::AbstractMatrix, coefs::AbstractVector{T}) where {T <: AbstractFloat}
    m, n = size(basis)
    scales = [norm(col) / sqrt(T(m)) for col in eachcol(basis)]
    all(s -> isfinite(s) && s > 0, scales) ||
        throw(ArgumentError("basis columns must have finite, nonzero norms"))
    conditioned = basis ./ reshape(scales, 1, :)
    transformed = scales .* coefs
    all(isfinite, conditioned) && all(isfinite, transformed) ||
        throw(ArgumentError("preconditioned basis and coefs must be finite"))
    return BasisExpansion(conditioned, transformed, scales)
end

function BasisExpansion(::Val{:qr}, basis::AbstractMatrix, coefs::AbstractVector{T}) where {T <: AbstractFloat}
    m, n = size(basis)
    m >= n || throw(ArgumentError("QR preconditioning requires a full-column-rank basis"))
    F = qr(Matrix(basis))
    R = F.R
    rank(R) == n || throw(ArgumentError("QR preconditioning requires a full-column-rank basis"))
    transformed = R * coefs
    all(isfinite, transformed) || throw(ArgumentError("preconditioned coefs must be finite"))
    return BasisExpansion(Matrix(F.Q), transformed, R)
end

((; basis, coefs)::BasisExpansion)() = basis * coefs
Optimisers.trainable(self::BasisExpansion) = (; coefs = self.coefs)

"""
    get_coefficients(representation)

Return the coefficients in the original basis. For a preconditioned
representation, convert its current trainable coefficients back to the
original basis.
"""
get_coefficients(self::BasisExpansion) = self.coefs
get_coefficients(self::BasisExpansion{T, N, B, C, <:AbstractVector}) where {T, N, B, C} =
    self.coefs ./ self.transform
get_coefficients(self::BasisExpansion{T, N, B, C, <:AbstractMatrix}) where {T, N, B, C} =
    self.transform \ self.coefs

"""
    set_coefficients!(representation, coefs)

Set the coefficients in the original basis and return `representation`.
For a preconditioned representation, convert them to the preconditioned
coordinates before updating its trainable coefficients.
"""
set_coefficients!(self::BasisExpansion, coefs) = (self.coefs .= coefs; self)
set_coefficients!(self::BasisExpansion{T, N, B, C, <:AbstractVector}, coefs) where {T, N, B, C} =
    (self.coefs .= self.transform .* coefs; self)
set_coefficients!(self::BasisExpansion{T, N, B, C, <:AbstractMatrix}, coefs) where {T, N, B, C} =
    (self.coefs .= self.transform * coefs; self)

"""
    polynomial_representation(evaluation_points, degree; precond = :qr)

Construct a polynomial [`BasisExpansion`](@ref) evaluated at the values in
`evaluation_points`, using powers from zero through `degree` of the points
shifted by their minimum. The coefficient vector has `degree + 1` entries;
its constant coefficient is initialized to the minimum evaluation point, and
its linear coefficient is initialized to one when `degree >= 2`.

`evaluation_points` must be nonempty and finite. `degree` must be positive
and no greater than the number of evaluation points. `precond` selects the
preconditioning method used by [`BasisExpansion`](@ref).
"""
function polynomial_representation(evaluation_points::AbstractVector{T}, degree::Integer; precond = :qr) where {T <: AbstractFloat}
    degree > 0 || throw(ArgumentError("degree must be positive"))
    degree <= length(evaluation_points) || throw(ArgumentError("degree must be less than or equal to the number of evaluation points"))
    all(isfinite, evaluation_points) || throw(ArgumentError("evaluation points must be finite"))
    minx = minimum(evaluation_points)
    shifted_points = evaluation_points .- minx
    basis = broadcast(^, shifted_points, (0:(degree))')
    coefs = zeros(T, degree + 1)
    coefs[1] = minx
    degree >= 2 && (coefs[2] = one(T))
    return BasisExpansion(basis, coefs; precond)
end

"""
    complex_polynomial_representation(evaluation_points, degree; precond = :qr)

Construct a complex-valued polynomial [`BasisExpansion`](@ref) evaluated at
the values in `evaluation_points`. Its basis contains powers from zero
through `degree` of the points shifted by their minimum, along with the
corresponding imaginary basis columns. The coefficient vector has
`2 * (degree + 1)` real entries.

`evaluation_points` must be nonempty and finite. `degree` must be positive
and no greater than the number of evaluation points. `precond` selects the
preconditioning method used by [`BasisExpansion`](@ref).
"""
function complex_polynomial_representation(evaluation_points::AbstractVector{T}, degree::Integer; precond = :qr) where {T <: AbstractFloat}
    degree > 0 || throw(ArgumentError("degree must be positive"))
    degree <= length(evaluation_points) || throw(ArgumentError("degree must be less than or equal to the number of evaluation points"))
    all(isfinite, evaluation_points) || throw(ArgumentError("evaluation points must be finite"))
    minx = minimum(evaluation_points)
    shifted_points = evaluation_points .- minx
    basis = broadcast(^, Complex.(shifted_points), (0:degree)')
    basis = hcat(basis, im .* basis)

    coefs = zeros(T, 2 * (degree + 1))
    coefs[1] = minx
    coefs[2] = minx
    coefs[3] = one(T)
    coefs[4] = one(T)
    return BasisExpansion(basis, coefs; precond)
end

end
