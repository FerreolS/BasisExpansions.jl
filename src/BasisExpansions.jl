module BasisExpansions

using LinearAlgebra
using Optimisers
using TypeUtils: get_precision

export BasisExpansion,
    complex_polynomial_representation,
    get_coefficients,
    polynomial_representation,
    set_coefficients!

"""
    BasisExpansion(basis, coefs; precond = nothing)

Represent `basis * coefs` using trainable coefficients, optionally in
preconditioned coordinates. The input `coefs` are coefficients of the
original basis. Use [`get_coefficients`](@ref) to recover original-basis
coefficients after optimization.

With `precond = :column_scaling`, each basis column is scaled to have unit
root mean square. With `precond = :qr`, the columns are orthonormalized using
a thin QR factorization of a dense copy; the basis must have full column rank.
All methods reject nonfinite inputs and empty dimensions. Column scaling also
rejects zero columns. Inputs must use one-based indexing.

The representation behaves as an `AbstractArray` with the element type of
`basis * coefs` and the row axis of `basis`.
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

Base.size(self::BasisExpansion) = (size(self.basis, 1), size(self.coefs)[2:end]...)
Base.axes(self::BasisExpansion) = (axes(self.basis, 1), axes(self.coefs)[2:end]...)
Base.getindex(self::BasisExpansion, indices...) =
    transpose(@view self.basis[indices[1], :]) * @view self.coefs[:, indices[2:end]...]
Base.IndexStyle(::Type{<:BasisExpansion}) = IndexCartesian()
Base.broadcastable(self::BasisExpansion) = self()

Optimisers.isnumeric(::BasisExpansion) = false
function Optimisers.functor(
        ::Type{BasisExpansion{T, N, B, C, P}}, self
    ) where {T, N, B, C, P}
    fields = (; basis = self.basis, coefs = self.coefs, transform = self.transform)
    reconstruct(fields) = BasisExpansion{T, N, B, C, P}(
        fields.basis, fields.coefs, fields.transform
    )
    return fields, reconstruct
end

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

Return the coefficients of the original basis. For a preconditioned
representation, these are computed from its current trainable coefficients.
"""
get_coefficients(self::BasisExpansion) = self.coefs
get_coefficients(self::BasisExpansion{T, N, B, C, <:AbstractVector}) where {T, N, B, C} =
    self.coefs ./ self.transform
get_coefficients(self::BasisExpansion{T, N, B, C, <:AbstractMatrix}) where {T, N, B, C} =
    self.transform \ self.coefs

"""
    set_coefficients!(representation, coefs)

Set the coefficients of the representation. For a preconditioned
representation, the coefficients are transformed to the preconditioned space.
"""
set_coefficients!(self::BasisExpansion, coefs) = (self.coefs .= coefs; self)
set_coefficients!(self::BasisExpansion{T, N, B, C, <:AbstractVector}, coefs) where {T, N, B, C} =
    (self.coefs .= self.transform .* coefs; self)
set_coefficients!(self::BasisExpansion{T, N, B, C, <:AbstractMatrix}, coefs) where {T, N, B, C} =
    (self.coefs .= self.transform * coefs; self)

function polynomial_representation(knots::AbstractVector{T}, degree::Integer; precond = :qr) where {T <: AbstractFloat}
    degree > 0 || throw(ArgumentError("degree must be positive"))
    degree <= length(knots) || throw(ArgumentError("degree must be less than or equal to the number of knots"))
    all(isfinite, knots) || throw(ArgumentError("knots must be finite"))
    mink = minimum(knots)
    knots = knots .- mink
    basis = broadcast(^, knots, (0:(degree - 1))')
    coefs = zeros(T, degree)
    coefs[1] = mink
    degree >= 2 && (coefs[2] = one(T))
    return BasisExpansion(basis, coefs; precond)
end

function complex_polynomial_representation(knots::AbstractVector{T}, degree::Integer; precond = :qr) where {T <: AbstractFloat}
    degree > 0 || throw(ArgumentError("degree must be positive"))
    degree <= length(knots) || throw(ArgumentError("degree must be less than or equal to the number of knots"))
    all(isfinite, knots) || throw(ArgumentError("knots must be finite"))
    mink = minimum(knots)
    knots = knots .- mink
    basis = broadcast(^, Complex.(knots), (0:degree)')
    basis = hcat(basis, im .* basis)

    coefs = zeros(T, 2 * (degree + 1))
    coefs[1] = mink
    coefs[2] = mink
    coefs[3] = one(T)
    coefs[4] = one(T)
    return BasisExpansion(basis, coefs; precond)
end

end
