using BasisExpansions
using LinearAlgebra
using Optimisers
using SparseArrays
using Test

using BasisExpansions: get_coefficients, set_coefficients!, polynomial_representation, complex_polynomial_representation
@testset "BasisExpansion" begin
    knots = 0.0f0:0.5f0:4.0f0
    basis = hcat([knots .^ n for n in 0:2]...)
    coefs = Float32[0, 0.1, 0.1]
    original = BasisExpansion(basis, coefs)

    @test original isa AbstractArray{Float32, 1}
    @test size(original) == size(original())
    @test axes(original) == (axes(basis, 1),)
    @test original[2] ≈ original()[2]
    @test sin.(original) ≈ sin.(original())

    complex_basis = ComplexF32[1 0; im 1; 2 - im 0.5im]
    complex_coefs = Float32[0.5, -0.25]
    complex_model = BasisExpansion(complex_basis, complex_coefs)
    @test complex_model() ≈ complex_basis * complex_coefs

    for precond in (:column_scaling, :qr)
        model = BasisExpansion(basis, copy(coefs); precond)
        @test model() ≈ original()
        @test get_coefficients(model) ≈ coefs atol = 1.0f-6
        @test set_coefficients!(model, coefs) === model
        @test get_coefficients(model) ≈ coefs atol = 1.0f-6
        @test keys(Optimisers.trainable(model)) == (:coefs,)

        model.coefs .+= Float32[0.2, -0.3, 0.4]
        @test model() ≈ basis * get_coefficients(model) atol = 1.0f-6

        optimizer = Optimisers.Descent(0.1f0)
        state = Optimisers.setup(optimizer, model)
        gradient = (; basis = nothing, coefs = ones(Float32, 3), transform = nothing)
        _, updated = Optimisers.update(state, model, gradient)
        @test updated.coefs ≈ model.coefs .- 0.1f0
    end

    for T in (Float32, Float64)
        sparse_basis = sparse(T[1 0; 0 2; 1 1])
        sparse_coefs = T[1, 2]
        for precond in (nothing, :column_scaling, :qr)
            model = BasisExpansion(sparse_basis, copy(sparse_coefs); precond)
            @test model() ≈ sparse_basis * sparse_coefs
            @test get_coefficients(model) ≈ sparse_coefs
            if precond !== :qr
                @test model.basis isa SparseMatrixCSC
            end
        end
    end

    @test polynomial_representation(Float32[2, 3, 4], 1; precond = nothing)() == fill(2.0f0, 3)
    @test polynomial_representation(knots, 3)() ≈ knots
    @test_throws ArgumentError BasisExpansion(ones(Float32, 2, 3), coefs; precond = :qr)
    @test_throws ArgumentError BasisExpansion(ones(Float32, 3, 2), Float32[1, 2]; precond = :qr)
    @test_throws ArgumentError BasisExpansion(zeros(Float32, 3, 1), Float32[1]; precond = :column_scaling)
    @test_throws "basis columns must match the number of coefs" BasisExpansion(basis, coefs[1:2])
    @test_throws "basis must have nonempty dimensions" BasisExpansion(zeros(Float32, 0, 1), Float32[1])
    @test_throws "basis and coefs must be finite" BasisExpansion(basis, Float32[0, NaN, 0])
end
