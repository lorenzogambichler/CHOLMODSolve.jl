using CHOLMODSolve
using SparseArrays
using LinearAlgebra
using Test

@testset "CHOLMODSolve" begin
    for T in (Float64, Float32)
        N = 200
        A = sprand(T, N, N, 0.01) + N * I(N)
        A = T(0.5) * (A + A')
        b = randn(T, N)
        x = similar(b)

        F = cholesky(A)
        s = CholSolve(F)

        solve!(s, x, b)
        @test x ≈ F \ b rtol=sqrt(eps(T))

        B = randn(T, N, 3)
        X = similar(B)
        s2 = CholSolve(F; nrhs=3)
        solve!(s2, X, B)
        @test X ≈ F \ B rtol=sqrt(eps(T))

        free_workspace!(s)
        free_workspace!(s2)
    end
end
