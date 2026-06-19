using CHOLMODSolve
using SparseArrays
using LinearAlgebra
using BenchmarkTools

N = 5000

A = sprand(N, N, 0.005) + N * I(N)
A = 0.5 * (A + A')
b = randn(N)
x = similar(b)

A_lu = lu(sparse(A))
A_chol = cholesky(sparse(A))
s = CholSolve(A_chol)
solve!(s, x, b)  # warm-up

println("UMFPACK (lu):")
@btime ldiv!(x, $A_lu, $b)

println("CHOLMOD standard (cholesky):")
@btime ldiv!(x, $A_chol, $b)

println("CHOLMODSolve (solve!):")
@btime solve!($s, $x, $b)

free_workspace!(s)