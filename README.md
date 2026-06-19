# CHOLMODSolve.jl

A thin wrapper around CHOLMOD's `cholmod_l_solve2` that eliminates per-solve heap allocations when repeatedly solving sparse symmetric positive definite systems with the same factorization.

## Motivation

Julia's standard `cholesky(A)` allocates memory every solve, scaling with the problem dimension, which is caused by the allocation and freeing of internal workspace buffers (`Y` and `E`) on every call. For cases where many repeated solves are required, e.g. in PDE-constrained optimization, these allocations can add up to a significant amount.
`CHOLMODSolve` pre-allocates these buffers once and reuses them, reducing allocations to a constant 16 bytes per solve regardless of matrix size.

## Usage

```julia
using SparseArrays, LinearAlgebra, CHOLMODSolve

A = # sparse SPD
F = cholesky(A)

s = CholSolve(F) # pre-allocate workspace
x = similar(b)

solve!(s, x, b) # solves F * x = b

free_workspace!(s) # release CHOLMOD Y and E buffers 
```

Multiple right-hand sides are supported via the `nrhs` keyword:

```julia
s = CholSolve(F; nrhs=3)
X = zeros(n, 3)
solve!(s, X, B)
```

## Notice

This package calls internal APIs from `SparseArrays.CHOLMOD` that are not part of Julia's public interface and may break on future Julia versions. Tested on Julia 1.12.
