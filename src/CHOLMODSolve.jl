module CHOLMODSolve

export CholSolve, solve!, free_workspace!

using SparseArrays.CHOLMOD: Factor, cholmod_dense_struct, wrap_dense_and_ptr,
    getcommon, cholmod_common, CHOLMOD_A, free!, xtyp, dtyp
using SparseArrays.CHOLMOD.LibSuiteSparse: cholmod_l_solve2

mutable struct CholSolve{T}
    F::Factor{T,Int64}
    dense_x::cholmod_dense_struct 
    dense_b::cholmod_dense_struct
    X::Ref{Ptr{cholmod_dense_struct}} 
    Y::Ref{Ptr{cholmod_dense_struct}} 
    E::Ref{Ptr{cholmod_dense_struct}}
    common::Base.RefValue{cholmod_common}
end

function _make_dense(::Type{T}, n, nrhs) where {T}
    d = cholmod_dense_struct()
    d.nrow, d.ncol, d.nzmax, d.d = n, nrhs, n * nrhs, n
    d.x, d.z = C_NULL, C_NULL
    d.xtype, d.dtype = xtyp(T), dtyp(T)
    return d
end

function CholSolve(F::Factor{T,Int64}; nrhs::Integer=1) where {T}
    n = size(F, 1)
    nul() = Ref(Ptr{cholmod_dense_struct}(C_NULL))
    CholSolve{T}(F, _make_dense(T, n, nrhs), _make_dense(T, n, nrhs),
        nul(), nul(), nul(), getcommon(Int64))
end

function solve!(s::CholSolve{T}, x::StridedVecOrMat{T}, b::StridedVecOrMat{T}) where {T}
    s.dense_x.x = pointer(x) # mutate in place
    s.dense_b.x = pointer(b)
    s.X[] = Ptr{cholmod_dense_struct}(pointer_from_objref(s.dense_x))
    Bptr = Ptr{cholmod_dense_struct}(pointer_from_objref(s.dense_b))
    status = GC.@preserve x b s begin 
        cholmod_l_solve2(CHOLMOD_A, s.F, Bptr, C_NULL,
            s.X, C_NULL, s.Y, s.E, s.common) # allocates 16 bytes
    end
    @assert !iszero(status)
end

function free_workspace!(s::CholSolve)
    s.Y[] != C_NULL && (free!(s.Y[]); s.Y[] = Ptr{cholmod_dense_struct}(C_NULL))
    s.E[] != C_NULL && (free!(s.E[]); s.E[] = Ptr{cholmod_dense_struct}(C_NULL))
    nothing
end

end # module
