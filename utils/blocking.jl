# Libraries ____________________________________________________________________
using LinearAlgebra, SparseArrays

# Functions ____________________________________________________________________
"""
    κ = BlockIndexing(N, S, Sb)

    Obtain a matrix of indices κ[nx,b] where nx is the set of variable indices (∈ 1:N)
    associated with the b-th block (by partitioning the space S into blocks of size Sb)
"""
function BlockIndexing(N::Int64, S::Tuple{Int64,Int64}, Sb::Tuple{Int64,Int64})
    Nb = (S[1]÷Sb[1])*(S[2]÷Sb[2]);     # Number of blocks
    if N >= (S[1]*S[2])  
        # Creates the matrix of indices in the spatial coordinates
        κ_2d = reshape(1:S[1]*S[2], S)
        κ_2d = vcat([κ_2d[:,c:c+Sb[2]-1] for c in 1:Sb[2]:S[2]]...);            # Slices columnwise them vcat
        κ_2d = vec.([κ_2d[r:r+Sb[1]-1,:] for r in 1:Sb[1]:S[1]*(S[2]÷Sb[2])]);  # Slices rowwise them vectorizes
    
        Nv = N ÷ (S[1]*S[2]);             # Number of variables per grid-point
        κ_1d = Matrix{Int64}(undef, Nv*Sb[1]*Sb[2], Nb)
        for b in axes(κ_1d,2)
            κ_1d[:,b] = vec(κ_2d[b] .+ (0:Nv-1)'.*(S[1]*S[2]))
        end    
    else
        Nr = (S[1]*S[2]) ÷ N;
        # Creates the matrix of indices in the spatial coordinates
        Sy = Int.(ceil.(S./Nr))
        Sby = Int.(ceil.(Sb./Nr))
        κ_2d = reshape(1:Sy[1]*Sy[2], Sy)
        κ_2d = vcat([κ_2d[:,c:c+Sby[2]-1] for c in 1:Sby[2]:Sy[2]]...);            # Slices columnwise them vcat
        κ_2d = vec.([κ_2d[r:r+Sby[1]-1,:] for r in 1:Sby[1]:Sy[1]*(Sy[2]÷Sby[2])]);  # Slices rowwise them vectorizes
    
        κ_1d = Matrix{Int64}(undef, Int((1/Nr)*Sb[1]*Sb[2]), Nb)
        for b in axes(κ_1d,2)
            κ_1d[:,b] = vec(κ_2d[b] .+ (0:(N ÷ (S[1]*S[2])))'.*(Sy[1]*Sy[2]))
        end
    end
    
    # Creates the matrix of flatenned indices
    
    return κ_1d
end

# ______________________________________________________________________________
