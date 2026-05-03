#=
The adjacency matrix contains information about the connectivity of the lattice.
For example, take a 3x3 square lattice with periodic boundary conditions:

	3 -- 6 -- 9
	|    |    |
	2 -- 5 -- 8
	|    |    |
	1 -- 4 -- 7

then the adjacency matrix A is

    2  3  4  7	(neighbors of 1)
 	1  3  5  8	(neighbors of 2)
	1  2  6  9	(neighbors of 3)
	1  5  6  7	(neighbors of 4)
	2  4  6  8	(neighbors of 5)
	3  4  5  9	(neighbors of 6)
	1  4  8  9	(neighbors of 7)
	2  5  7  9	(neighbors of 8)
	3  6  7  8	(neighbors of 9)

such that A[i,:] are the neighbors of site i.

With open boundary conditions, the adjacency matrix is

    2     4   	(neighbors of 1)
	1  3  5   	(neighbors of 2)
	   2  6   	(neighbors of 3)
	1  5     7	(neighbors of 4)
	2  4  6  8	(neighbors of 5)
	3     5  9	(neighbors of 6)
	   4  8   	(neighbors of 7)
	   5  7  9	(neighbors of 8)
	   6     8	(neighbors of 9)

where sites now have different coordination numbers.
=#

using LatticeModels
using SparseArrays

struct Adjacency_Matrix <: AbstractMatrixData
    data::SparseMatrixCSC{Int64,Int64}
    L::Int64              # first axis length (Ls[1]); preserved for backward compat
    D::Int64
    M::Int64
    boundary_condition::String   # uniform BC string, or "mixed" for per-axis
    total_nn::Vector{Int64}
    Ls::Vector{Int64}            # per-axis lengths
    bcs::Vector{String}          # per-axis boundary conditions
end

# Returns the uniform BC string when all axes match, otherwise "mixed".
_uniform_bc(bcs::Vector{String}) = all(==(bcs[1]), bcs) ? bcs[1] : "mixed"

# ── Backward-compatible scalar interface ──────────────────────────────────────
function create_adjacency_matrix(L::Int64, D::Int64, geometry::String, boundary_condition::String)
    n_axes = geometry == "square" ? D : 2
    create_adjacency_matrix(fill(L, n_axes), D, geometry, fill(boundary_condition, n_axes))
end

# ── Primary vector interface ──────────────────────────────────────────────────
function create_adjacency_matrix(Ls::Vector{Int64}, D::Int64, geometry::String, bcs::Vector{String};
                                  unitcell::Union{UnitCell,Nothing}=nothing)

    if !(geometry in ["square","triangular","honeycomb","kagome","custom"])
        error("Unknown geometry '$geometry'. Choose: square, triangular, honeycomb, kagome, or custom.")
    end
    if geometry in ["triangular","honeycomb","kagome"] && D != 2
        error("Geometry '$geometry' requires D=2.")
    end
    if any(L -> L < 2, Ls) && geometry != "custom"
        error("Lattice size must be at least 2 along each axis.")
    end
    for bc in bcs
        bc in ["pbc","obc"] || error("Boundary conditions must be pbc or obc; got '$bc'.")
    end
    if geometry == "custom" && unitcell === nothing
        error("Custom geometry requires a unit cell.")
    end

    if geometry == "square"
        return square_adjacency_matrix(Ls, D, bcs)
    elseif geometry == "triangular"
        return triangular_adjacency_matrix(Ls, bcs)
    elseif geometry == "honeycomb"
        return honeycomb_adjacency_matrix(Ls, bcs)
    elseif geometry == "kagome"
        return kagome_adjacency_matrix(Ls, bcs)
    else
        return custom_adjacency_matrix(Ls, bcs, unitcell)
    end
end

# ── Per-geometry construction ─────────────────────────────────────────────────

function square_adjacency_matrix(Ls::Vector{Int64}, D::Int64, bcs::Vector{String})
    lattice = SquareLattice(Ls...)
    if D == 1
        # 1D SquareLattice has 2D Bravais coordinates internally.
        if bcs[1] == "pbc"
            lattice = setboundaries(lattice, PeriodicBoundary([Ls[1], 0]))
        end
    else
        pbcs = [PeriodicBoundary([(j == i ? Ls[i] : 0) for j=1:D]) for i=1:D if bcs[i]=="pbc"]
        isempty(pbcs) || (lattice = setboundaries(lattice, pbcs...))
    end
    sparse_mat = AdjacencyMatrix(lattice, NearestNeighbor()).mat
    total_nn = [nnz(sparse_mat[:, i]) for i in axes(sparse_mat, 2)]
    return Adjacency_Matrix(sparse_mat, Ls[1], D, prod(Ls), _uniform_bc(bcs), total_nn, Ls, bcs)
end

#=
Example of triangular lattice site numbering for L=3:

	    3 -- 6 -- 9
	   /  \ /  \ /
	  2 -- 5 -- 8
	 /  \ /  \ /
	1 -- 4 -- 7

=#
function triangular_adjacency_matrix(Ls::Vector{Int64}, bcs::Vector{String})
    lattice = TriangularLattice(Ls[1], Ls[2],
        boundaries=(:axis1 => bcs[1]=="pbc", :axis2 => bcs[2]=="pbc"))
    sparse_mat = AdjacencyMatrix(lattice, NearestNeighbor()).mat
    total_nn = [nnz(sparse_mat[:, i]) for i in axes(sparse_mat, 2)]
    return Adjacency_Matrix(sparse_mat, Ls[1], 2, prod(Ls), _uniform_bc(bcs), total_nn, Ls, bcs)
end

#=
Example of honeycomb lattice site numbering for L=3:

	            6       12      18
	          /   \   /   \   /
	        5       11      17
	        |       |       |
	        4       10      16
	      /   \   /   \   /
	    3       9       15
	    |       |       |
	    2       8       14
	  /   \   /   \   /
	 1      7       13

=#
function honeycomb_adjacency_matrix(Ls::Vector{Int64}, bcs::Vector{String})
    lattice = HoneycombLattice(Ls[1], Ls[2],
        boundaries=(:axis1 => bcs[1]=="pbc", :axis2 => bcs[2]=="pbc"))
    sparse_mat = AdjacencyMatrix(lattice, NearestNeighbor()).mat
    total_nn = [nnz(sparse_mat[:, i]) for i in axes(sparse_mat, 2)]
    return Adjacency_Matrix(sparse_mat, Ls[1], 2, 2*prod(Ls), _uniform_bc(bcs), total_nn, Ls, bcs)
end

#=
Example of kagome lattice site numbering for L=3:

	               9           18          27
	              /  \        /  \        /  \
	            7  -- 8  -- 16 -- 17 -- 25 -- 26
	           /        \  /        \  /
	         6           15          24
	        /  \        /  \        /  \
	      4  -- 5  -- 13 -- 14 -- 22 -- 23
	     /        \  /        \  /
	   3           12          21
	  /  \        /  \        /  \
	1  -- 2  -- 10 -- 11 -- 19 -- 20

=#
function kagome_adjacency_matrix(Ls::Vector{Int64}, bcs::Vector{String})
    lattice = KagomeLattice(Ls[1], Ls[2],
        boundaries=(:axis1 => bcs[1]=="pbc", :axis2 => bcs[2]=="pbc"))
    sparse_mat = AdjacencyMatrix(lattice, NearestNeighbor()).mat
    total_nn = [nnz(sparse_mat[:, i]) for i in axes(sparse_mat, 2)]
    return Adjacency_Matrix(sparse_mat, Ls[1], 2, 3*prod(Ls), _uniform_bc(bcs), total_nn, Ls, bcs)
end

function custom_adjacency_matrix(Ls::Vector{Int64}, bcs::Vector{String}, uc::UnitCell)
    D = length(Ls)
    lattice = span_unitcells(uc, (0:(L-1) for L in Ls)...)
    M = length(lattice)
    pbcs = [PeriodicBoundary([(j == i ? Ls[i] : 0) for j=1:D]) for i=1:D if bcs[i]=="pbc"]
    isempty(pbcs) || (lattice = setboundaries(lattice, pbcs...))
    sparse_mat = AdjacencyMatrix(lattice, NearestNeighbor()).mat
    total_nn = [nnz(sparse_mat[:, i]) for i in axes(sparse_mat, 2)]
    return Adjacency_Matrix(sparse_mat, Ls[1], D, M, _uniform_bc(bcs), total_nn, Ls, bcs)
end
