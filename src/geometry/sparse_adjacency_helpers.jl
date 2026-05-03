"""
Helper functions for accessing and sampling from sparse adjacency matrices.
"""

"""
Get all neighboring sites for a given site from the sparse adjacency matrix.
"""
@inline function get_neighbors(adjacency_matrix::Adjacency_Matrix, site::Int64)::Vector{Int64}
    A = adjacency_matrix.data
    # For column-compressed sparse matrix, get row indices of nonzero entries in column `site`
    start_idx = A.colptr[site + begin]
    end_idx = A.colptr[site + 1 + begin] - 1
    return A.rowval[start_idx:end_idx] .- 1
end

"""
Get the number of neighbors for a given site.
"""
@inline function get_neighbor_count(adjacency_matrix::Adjacency_Matrix, site::Int64)::Int64
    return adjacency_matrix.total_nn[site + begin]
end

"""
Randomly select a neighboring site for a given site from the sparse adjacency matrix.
Uses uniform sampling across all neighbors.
"""
@inline function get_random_neighbor(rng::AbstractRNG, adjacency_matrix::Adjacency_Matrix, site::Int64)::Int64
    neighbors = get_neighbors(adjacency_matrix, site)
    # Randomly select one of the neighbors
    neighbor_idx = rand(rng, eachindex(neighbors))
    return neighbors[neighbor_idx]
end
