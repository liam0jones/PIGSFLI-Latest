struct Path <: AbstractVectorData 
    data::Vector{Kink} 
end
Path(num_empty_kinks::Int64) = Path([Kink(-1.0,-1,-1,-1,-1,-1,-1,-1) for _ in 1:num_empty_kinks]) 
 
"""Create a path from a fock state."""
function create_paths(fock_state::FockState, M::Int64, replica_idx::Int64)::Path
    # Set the number of kinks to pre-allocate based on lattice size
    num_empty_kinks = M*1000  
    # Pre-allocate kinks
    paths = Path(num_empty_kinks)
    # Initialize first M kinks 
    for site in 0:M-1
        # replica index shifted by -1 due to 0 vs 1 indexing 
        fill_kink!(paths[site+begin], 0.0, fock_state[site+begin], site, site, -1, -1, replica_idx-1, replica_idx-1)
    end
    return paths
end

function fill_kink!(
    kink::Kink,
    tau::Float64,
    n::Int64,
    src::Int64,
    dest::Int64,
    prev::Int64,
    next::Int64,
    src_replica::Int64,
    dest_replica::Int64
) 
    kink.tau = tau
    kink.n = n
    kink.src = src
    kink.dest = dest
    kink.prev = prev
    kink.next = next
    kink.src_replica = src_replica
    kink.dest_replica = dest_replica
    return nothing
end