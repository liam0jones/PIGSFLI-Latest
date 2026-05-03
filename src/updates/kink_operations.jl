"""Takes an array and two indices and swaps the elements at those indices. This function assumes zero-based indexing for the indices i and j."""
@inline function swap!(array, i::Int64, j::Int64)
    array[i+begin], array[j+begin] = array[j+begin], array[i+begin]
end 

"""Given the path vector and an index to a kink inside this vector, attempt to fix the links to this kink by setting the links from its next and previous indices. This function returns,  whether index is the last kink on the site."""
@inline function fix_links_to_kink!(path::Path, index::Int64)::Bool
    kink = path[index+begin]
    if kink.next != -1
        path[kink.next+begin].prev = index 
        is_last = false
    else 
        is_last = true
    end 
    path[kink.prev+begin].next = index
    if kink.partner != -1
        path[kink.partner+begin].partner = index
    end
    return is_last
end

"""Given the path vector and an index, links the neigbors of the kink together and returns whether index is the last kink on the site. This is a helper function for deletion of the kink with this index.
The links in the kink at index are not changed.
This is: 
    prev -> index -> next  ==>   prev -> next   return false
    prev -> index -> -1    ==>   prev -> -1     return true
"""
@inline function link_neighbors!(path::Path, index::Int64)::Bool
    is_last = false
    kink = path[index+begin]
    if kink.next == -1
        is_last = true
        path[kink.prev+begin].next = -1
    else 
        path[kink.prev+begin].next = kink.next
        path[kink.next+begin].prev = kink.prev
    end
    return is_last
end

"""Delete the kink at index and replica_index in the path.
Here index assumes 0-based indexing. Replica_index currently uses 1-based indexing.
"""
function delete_kink!(sim_state::SimState, replica_index::Int64, index::Int64)::Nothing
    path::Path = sim_state.paths[replica_index]
    # link neighbors of removed kink 
    deleted_is_last_on_site::Bool = link_neighbors!(path, index)  
    if deleted_is_last_on_site  
        # deleted the last kink on its site 
        sim_state.last_kinks[replica_index][path[index+begin].src+begin] = path[index+begin].prev 
    end
    # set partner to -1 for deleted kink
    path[index+begin].partner = -1
    # swap kink to be removed with the last active kink in the path  
    last_active_index = sim_state.num_kinks[replica_index]-1
    swap!(path, index, last_active_index)
    swapped_kink_index = index
    # repair swapped kinks neighbors' links 
    if swapped_kink_index != last_active_index 
        fix_links_to_kink!(path, swapped_kink_index)
    end
    # check if the head was swapped or removed 
    if sim_state.head_idx[replica_index] == index 
        sim_state.head_idx[replica_index] = -1
    elseif sim_state.head_idx[replica_index] == last_active_index
        sim_state.head_idx[replica_index] = swapped_kink_index 
    end
    # check if the tail was swapped of removed 
    if sim_state.tail_idx[replica_index] == index 
        sim_state.tail_idx[replica_index] = -1
    elseif sim_state.tail_idx[replica_index] == last_active_index
        sim_state.tail_idx[replica_index] = swapped_kink_index 
    end
    # check if final index on site was swapped or removed
    last_kink_idx_on_swapped_kinks_site = sim_state.last_kinks[replica_index][path[swapped_kink_index+begin].src+begin]
    if last_kink_idx_on_swapped_kinks_site == last_active_index
        sim_state.last_kinks[replica_index][path[swapped_kink_index+begin].src+begin] = swapped_kink_index
    end  
    # change the number of active kinks in the path
    sim_state.num_kinks[replica_index] -= 1

    return nothing
end

"""Deletes the kinks at indices index_1 and index_2 for replica_index in the paths vector.
Here index assumes 0-based indexing. Replica_index currently uses 1-based indexing.

In addition to calling delete_kink! twice, this keeps track whether the second index has been changed by the first deletion and adjusts the second index accordingly.

Warning: This function may behave unexpectedly if the same index is passed twice or if the index is not an active kink index. For speed reasons, this is not checked.
"""
@inline function delete_kink_pair!(sim_state::SimState, replica_index::Int64, index_1::Int64, index_2::Int64)::Nothing 
    last_active_index = sim_state.num_kinks[replica_index]-1
    delete_kink!(sim_state, replica_index, index_1)
    if index_2 == last_active_index
        index_2 = index_1
    end
    delete_kink!(sim_state, replica_index, index_2)
    return nothing
end

"""Moves a kink (the tail or head) with index to a new site. 
After the move, the path will be back in a valid state. The kink index within the path vector will be unchanged."""
@inline function move_kink_to_other_site!(sim_state::SimState, replica_index::Int64, kink_index::Int64, new_n::Int64, new_prev::Int64, new_next::Int64, new_site::Int64)::Nothing 
    path = sim_state.paths[replica_index] 
    # link the neighbors from the old site together
    is_last = link_neighbors!(path, kink_index) 
    # fix if kink is last on its site  
    if is_last 
        sim_state.last_kinks[replica_index][path[kink_index+begin].src+begin] = path[kink_index+begin].prev
    end
    # put the kink at the new site
    path[kink_index+begin].n = new_n
    path[kink_index+begin].src = new_site 
    path[kink_index+begin].dest = new_site
    # link the kink to the neighbors at the new site
    path[kink_index+begin].next = new_next 
    path[kink_index+begin].prev = new_prev
    is_last = fix_links_to_kink!(path, kink_index)
    # check if the kink is now the last kink on its site
    if is_last
        sim_state.last_kinks[replica_index][new_site+begin] = kink_index
    end 

    return nothing
end
@inline move_kink_to_other_site!(sim_state::SimState, replica_index::Int64, kink_index::Int64, new_n::Int64, new_prev::Int64) = move_kink_to_other_site!(sim_state, replica_index, kink_index, new_n, new_prev, sim_state.paths[replica_index][new_prev+begin].next, sim_state.paths[replica_index][new_prev+begin].src)  

"""Insert a kink between kink_below and the next kink in the path. Returns the index of the inserted kink in the path. 

Note that there are separate functions for inserting a head or tail kink that keep track of the head and tail indices in the sim_state (which is not done by this function here). Inserting a single kink also does not take into account its potential partner kink (for inserting a pair of kinks, use insert_pair_of_kinks!)."""
@inline function insert_kink!(sim_state::SimState, replica_index::Int64, n::Int64, tau::Float64, dest::Int64, index_kink_below::Int64)::Int64
    # extract path and last_kinks vector
    path = sim_state.paths[replica_index]
    kink_below = path[index_kink_below+begin] 
    # put new kink on end of path (zero indexing)
    num_kinks = sim_state.num_kinks[replica_index]
    fill_kink!(path[num_kinks+begin], tau, n, kink_below.src, dest, index_kink_below, kink_below.next, kink_below.src_replica, kink_below.src_replica)
    # link the new kink to its neighbors
    fix_links_to_kink!(path, num_kinks)
    # check if the new kink is last on its site
    if index_kink_below == sim_state.last_kinks[replica_index][kink_below.src+begin]
        sim_state.last_kinks[replica_index][kink_below.src+begin] = num_kinks
    end
    # increment the number of kinks in the path
    sim_state.num_kinks[replica_index] += 1
    # return the index of the new kink
    return num_kinks
end

"""Insert a pair of connected kinks with particle numbers ni and nj. Here, index_kink_below_i is the index of the kink below the new kink on i and index_kink_below_j is the index of the kink below the new kink on site j."""
@inline function insert_pair_of_kinks!(sim_state::SimState, replica_index::Int64, ni::Int64, nj::Int64, tau::Float64, index_kink_below_i::Int64, index_kink_below_j::Int64) 
    path = sim_state.paths[replica_index]
    dest_i = path[index_kink_below_j+begin].src
    index_i = insert_kink!(sim_state, replica_index, ni, tau, dest_i, index_kink_below_i)
    dest_j = path[index_kink_below_i+begin].src
    index_j = insert_kink!(sim_state, replica_index, nj, tau, dest_j, index_kink_below_j)
    # set partner indices
    path[index_i+begin].partner = index_j
    path[index_j+begin].partner = index_i
    return index_i, index_j
end

"""Insert a head kink. This function assumes that there is no head present on path.""" 
@inline function insert_head!(sim_state::SimState, replica_index::Int64, n::Int64, tau::Float64, index_kink_below::Int64)::Int64
    path = sim_state.paths[replica_index]
    num_kinks = insert_kink!(sim_state, replica_index, n, tau, path[index_kink_below+begin].src, index_kink_below)
    sim_state.head_idx[replica_index] = num_kinks 
    return num_kinks
end 

"""Insert a tail kink. This function assumes that there is no tail present on path.""" 
@inline function insert_tail!(sim_state::SimState, replica_index::Int64, n::Int64, tau::Float64, index_kink_below::Int64)::Int64
    path = sim_state.paths[replica_index]
    num_kinks = insert_kink!(sim_state, replica_index, n, tau, path[index_kink_below+begin].src, index_kink_below)
    sim_state.tail_idx[replica_index] = num_kinks 
    return num_kinks
end

"""Finds the kink index in the paths array that is directly below a certain value of tau."""
function find_kink_below_tau_on_site(sim_state::SimState, replica_index::Int64, tau::Float64, site_index::Int64)
    path = sim_state.paths[replica_index] 
    current_kink_idx = site_index
    for _ in 1:sim_state.num_kinks[replica_index]
        next = path[current_kink_idx+begin].next
        if next == -1 || path[next+begin].tau > tau - DELTA_TAU/2
            return current_kink_idx  
        end
        current_kink_idx = next
    end
    error("Could not find a link below tau on site $site_index. This may be due to an invalid path.")
end

"""Sample a flat interval."""
@inline function sample_flat_interval(rng::AbstractRNG, sim_state::SimState, replica_index::Int64) 
    path = sim_state.paths[replica_index]
    # randomly sample a kink index, flat interval is from this kink to its next
    kink_index::Int64 = rand(rng, 0:sim_state.num_kinks[replica_index]-1)
    kink_below = path[kink_index+begin]
    tau_prev = kink_below.tau
    tau_next = kink_below.next == -1 ? sim_state.beta : path[kink_below.next+begin].tau

    return kink_below, kink_index, tau_prev, tau_next
end