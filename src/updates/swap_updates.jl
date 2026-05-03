function insert_swap_kink!(rng::AbstractRNG, sim_state::SimState, src_replica_index::Int64, sim_tracker:: SimTracker)
    # Extract some parameters 
    #rng::RandomNumberGenerator = sim_state.rng
    paths::Vector{Path} = sim_state.paths
    step_counters::MCStepCounters = sim_tracker.step_counters
    num_kinks::Vector{Int64} = sim_state.num_kinks 
    t::Float64 = sim_state.t
    U::Float64 = sim_state.U
    N::Int64 = sim_state.N
    mu::Float64 = sim_state.mu
    eta::Float64 = sim_state.eta 

    adjacency_matrix::Adjacency_Matrix = sim_state.adjacency_matrix
    num_swaps::Int64 = sim_state.num_swaps
    m_A::Int64 = sim_state.m_A
    sub_sites::Sites = sim_state.sub_sites
    last_kinks::Vector{Vector{Int64}} = sim_state.last_kinks


    if (length(paths) < 2)
        return
    end 
    # Can't perform update if SWAP region is full
    if ( num_swaps == m_A)
        return
    end 
    step_counters.insert_swap_kink_attempts += 1  
    # Retrieve source replica index and randomly choose destination replica
    src_replica = src_replica_index 
    if src_replica == 1 && length(paths) == 2
        dest_replica = 2
    elseif src_replica == 1
        dest_replica = rand(rng, 2:length(paths))
    else 
        dest_replica = rand(rng, vcat(1:src_replica-1,src_replica+1:length(paths)))
    end
    # Propose the next site to swap
    next_swap_site = sub_sites[num_swaps+begin] 
    
    #/*---------TEST THIS!!!*----------*/
    # Check if no. of particles at beta/2 is the same on both replicas
    # Source Replica
    tau = 0.0 
    next = next_swap_site  # next variable refers to "next kink" in worldline
    n_src = -1 
    prev_src = -1 
    while (tau<0.5*beta) 
        n_src = paths[src_replica][next+begin].n 
        prev_src = next 
        next = paths[src_replica][next+begin].next 

        if (next == -1)
            break
        end
        tau = paths[src_replica][next+begin].tau 
    end
    next_src = next 
    # Destination Replica
    tau = 0.0 
    next = next_swap_site 
    n_dest = -1 
    prev_dest = -1 
    while (tau < 0.5*beta) 
        n_dest = paths[dest_replica][next+begin].n 
        prev_dest = next;
        next = paths[dest_replica][next+begin].next 

        if (next == -1)
            break
        end
        tau = paths[dest_replica][next+begin].tau 
    end
    next_dest = next 
    
    if (n_src != n_dest)
        return
    end
                    
    # Build and insert kinks to the paths of the src and the dest replica
    num_kinks_src = num_kinks[src_replica] 
    num_kinks_dest = num_kinks[dest_replica] 
    #paths[src_replica][num_kinks_src+begin] =  Kink(beta/2.0,n_src,next_swap_site,next_swap_site,prev_src,next_src,src_replica,dest_replica);
    fill_kink!(paths[src_replica][num_kinks_src+begin] , beta/2.0,n_src,next_swap_site,next_swap_site,prev_src,next_src,src_replica,dest_replica)
    #paths[dest_replica][num_kinks_dest+begin] =  Kink(beta/2.0,n_src,next_swap_site,next_swap_site,prev_dest,next_dest,dest_replica,src_replica);
    fill_kink!(paths[dest_replica][num_kinks_dest+begin], beta/2.0,n_src,next_swap_site,next_swap_site,prev_dest,next_dest,dest_replica,src_replica)
    # Had to change the meaning of src_replica and dest_replica ATTRIBUTES
    # They now actually have directional meaning. From origin replica to other.

    # Connect next of prev_src to swap_kink
    paths[src_replica][prev_src+begin].next = num_kinks_src  
    # Connect prev of next_src to swap_kink
    if (next_src != -1)
        paths[src_replica][next_src+begin].prev = num_kinks_src 
    end 
    # Connect next of prev_dest to swap_kink
    paths[dest_replica][prev_dest+begin].next = num_kinks_dest 
    # Connect prev of next_dest to swap_kink
    if (next_dest != -1)
        paths[dest_replica][next_dest+begin].prev = num_kinks_dest 
    end
    # Edit the last kinks vector of each replica if necessary
    if (next_src == -1) 
        last_kinks[src_replica][next_swap_site+begin] = num_kinks_src 
    end
    if (next_dest == -1) 
        last_kinks[dest_replica][next_swap_site+begin] = num_kinks_dest 
    end 
    # Update number of swapped sites tracker
    sim_state.num_swaps += 1 
    num_swaps = sim_state.num_swaps 
    # Update number of kinks tracker of each replica
    num_kinks[src_replica] += 1 
    num_kinks[dest_replica] += 1 
    
    step_counters.insert_swap_kink_accepts += 1 

    return
end

function delete_swap_kink!(rng::AbstractRNG, sim_state::SimState, src_replica_index::Int64, sim_tracker:: SimTracker)
    # Extract some parameters 
    #rng::RandomNumberGenerator = sim_state.rng
    paths::Vector{Path} = sim_state.paths
    step_counters::MCStepCounters = sim_tracker.step_counters
    num_kinks::Vector{Int64} = sim_state.num_kinks 
    t::Float64 = sim_state.t
    U::Float64 = sim_state.U
    N::Int64 = sim_state.N
    mu::Float64 = sim_state.mu
    eta::Float64 = sim_state.eta 
    adjacency_matrix::Adjacency_Matrix = sim_state.adjacency_matrix
    num_swaps::Int64 = sim_state.num_swaps
    m_A::Int64 = sim_state.m_A
    sub_sites::Sites = sim_state.sub_sites
    last_kinks::Vector{Vector{Int64}} = sim_state.last_kinks
    head_idx::Vector{Int64} = sim_state.head_idx
    tail_idx::Vector{Int64} = sim_state.tail_idx

    # Need at least two replicas to perform delete_swap_kink
    if (length(paths) < 2)
        return
    end 
    # Need at least one swapped site to perform delete_swap_kink
    if ( num_swaps == 0)
        return
    end 
    step_counters.delete_swap_kink_attempts += 1  
    # Retrieve source replica index and randomly choose destination replica
    src_replica = src_replica_index 
    if src_replica == 1 && length(paths) == 2
        dest_replica = 2
    elseif src_replica == 1
        dest_replica = rand(rng, 2:length(paths))
    else 
        dest_replica = rand(rng, vcat(1:src_replica-1,src_replica+1:length(paths)))
    end 
    # Get the number of kinks on each replica
    num_kinks_src = num_kinks[src_replica] 
    num_kinks_dest = num_kinks[dest_replica]  
    # Randomly choose a swapped site to unswap ????
    site_to_unswap = sub_sites[num_swaps-1+begin] 
    # Get swap kink indices
    # source replica
    next = site_to_unswap 
    while (paths[src_replica][next+begin].dest_replica == paths[src_replica][next+begin].src_replica) 
        next = paths[src_replica][next+begin].next 
    end
    kink_out_of_src = next  
    # destination replica
    next = site_to_unswap; 
    while (paths[dest_replica][next+begin].dest_replica == paths[dest_replica][next+begin].src_replica) 
        next = paths[dest_replica][next+begin].next 
    end
    kink_out_of_dest = next  
    # Get lower and upper bounds of the flat interval on each replica
    prev_src = paths[src_replica][kink_out_of_src+begin].prev 
    next_src = paths[src_replica][kink_out_of_src+begin].next 
    prev_dest = paths[dest_replica][kink_out_of_dest+begin].prev  
    next_dest = paths[dest_replica][kink_out_of_dest+begin].next  
    # Get number of particles to the left and righ of the swap kinks
    n_src_left = paths[src_replica][prev_src+begin].n 
    n_src_right = paths[src_replica][kink_out_of_src+begin].n 
    n_dest_left = paths[dest_replica][prev_dest+begin].n 
    n_dest_right = paths[dest_replica][kink_out_of_dest+begin].n  
    # Can only delete swap kink if pre/post no. of particles is the same
    if (n_src_left != n_src_right)
        return
    end
    if (n_dest_left != n_dest_right)
        return
    end 
    # Stage 1: delete kink coming out of source replica
    # Modify links to kink at end of paths vector that will be swapped
    if (paths[src_replica][num_kinks_src-1+begin].next!=-1)
        paths[src_replica][paths[src_replica][num_kinks_src-1+begin].next+begin].prev = kink_out_of_src 
    end
    paths[src_replica][paths[src_replica][num_kinks_src-1+begin].prev+begin].next = kink_out_of_src 
    
    swap!(paths[src_replica],kink_out_of_src,num_kinks_src-1) 
    # Important kinks might've been swapped. Correct if so.
    if (prev_src == num_kinks_src-1)
        prev_src = kink_out_of_src 
    elseif (next_src == num_kinks_src-1)
        next_src = kink_out_of_src 
    end 
    # Head & tail separately b.c they could've been one of the kinks above too
    if (head_idx[src_replica] == num_kinks_src-1)
        head_idx[src_replica] = kink_out_of_src 
    elseif (tail_idx[src_replica] == num_kinks_src-1)
        tail_idx[src_replica] = kink_out_of_src 
    end 
    # The kink sent to where deleted kink was might be last on it's site
    if (paths[src_replica][kink_out_of_src+begin].next == -1) 
        last_kinks[src_replica][paths[src_replica][kink_out_of_src+begin].src+begin] = kink_out_of_src
    end 
    # Reconnect upper and lower bounds of the flat
    if (next_src != -1)
        paths[src_replica][next_src+begin].prev = prev_src 
    end
    paths[src_replica][prev_src+begin].next = next_src  
    # Lower bound of flat could be the last kink in the site
    if (next_src == -1)
        last_kinks[src_replica][site_to_unswap+begin] = prev_src
    end 
    # Stage 3: delete kink coming out of destination replica
    # Modify links to kink at end of paths vector that will be swapped
    if (paths[dest_replica][num_kinks_dest-1+begin].next != -1)
        paths[dest_replica][paths[dest_replica][num_kinks_dest-1+begin].next+begin].prev = kink_out_of_dest 
    end
    paths[dest_replica][paths[dest_replica][num_kinks_dest-1+begin].prev+begin].next = kink_out_of_dest
    
    swap!(paths[dest_replica],kink_out_of_dest,num_kinks_dest-1)
    
    # Important kinks might've been swapped. Correct if so.
    if (prev_dest == num_kinks_dest-1)
        prev_dest = kink_out_of_dest
    elseif (next_dest == num_kinks_dest-1)
        next_dest = kink_out_of_dest 
    end 
    # Head & tail separately b.c they could've been one of the kinks above too
    if (head_idx[dest_replica] == num_kinks_dest-1)
        head_idx[dest_replica] = kink_out_of_dest 
    elseif (tail_idx[dest_replica] == num_kinks_dest-1)
        tail_idx[dest_replica] = kink_out_of_dest 
    end 
    # The kink sent to where deleted kink was might be last on it's site
    if (paths[dest_replica][kink_out_of_dest+begin].next==-1) 
        last_kinks[dest_replica][paths[dest_replica][kink_out_of_dest+begin].src+begin] = kink_out_of_dest 
    end 
    # Reconnect upper and lower bounds of the flat
    if (next_dest != -1)
        paths[dest_replica][next_dest+begin].prev = prev_dest 
    end
    paths[dest_replica][prev_dest+begin].next = next_dest  
    # Lower bound of flat could be the last kink in the site
    if (next_dest == -1)
        last_kinks[dest_replica][site_to_unswap+begin] = prev_dest 
    end 
    # Modify number of swaps tracker
    sim_state.num_swaps -= 1 
    num_swaps =sim_state.num_swaps 
    # Modify number of kinks trackers for each replica
    num_kinks[src_replica] -= 1 
    num_kinks[dest_replica] -= 1 
    
    step_counters.delete_swap_kink_accepts += 1 
    
    return 
end

function swap_timeshift_head!(rng::AbstractRNG, sim_state::SimState, src_replica_index::Int64, sim_tracker:: SimTracker)
    # Extract some parameters 
    #rng::RandomNumberGenerator = sim_state.rng
    paths::Vector{Path} = sim_state.paths
    step_counters::MCStepCounters = sim_tracker.step_counters
    num_kinks::Vector{Int64} = sim_state.num_kinks 
    t::Float64 = sim_state.t
    U::Float64 = sim_state.U
    N::Int64 = sim_state.N
    mu::Float64 = sim_state.mu
    eta::Float64 = sim_state.eta 
    beta::Float64 = sim_state.beta
    adjacency_matrix::Adjacency_Matrix = sim_state.adjacency_matrix
    num_swaps::Int64 = sim_state.num_swaps
    m_A::Int64 = sim_state.m_A
    sub_sites::Sites = sim_state.sub_sites
    last_kinks::Vector{Vector{Int64}} = sim_state.last_kinks
    head_idx::Vector{Int64} = sim_state.head_idx
    tail_idx::Vector{Int64} = sim_state.tail_idx
    canonical::Bool = sim_state.canonical
    N_tracker::Vector{Float64} = sim_tracker.Ns

    # Need at least two replicas to perform a spaceshift
    if (length(paths) < 2)
        return
    end
    # Need at least one swap kink to perform a timeshift through swap kink
    if (num_swaps == 0)
        return
    end 
    # Retrieve worm head indices. -1 means no worm head
    head_idx_0 = head_idx[0+begin] 
    head_idx_1 = head_idx[1+begin]  
    # There had to be STRICTLY ONE worm head to timeshift over swap kink
    if ( head_idx_0 != -1 && head_idx_1 != -1 )
        return # two worms present
    end
    if ( head_idx_0 == -1 && head_idx_1 == -1 )
        return # no worms present
    end 
    # Choose the "source" and "destination" replica
    if ( head_idx_0 != -1 )
        src_replica = 1 
    else
        src_replica = 2
    end
    dest_replica = 3 - src_replica   
    # Get index of the worm head to be moved
    worm_end_idx = head_idx[src_replica]  
    # Extract worm head time and site
    tau = paths[src_replica][worm_end_idx+begin].tau 
    worm_end_site = paths[src_replica][worm_end_idx+begin].src 
    n = paths[src_replica][worm_end_idx+begin].n  
    # Get lower and upper adjacent kinks of worm head to be moved
    # NOTE: One of the two bounds might be the swap kink.
    prev_src = paths[src_replica][worm_end_idx+begin].prev 
    next_src = paths[src_replica][worm_end_idx+begin].next  
    # Check if worm head is adjacent to a swap kink
    if (next_src!=-1) 
        if (paths[src_replica][next_src+begin].src_replica != paths[src_replica][next_src+begin].dest_replica)
            swap_in_front = true
        elseif (paths[src_replica][prev_src+begin].src_replica != paths[src_replica][prev_src+begin].dest_replica)      
            swap_in_front = false 
        else 
            return
        end 
    else 
        if (paths[src_replica][prev_src+begin].src_replica != paths[src_replica][prev_src+begin].dest_replica)
            swap_in_front = false 
        else 
            return 
        end
    end         
    # Get index of central time slice at destination replica
    current_kink = worm_end_site  # next refers to index of next kink on site
    while (paths[dest_replica][current_kink+begin].dest_replica == paths[dest_replica][current_kink+begin].src_replica) 
        current_kink = paths[dest_replica][current_kink+begin].next 
    end
    kink_out_of_dest = current_kink  
    # Get indices of kinks before & after swap kink in destination replica
    prev_dest = paths[dest_replica][kink_out_of_dest+begin].prev 
    next_dest = paths[dest_replica][kink_out_of_dest+begin].next  
    # Determine the lower and upper bounds of the worm end to be timeshifted
    if (swap_in_front) 
        if (next_dest != -1)
            tau_next = paths[dest_replica][next_dest+begin].tau 
        else
            tau_next = beta 
        end
        tau_prev = paths[src_replica][prev_src+begin].tau  
    else  # swap kink behind
        if (next_src != -1)
            tau_next = paths[src_replica][next_src+begin].tau  
        else
            tau_next = beta 
        end
        tau_prev = paths[dest_replica][prev_dest+begin].tau 
    end 
    # Calculate change in diagonal energy
#    shift_head=true; # we are always moving head in this update. set to true.
#    dV=U*(n-!shift_head)-mu;
    dV = U * n - mu 
    
    # To make acceptance ratio unity,shift tail needs to sample w/ dV=eps-eps_w
#    if (!shift_head){dV *= -1;} # dV=eps-eps_w
    
    #boost::random::uniform_real_distribution<double> rnum(0.0, 1.0);
    # Sample the new time of the worm end from truncated exponential dist.
    #/*:::::::::::::::::::: Truncated Exponential RVS :::::::::::::::::::::::::*/
    Z = 1.0 - exp(-dV*(tau_next-tau_prev)) 
    tau_new = tau_prev - log(1.0-Z*rand(rng))  / dV 
    #/*::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*/
    
    # Check if advance or recede
    if (tau_new > tau) 
        is_advance = true 
        step_counters.swap_advance_head_attempts += 1  
    else 
        is_advance = false 
        step_counters.swap_recede_head_attempts += 1 
    end 
    # Check if timeshift takes the worm end over the swap kink
    is_over_swap = false 
    if ( (is_advance && swap_in_front && (tau_new > beta/2)) || (!is_advance && !swap_in_front && (tau_new<beta/2)) )
         is_over_swap = true
    end 
    # Determine the length of path to be modified in each replica
    if (is_over_swap) 
        l_path_src = beta/2 - tau 
        l_path_dest::Float64 = tau_new - beta/2
    else  
        # did not go over swap
        l_path_src = tau_new - tau  
        l_path_dest  = 0.0 
    end 
    # Determine the total particle change based on wormend to be shifted
    dN_src  = 1.0 * l_path_src/beta 
    dN_dest = 1.0 * l_path_dest/beta  
    # Canonical simulations: Restrict updates to interval N:(N-1,N+1)
    if (canonical) 
        if ((N_tracker[src_replica]+dN_src)   < (N-1)  || (N_tracker[src_replica]+dN_src)   > (N+1)  || (N_tracker[dest_replica]+dN_dest) < (N-1)  || (N_tracker[dest_replica]+dN_dest) > (N+1))
            return 
        end
    end 
    # Get number of particles after: worm end @ src & central kink @ dest
    n_after_worm_end = paths[src_replica][worm_end_idx+begin].n 
    n_after_swap_kink = paths[dest_replica][kink_out_of_dest+begin].n 
    n_before_swap_kink = paths[dest_replica][prev_dest+begin].n  
    # Get number of kinks in source and destination replicas (before update)
    num_kinks_src = num_kinks[src_replica] 
    num_kinks_dest = num_kinks[dest_replica]  
    # Build the Metropolis condition (R)
    R = 1.0  # Sampling worm end time from truncated exponential makes R unity. 
    # Metropolis sampling
    if (rand(rng) < R) 
        if (!is_over_swap)  # worm end does not go over swap kink
            paths[src_replica][worm_end_idx+begin].tau = tau_new 
            N_tracker[src_replica] += dN_src 
            if (is_advance)
                step_counters.swap_advance_head_accepts += 1 
            else 
                step_counters.swap_recede_head_accepts += 1 
            end 
        else  # We go Over Swap 
            if (is_advance)  # advance OVER SWAP

                #/*--------- Deletion of worm end from SOURCE replica ---------*/
                
                # num_kinks_src-1 will be swapped. Modify links to it
                if (paths[src_replica][num_kinks_src-1+begin].next != -1)  
                    paths[src_replica][paths[src_replica][num_kinks_src-1+begin].next+begin].prev = worm_end_idx 
                end
                paths[src_replica][paths[src_replica][num_kinks_src-1+begin].prev+begin].next = worm_end_idx 
                
                swap!(paths[src_replica],worm_end_idx,num_kinks_src-1)
                
                # Upper or lower bound of flat could've been swapped. Correct.
                if (next_src == num_kinks_src - 1)
                    next_src=worm_end_idx 
                elseif (prev_src == num_kinks_src - 1)
                    prev_src = worm_end_idx 
                end 
                # Tail could've been swapped. Correct if so.
                if (tail_idx[src_replica] == num_kinks_src - 1)
                    tail_idx[src_replica] = worm_end_idx 
                end 
                # Whatever kink was swapped could've been the last on its site
                if (paths[src_replica][worm_end_idx+begin].next == -1) 
                    last_kinks[src_replica][paths[src_replica][worm_end_idx+begin].src+begin] = worm_end_idx 
                end 
                # Reconnect upper and lower bounds of the flat
                if (next_src != -1)
                    paths[src_replica][next_src+begin].prev = prev_src 
                end
                paths[src_replica][prev_src+begin].next = next_src  
                # Deactivate the worm end
                head_idx[src_replica] = -1  
                # Update trackers for: no. of active kinks,total particles
                num_kinks[src_replica] -= 1 
                N_tracker[src_replica] += dN_src  
                #/*------- Insertion of worm end in DESTINATION replica -------*/ 
                # Activate first available kink
                #paths[dest_replica][num_kinks_dest+begin] = Kink(tau_new,n_after_swap_kink,worm_end_site,worm_end_site,kink_out_of_dest,next_dest,dest_replica,dest_replica)  
                fill_kink!(paths[dest_replica][num_kinks_dest+begin],tau_new,n_after_swap_kink,worm_end_site,worm_end_site,kink_out_of_dest,next_dest,dest_replica,dest_replica)
                # Save head index
                head_idx[dest_replica] = num_kinks_dest  
                # Update number of particles after swap kink in dest replica
                paths[dest_replica][kink_out_of_dest+begin].n = n_after_swap_kink + 1 
                # Add to acceptance counter
                step_counters.swap_advance_head_accepts += 1  
                # Modify links of swap kink and next_dest kink
                if (next_dest!=-1)
                    paths[dest_replica][next_dest+begin].prev = num_kinks_dest 
                end                
                paths[dest_replica][kink_out_of_dest+begin].next = num_kinks_dest  
                # Update trackers for: no. of active kinks,total particles
                num_kinks[dest_replica] += 1 
                N_tracker[dest_replica] += dN_dest  
                # Created kink might be last on its site
                if (next_dest == -1)
                    last_kinks[dest_replica][worm_end_site+begin] = num_kinks_dest 
                end
            else  
                # Recede OVER SWAP 
                # Cannot recede head over swap if no particles on destination
                if (n_before_swap_kink == 0)
                    return 
                end 
                #/*--------- Deletion of worm end from SOURCE replica ---------*/
                # num_kinks_src-1 will be swapped. Modify links to it
                if (paths[src_replica][num_kinks_src-1+begin].next != -1) 
                    paths[src_replica][paths[src_replica][num_kinks_src-1+begin].next+begin].prev = worm_end_idx 
                end
                paths[src_replica][paths[src_replica][num_kinks_src-1+begin].prev+begin].next = worm_end_idx 
                
                swap!(paths[src_replica],worm_end_idx,num_kinks_src-1)
                
                # Upper or lower bound of flat could've been swapped. Correct.
                if (next_src == num_kinks_src-1)
                    next_src = worm_end_idx 
                elseif (prev_src == num_kinks_src - 1)
                    prev_src = worm_end_idx 
                end 
                # Tail could've been swapped. Correct if so.
                if (tail_idx[src_replica] == num_kinks_src - 1)
                    tail_idx[src_replica] = worm_end_idx
                end 
                # Whatever kink was swapped could've been the last on its site
                if (paths[src_replica][worm_end_idx+begin].next == -1) 
                    last_kinks[src_replica][paths[src_replica][worm_end_idx+begin].src+begin] = worm_end_idx 
                end 
                # Modify particles in prev_src
                paths[src_replica][prev_src+begin].n = n_after_worm_end  
                # Reconnect upper and lower bounds of the flat
                if (next_src != -1)
                    paths[src_replica][next_src+begin].prev = prev_src 
                end
                paths[src_replica][prev_src+begin].next = next_src  
                # Deactivate the worm end
                head_idx[src_replica] = -1    
                # Update trackers for: no. of active kinks,total particles
                num_kinks[src_replica] -= 1 
                N_tracker[src_replica] += dN_src  
                # Swap kink on src might be last kink on it's site
                if (next_src == -1) 
                    last_kinks[src_replica][worm_end_site+begin] = prev_src 
                end 

                #/*------- Insertion of worm end in DESTINATION replica -------*/ 
                # Activate first available kink
                #paths[dest_replica][num_kinks_dest+begin] = Kink(tau_new,n_before_swap_kink-1,worm_end_site,worm_end_site,prev_dest,kink_out_of_dest,dest_replica,dest_replica)  
                fill_kink!(paths[dest_replica][num_kinks_dest+begin],tau_new,n_before_swap_kink-1,worm_end_site,worm_end_site,prev_dest,kink_out_of_dest,dest_replica,dest_replica)
                # Save head index
                head_idx[dest_replica] = num_kinks_dest  
                # Add to acceptance counter
                step_counters.swap_recede_head_accepts += 1  
                # Modify links to worm head in destination replica
                paths[dest_replica][kink_out_of_dest+begin].prev = num_kinks_dest 
                paths[dest_replica][prev_dest+begin].next = num_kinks_dest  
                # Update trackers for: no. of active kinks,total particles
                num_kinks[dest_replica] += 1 
                N_tracker[dest_replica] += dN_dest 
                
            #                cout<<"Receded head over swap AFTER (left/right fock state)"<<endl;
            #                for (int i=0; i<1; i++){
            #                    cout<<paths[src_replica][paths[src_replica][prev_src].prev].n;
            #                }
            #                cout << " || ";
            #                for (int i=0; i<1; i++){
            #                    cout<<paths[src_replica][prev_src].n;
            #                }
            #
            #                cout << "    ";
            #
            #                for (int i=0; i<1; i++){
            #                    cout<<paths[dest_replica][num_kinks_dest].n;
            #                }
            #                cout << " || ";
            #                for (int i=0; i<1; i++){
            #                    cout<<paths[dest_replica][kink_out_of_dest].n;
            #                }
            #
            #                cout << endl;
                            
            #                cout << "2 (recede)" << endl;
            end
        end
        return  
    else # Reject
        return 
    end
end

function swap_timeshift_tail!(rng::AbstractRNG, sim_state::SimState, src_replica_index::Int64, sim_tracker:: SimTracker)
    # Extract some parameters 
    #rng::RandomNumberGenerator = sim_state.rng
    paths::Vector{Path} = sim_state.paths
    step_counters::MCStepCounters = sim_tracker.step_counters
    num_kinks::Vector{Int64} = sim_state.num_kinks 
    t::Float64 = sim_state.t
    U::Float64 = sim_state.U
    N::Int64 = sim_state.N
    mu::Float64 = sim_state.mu
    eta::Float64 = sim_state.eta 
    beta::Float64 = sim_state.beta
    adjacency_matrix::Adjacency_Matrix = sim_state.adjacency_matrix
    num_swaps::Int64 = sim_state.num_swaps
    m_A::Int64 = sim_state.m_A
    sub_sites::Sites = sim_state.sub_sites
    last_kinks::Vector{Vector{Int64}} = sim_state.last_kinks
    head_idx::Vector{Int64} = sim_state.head_idx
    tail_idx::Vector{Int64} = sim_state.tail_idx
    canonical::Bool = sim_state.canonical	
    N_tracker::Vector{Float64} = sim_tracker.Ns

    # Need at least two replicas to perform a spaceshift
    if (length(paths) < 2)
        return
    end 
    # Need at least one swap kink to perform a timeshift through swap kink
    if (num_swaps==0)
        return
    end 
    # Retrieve worm tail indices. -1 means no worm tail
    tail_idx_0 = tail_idx[1] 
    tail_idx_1 = tail_idx[2]  
    # There had to be STRICTLY ONE worm tail to timeshift over swap kink
    if (tail_idx_0 != -1 && tail_idx_1 != -1)
        return  # two worms present
    end
    if (tail_idx_0 == -1 && tail_idx_1 == -1)
        return # no worms present
    end 
    # Choose the "source" and "destination" replica
    if (tail_idx_0 != -1)
        src_replica = 1
    else
        src_replica = 2 
    end
    dest_replica = 3 - src_replica  
    # Get index of the worm tail to be moved
    worm_end_idx = tail_idx[src_replica]  
    # Extract worm tail time and site
    tau = paths[src_replica][worm_end_idx+begin].tau 
    worm_end_site = paths[src_replica][worm_end_idx+begin].src 
    n = paths[src_replica][worm_end_idx+begin].n 
    # Get lower and upper adjacent kinks of worm tail to be moved
    # NOTE: One of the two bounds might be the swap kink.
    prev_src = paths[src_replica][worm_end_idx+begin].prev  
    next_src = paths[src_replica][worm_end_idx+begin].next  
    # Check if worm tail is adjacent to a swap kink
    if (next_src!=-1) 
        if (paths[src_replica][next_src+begin].src_replica != paths[src_replica][next_src+begin].dest_replica)
            swap_in_front = true 
        elseif (paths[src_replica][prev_src+begin].src_replica != paths[src_replica][prev_src+begin].dest_replica)
            swap_in_front = false 
        else 
            return
        end 
    else 
        if (paths[src_replica][prev_src+begin].src_replica != paths[src_replica][prev_src+begin].dest_replica)
            swap_in_front = false
        else 
            return
        end
    end         
    # Get index of central time slice at destination replica
    current_kink = worm_end_site  # next refers to index of next kink on site
    while (paths[dest_replica][current_kink+begin].dest_replica == paths[dest_replica][current_kink+begin].src_replica) 
        current_kink = paths[dest_replica][current_kink+begin].next 
    end
    kink_out_of_dest = current_kink  
    # Get indices of kinks before & after swap kink in destination replica
    prev_dest = paths[dest_replica][kink_out_of_dest+begin].prev 
    next_dest = paths[dest_replica][kink_out_of_dest+begin].next 
    # Determine the lower and upper bounds of the worm end to be timeshifted
    if (swap_in_front) 
        if (next_dest != -1)
            tau_next = paths[dest_replica][next_dest+begin].tau 
        else
            tau_next = beta 
        end
        tau_prev = paths[src_replica][prev_src+begin].tau  
    else  # swap kink behind
        if (next_src != -1)
            tau_next = paths[src_replica][next_src+begin].tau 
        else
            tau_next = beta 
        end
        tau_prev = paths[dest_replica][prev_dest+begin].tau 
    end
    
    # Calculate change in diagonal energy
    #    shift_tail=true; # we are always moving tail in this update. set to true.
    #    dV=U*(n-!shift_head)-mu;
    dV = U * (n-1) - mu 
    
    # To make acceptance ratio unity,shift tail needs to sample w/ dV=eps-eps_w
    dV *= -1  # dV=eps-eps_w
    
    #boost::random::uniform_real_distribution<double> rnum(0.0, 1.0);
    # Sample the new time of the worm end from truncated exponential dist.
    #/*:::::::::::::::::::: Truncated Exponential RVS :::::::::::::::::::::::::*/
    Z = 1.0 - exp(-dV*(tau_next-tau_prev)) 
    tau_new = tau_prev - log(1.0-Z*rand(rng))  / dV 
    #/*::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*/
    
    # Check if advance or recede
    if (tau_new > tau) 
        is_advance = true 
        step_counters.swap_advance_tail_attempts += 1  
    else  
        is_advance = false 
        step_counters.swap_recede_tail_attempts += 1 
    end 
    # Check if timeshift takes the worm end over the swap kink
    is_over_swap = false 
    if ( (is_advance && swap_in_front && (tau_new>beta/2)) || (!is_advance && !swap_in_front && (tau_new<beta/2)) )
        is_over_swap = true
    end 
    # Determine the length of path to be modified in each replica
    if (is_over_swap) 
        l_path_src = beta/2 - tau 
        l_path_dest::Float64 =  tau_new - beta/2  
    else  
        # did not go over swap
        l_path_src = tau_new - tau 
        l_path_dest = 0.0 
    end 
    # Determine the total particle change based on wormend to be shifted
    dN_src  = -1.0 * l_path_src/beta 
    dN_dest = -1.0 * l_path_dest/beta 
    # Canonical simulations: Restrict updates to interval N:(N-1,N+1)
    if (canonical) 
        if ((N_tracker[src_replica]+dN_src)   < (N-1)  ||  (N_tracker[src_replica]+dN_src)   > (N+1)  || (N_tracker[dest_replica]+dN_dest) < (N-1)  || (N_tracker[dest_replica]+dN_dest) > (N+1))
            return 
        end
    end 
    # Get number of particles after: worm end @ src & central kink @ dest
    n_after_worm_end = paths[src_replica][worm_end_idx+begin].n 
    n_after_swap_kink = paths[dest_replica][kink_out_of_dest+begin].n 
    n_before_swap_kink = paths[dest_replica][prev_dest+begin].n  
    # Get number of kinks in source and destination replicas (before update)
    num_kinks_src = num_kinks[src_replica] 
    num_kinks_dest = num_kinks[dest_replica]  
    # Build the Metropolis condition (R)
    R = 1.0; # Sampling worm end time from truncated exponential makes R unity. 
    # Metropolis sampling
    if (rand(rng) < R) 
        if (!is_over_swap)  # worm end does not go over swap kink
            paths[src_replica][worm_end_idx+begin].tau = tau_new 
            N_tracker[src_replica] += dN_src 
            if (is_advance)
                step_counters.swap_advance_tail_accepts += 1
            else 
                step_counters.swap_recede_tail_accepts += 1
            end 
        else  # We go Over Swap

            if (is_advance)  # advance OVER SWAP
                
                # Cannot advance tail over swap if no particles on destination
                if (n_after_swap_kink == 0)
                    return
                end
                #/*--------- Deletion of worm end from SOURCE replica ---------*/
                
                # num_kinks_src-1 will be swapped. Modify links to it
                if (paths[src_replica][num_kinks_src-1+begin].next != -1) 
                    paths[src_replica][paths[src_replica][num_kinks_src-1+begin].next+begin].prev = worm_end_idx 
                end
                paths[src_replica][paths[src_replica][num_kinks_src-1+begin].prev+begin].next = worm_end_idx 
                
                swap!(paths[src_replica],worm_end_idx,num_kinks_src-1)
                
                # Upper or lower bound of flat could've been swapped. Correct.
                if (next_src == num_kinks_src - 1)
                    next_src = worm_end_idx
                elseif (prev_src == num_kinks_src - 1)
                    prev_src = worm_end_idx
                end 
                # Head could've been swapped. Correct if so.
                if (head_idx[src_replica] == num_kinks_src - 1)
                    head_idx[src_replica] = worm_end_idx 
                end 
                # Whatever kink was swapped could've been the last on its site
                if (paths[src_replica][worm_end_idx+begin].next == -1)
                    last_kinks[src_replica][paths[src_replica][worm_end_idx+begin].src+begin] = worm_end_idx 
                end 
                # Reconnect upper and lower bounds of the flat
                if (next_src != -1)
                    paths[src_replica][next_src+begin].prev = prev_src 
                end
                paths[src_replica][prev_src+begin].next = next_src  
                # Deactivate the worm end
                tail_idx[src_replica] = -1    
                # Update trackers for: no. of active kinks,total particles
                num_kinks[src_replica] -= 1 
                N_tracker[src_replica] += dN_src  
                #/*------- Insertion of worm end in DESTINATION replica -------*/
                
                # Activate first available kink
                #paths[dest_replica][num_kinks_dest+begin] = Kink(tau_new,n_after_swap_kink,worm_end_site,worm_end_site,kink_out_of_dest,next_dest,dest_replica,dest_replica); 
                fill_kink!(paths[dest_replica][num_kinks_dest+begin],tau_new,n_after_swap_kink,worm_end_site,worm_end_site,kink_out_of_dest,next_dest,dest_replica,dest_replica)
                # Save tail index
                tail_idx[dest_replica] = num_kinks_dest  
                # Update number of particles after swap kink in dest replica
                paths[dest_replica][kink_out_of_dest+begin].n=n_after_swap_kink-1;
                
                # Add to acceptance counter
                step_counters.swap_advance_tail_accepts+=1;
                
                # Modify links of swap kink and next_dest kink
                if (next_dest!=-1)
                    paths[dest_replica][next_dest+begin].prev=num_kinks_dest;
                end
                paths[dest_replica][kink_out_of_dest+begin].next=num_kinks_dest;
                
                # Update trackers for: no. of active kinks,total particles
                num_kinks[dest_replica] += 1;
                N_tracker[dest_replica] += dN_dest;

                # Created kink might be last on its site
                if (next_dest==-1)
                    last_kinks[dest_replica][worm_end_site+begin] = num_kinks_dest 
                end
                                 
            else  # Recede OVER SWAP
                
                #/*--------- Deletion of worm end from SOURCE replica ---------*/
                # num_kinks_src-1 will be swapped. Modify links to it
                if (paths[src_replica][num_kinks_src-1+begin].next != -1) 
                    paths[src_replica][paths[src_replica][num_kinks_src-1+begin].next+begin].prev = worm_end_idx 
                end
                paths[src_replica][paths[src_replica][num_kinks_src-1+begin].prev+begin].next = worm_end_idx 
                
                swap!(paths[src_replica],worm_end_idx,num_kinks_src-1) 
                
                # Upper or lower bound of flat could've been swapped. Correct.
                if (next_src == num_kinks_src-1)
                    next_src = worm_end_idx
                elseif (prev_src == num_kinks_src-1)
                    prev_src = worm_end_idx
                end 
                # Head could've been swapped. Correct if so.
                if (head_idx[src_replica] == num_kinks_src-1)
                    head_idx[src_replica] = worm_end_idx 
                end 
                # Whatever kink was swapped could've been the last on its site
                if (paths[src_replica][worm_end_idx+begin].next == -1) 
                  last_kinks[src_replica][paths[src_replica][worm_end_idx+begin].src+begin] = worm_end_idx 
                end 
                # Modify particles in prev_src
                paths[src_replica][prev_src+begin].n = n_after_worm_end  
                # Reconnect upper and lower bounds of the flat
                if (next_src != -1)
                    paths[src_replica][next_src+begin].prev = prev_src 
                end
                paths[src_replica][prev_src+begin].next = next_src   
                # Deactivate the worm end
                tail_idx[src_replica] = -1  
                # Update trackers for: no. of active kinks,total particles
                num_kinks[src_replica] -= 1 
                N_tracker[src_replica] += dN_src  
                # Swap kink on src might be last kink on it's site
                if (next_src == -1) 
                    last_kinks[src_replica][worm_end_site+begin] = prev_src 
                end
                 
                #/*------- Insertion of worm end in DESTINATION replica -------*/ 
                # Activate first available kink
                #paths[dest_replica][num_kinks_dest+begin] = Kink(tau_new,n_before_swap_kink+1,worm_end_site,worm_end_site,prev_dest,kink_out_of_dest,dest_replica,dest_replica)  
                fill_kink!(paths[dest_replica][num_kinks_dest+begin], tau_new,n_before_swap_kink+1,worm_end_site,worm_end_site,prev_dest,kink_out_of_dest,dest_replica,dest_replica)
                # Save tail index
                tail_idx[dest_replica] = num_kinks_dest   
                # Add to acceptance counter
                step_counters.swap_recede_tail_accepts += 1   
                # Modify links to worm head in destination replica
                paths[dest_replica][kink_out_of_dest+begin].prev = num_kinks_dest 
                paths[dest_replica][prev_dest+begin].next = num_kinks_dest  
                # Update trackers for: no. of active kinks,total particles
                num_kinks[dest_replica] += 1 
                N_tracker[dest_replica] += dN_dest 
            end
        end
        return  
    else # Reject
        return  
    end
end 

function random_swap_update!(rng::AbstractRNG, sim_state::SimState, src_replica_index::Int64, sim_tracker:: SimTracker)
    # select and run a single random update
    # possible_updates = [
    #     insert_swap_kink!,
    #     delete_swap_kink!,
    #     swap_timeshift_head!,
    #     swap_timeshift_tail!]
    # selected_update! = rand(sim_state.rng, possible_updates) 
    # selected_update!(sim_state, src_replica_index, sim_tracker)
    
    i = rand(rng, 1:4)
    if (i == 1)
        insert_swap_kink!(rng, sim_state, src_replica_index, sim_tracker)
    elseif (i == 2)
        delete_swap_kink!(rng, sim_state, src_replica_index, sim_tracker)
    elseif (i == 3)
        swap_timeshift_head!(rng, sim_state, src_replica_index, sim_tracker)
    elseif (i == 4)
        swap_timeshift_tail!(rng, sim_state, src_replica_index, sim_tracker)
    end

end

function random_swap_update_print!(rng::AbstractRNG, sim_state::SimState, src_replica_index::Int64, sim_tracker:: SimTracker)
    # select and run a single random update
    # possible_updates = [
    #     insert_swap_kink!,
    #     delete_swap_kink!,
    #     swap_timeshift_head!,
    #     swap_timeshift_tail!]
    # selected_update! = rand(sim_state.rng, possible_updates) 
    # selected_update!(sim_state, src_replica_index, sim_tracker)
    
    i = rand(rng, 1:4)
    if (i == 1)
        println("insert_swap_kink!")
        insert_swap_kink!(rng, sim_state, src_replica_index, sim_tracker)
    elseif (i == 2)
        println("delete_swap_kink!")
        delete_swap_kink!(rng, sim_state, src_replica_index, sim_tracker)
    elseif (i == 3)
        println("swap_timeshift_head!")
        swap_timeshift_head!(rng, sim_state, src_replica_index, sim_tracker)
    elseif (i == 4)
        println("swap_timeshift_tail!")
        swap_timeshift_tail!(rng, sim_state, src_replica_index, sim_tracker)
    end

end
