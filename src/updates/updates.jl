

@inline function random_mc_update!(rng::AbstractRNG, sim_state::SimState, replica_index::Int64, sim_tracker:: SimTracker) 
    #select and run a single random update  
    i = rand(rng, 1:15) 
    
    if (i == 1) 
        insert_worm_new!(rng, sim_state, replica_index, sim_tracker)
    elseif (i == 2) 
        delete_worm_new!(rng, sim_state, replica_index, sim_tracker)
    elseif (i == 3)    
        insert_zero_new!(rng, sim_state, replica_index, sim_tracker)
    elseif (i == 4) 
        delete_zero_new!(rng, sim_state, replica_index, sim_tracker) 
    elseif (i == 5) 
        insert_beta_new!(rng, sim_state, replica_index, sim_tracker)
    elseif (i == 6) 
        delete_beta_new!(rng, sim_state, replica_index, sim_tracker)
    elseif (i == 7) 
        timeshift_new!(rng, sim_state, replica_index, sim_tracker)
    elseif (i == 8) 
        insert_kink_before_head_new!(rng, sim_state, replica_index, sim_tracker)
    elseif (i == 9) 
        delete_kink_before_head_new!(rng, sim_state, replica_index, sim_tracker)
    elseif (i == 10) 
        insert_kink_after_head_new!(rng, sim_state, replica_index, sim_tracker)
    elseif (i == 11) 
        delete_kink_after_head_new!(rng, sim_state, replica_index, sim_tracker)
    elseif (i == 12) 
        insert_kink_before_tail_new!(rng, sim_state, replica_index, sim_tracker)
    elseif (i == 13) 
        delete_kink_before_tail_new!(rng, sim_state, replica_index, sim_tracker)
    elseif (i == 14) 
        insert_kink_after_tail_new!(rng, sim_state, replica_index, sim_tracker)
    elseif (i == 15)       
        delete_kink_after_tail_new!(rng, sim_state, replica_index, sim_tracker)
    end 
    sim_tracker.num_preq_updates += 1

end  

@inline function insert_worm_new!(rng::AbstractRNG, sim_state::SimState, replica_index::Int64, sim_tracker:: SimTracker)::Nothing
    # Can only perform update if there are no worm ends
    if (sim_state.head_idx[replica_index] != -1 || sim_state.tail_idx[replica_index] != -1)
        return nothing
    end
    # Extract some parameters
    step_counters = sim_tracker.step_counters
    N = sim_state.N
    eta = sim_state.eta  
    # Randomly sample a kink and the flat interval to its next 
    kink_below, kink_below_index, tau_prev, tau_next = sample_flat_interval(rng, sim_state, replica_index)   
    # Randomly choose to insert worm or antiworm
    is_worm = random_bool(rng) 
    # Determine the no. of particles after each worm end
    if (is_worm) 
        step_counters.insert_worm_attempts += 1 
        n_tail = kink_below.n + 1 
        n_head = kink_below.n  
    else 
        step_counters.insert_anti_attempts += 1 
        n_tail = kink_below.n 
        n_head = kink_below.n - 1 
    end 
    # Reject update if illegal worm insertion is proposed
    if (kink_below.n == 0 && !(is_worm))
        step_counters.insert_anti_attempts -= 1 
        return
    end 
    # Calculate the difference in diagonal energy dV = \epsilon_w - \epsilon
    dV = worm_energy_difference(sim_state.model, sim_state, n_tail, n_head, is_worm)
    # Truncated exponential sampling 
    tau_t, tau_h, Z, plausible = sample_joint_truncated_exponential_from_a(rng, tau_prev, tau_next, dV)
    !plausible && return
    if !is_worm
        # anti-worm, tau_t>tau_h
        tau_t, tau_h = tau_h, tau_t
    end
    # Determine length of modified path and particle change 
    l_path::Float64 = tau_h - tau_t 
    dN::Float64 = l_path / sim_state.beta  
    # Canonical simulations: Restrict updates to interval N:(N-1,N+1)
    if (sim_state.canonical)
        if ((sim_tracker.Ns[replica_index] + dN) < (N-1) || (sim_tracker.Ns[replica_index] + dN) > (N+1))
            return
        end
    end 
    # Build the Metropolis ratio (R)
    p_dw = 0.5
    p_iw = 0.5
    num_kinks = sim_state.num_kinks[replica_index]
    R = eta^2 * n_tail * Z * num_kinks * (p_dw/p_iw) * 2  
    # Metropolis sampling
    if (rand(rng) < R) # Accept 
        # Activate the first two available kinks
        if (is_worm) 
            # for worm, head is above tail 
            tail_idx = insert_tail!(sim_state,replica_index,n_tail,tau_t, kink_below_index)
            insert_head!(sim_state,replica_index,n_head,tau_h, tail_idx) 
            # Add to Acceptance counter
            step_counters.insert_worm_accepts += 1; 
        else  # Antiworm
            # for anti-worm, tail is above head 
            head_idx = insert_head!(sim_state,replica_index,n_head,tau_h, kink_below_index) 
            insert_tail!(sim_state,replica_index,n_tail,tau_t, head_idx) 
            # Add to Acceptance counter
            step_counters.insert_anti_accepts += 1
        end 
        # Update trackers for total particles 
        sim_tracker.Ns[replica_index] += dN  
        return  
    else # Reject
        return
    end
end

@inline function delete_worm_new!(rng::AbstractRNG, sim_state::SimState, replica_index::Int64, sim_tracker:: SimTracker)::Nothing 
    # Can only propose worm deletion if both worm ends are present
    if (sim_state.head_idx[replica_index] == -1 || sim_state.tail_idx[replica_index] == -1)
        return
    end  
    # Extract some parameters
    paths = sim_state.paths[replica_index]
    step_counters = sim_tracker.step_counters
    N = sim_state.N
    eta = sim_state.eta  
    head_idx = sim_state.head_idx[replica_index]
    tail_idx = sim_state.tail_idx[replica_index]
    # Can only delete worm if wormends are on same flat interval
    if (head_idx != paths[tail_idx+begin].prev && tail_idx != paths[head_idx+begin].prev)
        return
    end
    # Extract worm end attributes
    tau_h::Float64 = paths[head_idx+begin].tau  
    n_head::Int64 = paths[head_idx+begin].n  
    tau_t::Float64 = paths[tail_idx+begin].tau  
    n_tail::Int64 = paths[tail_idx+begin].n  
    # Identify the type of worm
    is_worm = tau_h > tau_t  
    # Identify lower and upper bound of flat interval where worm lives
    if (is_worm) 
        step_counters.delete_worm_attempts += 1
        # For worm, tail is below
        tau_prev = paths[paths[tail_idx+begin].prev+begin].tau
        tau_next = paths[head_idx+begin].next == -1 ? sim_state.beta : paths[paths[head_idx+begin].next+begin].tau 
    else  # antiworm
        step_counters.delete_anti_attempts += 1
        # For anti-worm, head is below
        tau_prev = paths[paths[head_idx+begin].prev+begin].tau
        tau_next = paths[tail_idx+begin].next == -1 ? sim_state.beta : paths[paths[tail_idx+begin].next+begin].tau  
    end 
    # Determine length of modified path and particle change
    l_path::Float64 = tau_h - tau_t 
    dN::Float64 = -l_path / sim_state.beta 
    # Canonical simulations: Restrict updates to interval N:(N-1,N+1)
    if (sim_state.canonical)
        if ((sim_tracker.Ns[replica_index] + dN) < (N-1) || (sim_tracker.Ns[replica_index] + dN) > (N+1))
            return 
        end
    end
    # Calculate the difference in diagonal energy dV = \epsilon_w - \epsilon
    dV = worm_energy_difference(sim_state.model, sim_state, n_tail, n_head, is_worm)
    # Compute normalization constant of joint distribution
    Z = Z_joint(tau_prev, tau_next, dV)
    # Build the Metropolis ratio (R)
    p_dw = 0.5
    p_iw = 0.5 
    num_kinks = sim_state.num_kinks[replica_index]
    R = eta * eta * n_tail * Z * num_kinks * (p_dw/p_iw) * 2 
    R = 1.0/R 
    # Metropolis sampling 
    if (rand(rng) < R) # Accept 
        # Add to Acceptance counter
        if (is_worm)
            step_counters.delete_worm_accepts += 1
        else
            step_counters.delete_anti_accepts += 1
        end
        # Delete tail and head kinks
        delete_kink_pair!(sim_state,replica_index,sim_state.tail_idx[replica_index],sim_state.head_idx[replica_index]) 
        # Adjust particle number tracker
        sim_tracker.Ns[replica_index] += dN
        
        return  
    else # Reject
        return
    end
end

@inline function insert_zero_new!(rng::AbstractRNG, sim_state::SimState, replica_index::Int64, sim_tracker:: SimTracker)::Nothing
    # Cannot insert if there's two worm ends present
    if (sim_state.head_idx[replica_index] != -1 && sim_state.tail_idx[replica_index] != -1)
        return
    end 
    # Extract some parameters
    paths = sim_state.paths[replica_index]
    step_counters = sim_tracker.step_counters
    N = sim_state.N
    M = sim_state.M
    eta = sim_state.eta  
    head_idx = sim_state.head_idx[replica_index]
    tail_idx = sim_state.tail_idx[replica_index] 
    # Randomly select site on which to insert worm/antiworm from tau=0 
    i = rand(rng, 0:M-1)   
    # Extract attributes of insertion flat
    n = paths[i+begin].n 
    # Determine the length of insertion flat interval
    next = paths[i+begin].next
    tau_flat = next == -1 ? sim_state.beta : paths[next+begin].tau  
    # Choose worm/antiworm insertion based on worm ends present
    if (head_idx == -1 && tail_idx == -1)
        # no worm ends present
        if (n==0) 
            # can only insert worm, not antiworm
            is_worm = true
            p_type = 1.0
        else
            # choose worm or antiworm insertion with equal probability
            is_worm = random_bool(rng) 
            p_type = 0.5
        end  
    elseif (head_idx != -1)
        # only worm head present, can insert antiworm only
        if (n==0)
            step_counters.insertZero_anti_attempts += 1
            return # cannot insert proposed antiworm, no particles present
        else
            is_worm = false
            p_type = 1.0
        end
    else # only tail present, can insert worm only
        is_worm = true
        p_type = 1.0
    end  
    # delete_zero (reverse update) might've had to choose either head or tail
    if (head_idx == -1 && tail_idx == -1) 
        # only one end after insertZero
        p_wormend = 1.0
    else
        # two worm ends after insertZero
        if (is_worm)
            if (paths[paths[tail_idx+begin].prev+begin].tau > DELTA_TAU/2)
                p_wormend = 1.0 # cannot choose tail. was not coming from tau=0.
            else
                p_wormend = 0.5 # delete_zero could choose either head or tail
            end
        else 
            # if insert anti (i.e, a tail) the end present was a head
            if (paths[paths[head_idx+begin].prev+begin].tau > DELTA_TAU/2)
                p_wormend = 1.0
            else
                p_wormend = 0.5
            end
        end
    end
    # Determine the number of particles after each worm end
    if (is_worm)
        step_counters.insertZero_worm_attempts += 1
        n_tail = n + 1
        n_head = n
    else
        step_counters.insertZero_anti_attempts += 1
        n_tail = n
        n_head = n - 1
    end
    n_tail == 0 && return # R will be zero
    # Calculate the diagonal energy difference dV = \epsilon_w - \epsilon
    dV = worm_energy_difference(sim_state.model, sim_state, n_tail, n_head, is_worm) 
    # Randomly choose where to insert worm end on the flat interval
    tau_new, Z, plausible = sample_truncated_exponential_from_a(rng, 0.0, tau_flat, dV)
    !plausible && return 
    # Determine the length of the path to be modified
    l_path = tau_new 
    # Determine the total particle change based on worm type
    dN = is_worm ? l_path/sim_state.beta : -l_path/sim_state.beta  
    # Canonical simulations: Restrict updates to interval N:(N-1,N+1)
    if (sim_state.canonical)
        if ((sim_tracker.Ns[replica_index] + dN) < (N-1) || (sim_tracker.Ns[replica_index] + dN) > (N+1))
            return
        end
    end   
    # Build the Metropolis Ratio (R)
    C = get_trial_state_factor(n_tail,is_worm,sim_state.trial_state,sim_state,sim_tracker,replica_index,"insert_zero") 
    p_dz = 0.5
    p_iz = 0.5
    R = eta * sqrt(n_tail) * C * (p_dz/p_iz) * M * p_wormend * Z / p_type  

    # Metropolis sampling
    if (rand(rng) < R)
        # Accept
        
        # Activate the first available kink
        if (is_worm)
            insert_head!(sim_state, replica_index, n_head, tau_new, i) 
            # Update the number of particles in the initial kink at site i
            paths[i+begin].n = n_tail
            # Add to Acceptance counter
            step_counters.insertZero_worm_accepts += 1
            # Worm inserted, add one to tau=0 particle tracker
            sim_tracker.N_zero[replica_index] += 1
        else 
            # antiworm
            insert_tail!(sim_state, replica_index, n_tail, tau_new, i) 
            # Update number of particles in initial kink of insertion site
            paths[i+begin].n = n_head
            # Add to Acceptance counter
            step_counters.insertZero_anti_accepts += 1
            # Antiworm inserted, subtract one from tau=0 particle tracker
            sim_tracker.N_zero[replica_index] -= 1
        end 
        # Update tracker for total particles  
        sim_tracker.Ns[replica_index] += dN 

        return
    else # Reject
        return
    end
end

@inline function delete_zero_new!(rng::AbstractRNG, sim_state::SimState, replica_index::Int64, sim_tracker:: SimTracker)::Nothing
    # Cannot delete if there are no worm ends present
    if (sim_state.head_idx[replica_index] == -1 && sim_state.tail_idx[replica_index] == -1)
        return
    end
    # Extract some parameters 
    #rng::RandomNumberGenerator = sim_state.rng
    paths::Path = sim_state.paths[replica_index]
    step_counters::MCStepCounters = sim_tracker.step_counters 
    N::Int64 = sim_state.N
    M::Int64 = sim_state.M 
    eta::Float64 = sim_state.eta
    head_idx::Int64 = sim_state.head_idx[replica_index]
    tail_idx::Int64 = sim_state.tail_idx[replica_index] 
    # Cannot delete if there are no worm ends coming from tau=0
    if (head_idx != -1 && tail_idx != -1)
        if (paths[paths[head_idx+begin].prev+begin].tau > DELTA_TAU/2 && paths[paths[tail_idx+begin].prev+begin].tau > DELTA_TAU/2)
            return
        end 
    elseif (head_idx != -1) # only head present
        if (paths[paths[head_idx+begin].prev+begin].tau > DELTA_TAU/2)
            return
        end
    else # only tail present
        if (paths[paths[tail_idx+begin].prev+begin].tau > DELTA_TAU/2)
            return
        end 
    end
    # Decide which worm end to delete
    if (head_idx!=-1 && tail_idx!=-1) # both wormends present
        if (paths[paths[head_idx+begin].prev+begin].tau < DELTA_TAU/2 && paths[paths[tail_idx+begin].prev+begin].tau < DELTA_TAU/2) 
            # both near 0
            delete_head::Bool = random_bool(rng)
            p_wormend = 0.5
        elseif (paths[paths[head_idx+begin].prev+begin].tau < DELTA_TAU/2)
            delete_head = true
            p_wormend = 1.0
        else # only the tail is near zero
            delete_head = false
            p_wormend = 1.0
        end
    elseif (head_idx != -1) 
        # only head present
        delete_head = true
        p_wormend = 1.0
    else # only tail present
        delete_head = false
        p_wormend = 1.0
    end 
    # Get index of worm end to be deleted
    worm_end_idx = delete_head ? head_idx : tail_idx  
    # Extract particle number
    n = paths[worm_end_idx+begin].n 
    # Calculate the length of the flat interval (excluding the wormend)
    next = paths[worm_end_idx+begin].next
    tau_next = (next == -1) ? sim_state.beta : paths[next+begin].tau 
    # No. of particles before,after the worm end to be deleted
    if (delete_head) # delete worm
        n_tail = n+1
    else #  delete antiworm
        n_tail = n
    end
    n_head = n_tail-1 
    # Worm insert (reverse update) probability of choosing worm or antiworm
    if ( head_idx != -1 && tail_idx != -1) # worm end present before insertion
        p_type = 1.0
    else # no worm ends present before insertion
        if (n==0)
            p_type = 1.0 # only worm can be inserted if no particles on flat
        else
            p_type = 0.5
        end
    end
    # Add to deleteZero PROPOSAL counters
    if (delete_head) 
        # delete head (delete worm)
        step_counters.deleteZero_worm_attempts += 1
    else                   
        # delete tail (delete antiworm)
        step_counters.deleteZero_anti_attempts += 1
    end
    # Determine the length of path to be modified
    l_path =  paths[worm_end_idx+begin].tau 
    # Determine the total particle change based on worm type
    dN = delete_head ? -l_path / sim_state.beta : l_path / sim_state.beta 
    # Canonical simulations: Restrict updates to interval N:(N-1,N+1)
    if (sim_state.canonical)
        if ((sim_tracker.Ns[replica_index] + dN) < (N-1) || (sim_tracker.Ns[replica_index] + dN) > (N+1))
            return
        end
    end 
    # Calculate diagonal energy difference
    dV = worm_energy_difference(sim_state.model, sim_state, n_tail, n_head, delete_head) 
    # Build the Metropolis Ratio  (R)
    C = get_trial_state_factor(n_tail,delete_head,sim_state.trial_state,sim_state,sim_tracker,replica_index,"delete_zero") 
    Z = Z_single(0.0, tau_next, dV)
    p_dz = 0.5
    p_iz = 0.5
    R = eta * sqrt(n_tail) * C * (p_dz/p_iz) * M * p_wormend * Z / p_type   
    R = 1.0/R
    
    # Metropolis sampling
    if (rand(rng) < R) 
        # accept
        # Update the number of particles in the initial kink of worm end site
        paths[paths[worm_end_idx+begin].prev+begin].n = n
        # delete kink
        delete_kink!(sim_state,replica_index,worm_end_idx)
        # Increment counters and particle number tracker
        if (delete_head) 
            step_counters.deleteZero_worm_accepts += 1
            # Worm deleted, subtract one to tau=0 particle tracker
            sim_tracker.N_zero[replica_index] -= 1
        else  
            step_counters.deleteZero_anti_accepts += 1
            # Antiworm deleted, add one to tau=0 particle tracker
            sim_tracker.N_zero[replica_index] += 1
        end
        # Update tracker for total particles 
        sim_tracker.Ns[replica_index] += dN
        
        return
    else # reject
        return
    end
end

@inline function insert_beta_new!(rng::AbstractRNG, sim_state::SimState, replica_index::Int64, sim_tracker:: SimTracker)::Nothing 
    # Cannot insert if there's two worm ends present
    if (sim_state.head_idx[replica_index] != -1 && sim_state.tail_idx[replica_index] != -1)
        return
    end
    # Extract some parameters 
    paths::Path = sim_state.paths[replica_index]
    step_counters::MCStepCounters = sim_tracker.step_counters
    N::Int64 = sim_state.N
    M::Int64 = sim_state.M
    eta::Float64 = sim_state.eta
    head_idx::Int64 = sim_state.head_idx[replica_index]
    tail_idx::Int64 = sim_state.tail_idx[replica_index]
    last_kinks::Vector{Int64} = sim_state.last_kinks[replica_index]
    # Randomly select site on which to insert worm/antiworm from tau=beta 
    i = rand(rng,0:M-1)  
    # Extract the flat interval where insertion is proposed & its attributes
    tau_prev = paths[last_kinks[i+begin]+begin].tau
    n = paths[last_kinks[i+begin]+begin].n 
    # Choose worm/antiworm insertion based on worm ends present
    if (head_idx == -1 && tail_idx == -1) 
        # no worm ends present
        if (n==0) 
            # can only insert worm, not antiworm
            is_worm = true
            p_type = 1.0
        else
            # choose worm or antiworm insertion with equal probability
            is_worm = random_bool(rng)
            p_type = 0.5
        end
    elseif (tail_idx != -1) # only worm tail present, can insert antiworm only
        if (n==0)
            step_counters.insertBeta_anti_attempts += 1.0
            # cannot insert proposed antiworm, no particles present
            return 
        else
            is_worm = false
            p_type = 1.0
        end
    else 
        # only head present, can insert worm only
        is_worm = true
        p_type = 1.0
    end 
    # Add to worm/antiworm insertion attempt counters
    if (is_worm)
        step_counters.insertBeta_worm_attempts += 1
    else 
        step_counters.insertBeta_anti_attempts += 1
    end 
    # Determine the no. of particles after each worm end
    if (is_worm)
        n_tail = n + 1
        n_head = n
    else
        n_tail = n
        n_head = n - 1
    end
    # Calculate the diagonal energy difference dV = \epsilon_w - \epsilon
    dV = worm_energy_difference(sim_state.model, sim_state, n_tail, n_head, is_worm)  
    # Sample time on flat interval from truncated exponential for insertion 
    tau_new, Z, plausible = sample_truncated_exponential_from_b(rng, tau_prev, sim_state.beta, dV)
    !plausible && return
    # deleteBeta (reverse update) might've had to choose either head or tail
    if (head_idx == -1 && tail_idx == -1) 
        # only one end after insertZero
        p_wormend = 1.0
    else                          
        # two worm ends after insertZero
        if (is_worm)
            if (paths[head_idx+begin].next != -1)
                p_wormend = 1.0 # cannot choose head.was not coming from beta.
            else
                p_wormend = 0.5 # deleteBeta could choose either head or tail
            end
        else # if insert anti (i.e, a head) the end present was a tail
            if (paths[tail_idx+begin].next != -1)
                p_wormend = 1.0
            else
                p_wormend = 0.5
            end
        end
    end
    # Determine the length of the path to be modified
    l_path = sim_state.beta - tau_new
    # Determine the total particle change based on worm type
    dN = is_worm ? l_path/sim_state.beta : -l_path/sim_state.beta 
    # Canonical simulations: Restrict updates to interval N:(N-1,N+1)
    if (sim_state.canonical)
        if ((sim_tracker.Ns[replica_index] + dN) < (N-1) || (sim_tracker.Ns[replica_index] + dN) > (N+1))
            return
        end
    end   
    # Build the Metropolis Ratio (R)
    C = get_trial_state_factor(n_tail,is_worm,sim_state.trial_state,sim_state,sim_tracker,replica_index,"insert_beta") 
    # Compared to the C++ code, Z here corresponds to Z/dV    
    p_db = 0.5
    p_ib = 0.5
    R = eta * sqrt(n_tail) * C * (p_db/p_ib) * M * p_wormend * Z / p_type  
    # Metropolis sampling 
    if (rand(rng) < R)
        # Accept
        # Activate the first available kink
        if (is_worm)
            insert_tail!(sim_state,replica_index,n_tail,tau_new,last_kinks[i+begin]) 
            # Add to Acceptance counter
            step_counters.insertBeta_worm_accepts += 1
            # Worm inserted, add one to tau=beta particle tracker
            sim_tracker.N_beta[replica_index] += 1
        else # antiworm
            insert_head!(sim_state,replica_index,n_head,tau_new,last_kinks[i+begin])  
            # Add to Acceptance counter
            step_counters.insertBeta_anti_accepts += 1 
            # Antiworm inserted, subtract one to tau=beta particle tracker
            sim_tracker.N_beta[replica_index] -= 1 
        end 
        # Update tracker for total particles 
        sim_tracker.Ns[replica_index] += dN  
        return
    else # Reject
        return
    end
end

@inline function delete_beta_new!(rng::AbstractRNG, sim_state::SimState, replica_index::Int64, sim_tracker:: SimTracker)::Nothing
    # Extract some parameters  
    paths::Path = sim_state.paths[replica_index]
    step_counters::MCStepCounters = sim_tracker.step_counters 
    N::Int64 = sim_state.N
    M::Int64 = sim_state.M 
    eta::Float64 = sim_state.eta 
    head_idx::Int64 = sim_state.head_idx[replica_index]
    tail_idx::Int64 = sim_state.tail_idx[replica_index] 
    # Cannot delete if there are no worm ends present
    if (head_idx == -1 && tail_idx == -1)
        return
    end
    # Cannot delete if there are no worm ends coming from tau=beta
    if (head_idx != -1 && tail_idx != -1)
        if (paths[head_idx+begin].next != -1 && paths[tail_idx+begin].next != -1)
            return
        end
    elseif (head_idx!=-1) # only head present
        if (paths[head_idx+begin].next != -1)
            return
        end 
    else # only tail present
        if (paths[tail_idx+begin].next != -1)
            return
        end
    end 
    # Decide which worm end to delete 
    if (head_idx != -1 && tail_idx != -1) 
        # both wormends present
        if (paths[head_idx+begin].next == -1 && paths[tail_idx+begin].next == -1) 
            # both last
            delete_head = random_bool(rng)
            p_wormend = 0.5
        elseif (paths[head_idx+begin].next == -1)
            delete_head = true
            p_wormend = 1.0
        else # only the tail is near zero
            delete_head = false
            p_wormend = 1.0
        end 
    elseif (head_idx != -1) # only head present
        delete_head = true
        p_wormend = 1.0
    else # only tail present
        delete_head = false
        p_wormend = 1.0
    end
    # Get index of worm end to be deleted
    worm_end_idx = delete_head ? head_idx : tail_idx 
    # Extract worm end attributes
    tau = paths[worm_end_idx+begin].tau
    n = paths[worm_end_idx+begin].n 
    prev = paths[worm_end_idx+begin].prev 
    # Calculate the length of the flat interval (excluding the worm end)
    tau_prev = paths[prev+begin].tau 
    # No. of particles before,after the worm end to be deleted
    if (delete_head) 
        # delete antiworm
        n_tail = n+1
    else #  delete worm
        n_tail = n
    end
    n_head = n_tail - 1
    # Worm insert (reverse update) probability of choosing worm or antiworm
    if ( head_idx != -1 && tail_idx != -1) 
        # worm end present before insertion
        p_type = 1.0
    else
        # no worm ends present before insertion
        if (paths[paths[worm_end_idx+begin].prev+begin].n == 0)
            p_type = 1.0 # only worm can be inserted if no particles on flat
        else
            p_type = 0.5
        end
    end 
    # Add to deleteBeta PROPOSAL counters
    if (delete_head) 
        # delete antiworm
        step_counters.deleteBeta_anti_attempts += 1
    else                   
        # delete worm
        step_counters.deleteBeta_worm_attempts += 1
    end
    # Determine the length of path to be modified
    l_path = sim_state.beta - tau
    # Determine the total particle change based on worm type
    dN = delete_head ? l_path/sim_state.beta : -l_path/sim_state.beta 
    # Canonical simulations: Restrict updates to interval N:(N-1,N+1)
    if (sim_state.canonical)
        if ((sim_tracker.Ns[replica_index] + dN) < (N-1) || (sim_tracker.Ns[replica_index] + dN) > (N+1))
            return
        end
    end
    # Calculate diagonal energy difference
    dV = worm_energy_difference(sim_state.model, sim_state, n_tail, n_head, !delete_head) 
    # Build the Metropolis Ratio  (R)
    C = get_trial_state_factor(n_tail,!delete_head,sim_state.trial_state,sim_state,sim_tracker,replica_index,"delete_beta")
    Z = Z_single(tau_prev, sim_state.beta, dV)
    # Metropolis ratio 
    p_db = 0.5
    p_ib = 0.5
    R = eta * sqrt(n_tail) * C * (p_db/p_ib) * M * p_wormend * Z / p_type  
    R = 1.0/R
    # Metropolis sampling
    if (rand(rng) < R)
        # accept    
        delete_kink!(sim_state,replica_index,worm_end_idx)  
        # Increment acceptance counters
        if (delete_head) 
            step_counters.deleteBeta_anti_accepts += 1;
            # Antiworm deleted, add one to tau=beta particle tracker
            sim_tracker.N_beta[replica_index] += 1
        else 
            step_counters.deleteBeta_worm_accepts += 1
            # Worm deleted, subtracts one to tau=beta particle tracker
            sim_tracker.N_beta[replica_index] -= 1
        end 
        # Update tracker for total particles
        sim_tracker.Ns[replica_index] += dN
        
        return;
    else # reject
        return
    end
        
end

@inline function timeshift_new!(rng::AbstractRNG, sim_state::SimState, replica_index::Int64, sim_tracker:: SimTracker)::Nothing
    # Reject update if there is no worm end present
    if (sim_state.head_idx[replica_index] == -1 && sim_state.tail_idx[replica_index] == -1)
        return
    end
    # Extract some parameters 
    paths::Path = sim_state.paths[replica_index]
    step_counters::MCStepCounters = sim_tracker.step_counters
    N::Int64 = sim_state.N
    head_idx::Int64 = sim_state.head_idx[replica_index]
    tail_idx::Int64 = sim_state.tail_idx[replica_index]
    N_tracker = sim_tracker.Ns[replica_index]  
    # Choose which worm end to move 
    if (head_idx != -1 && tail_idx != -1)
        # both worm ends present
        # Randomly choose to shift HEAD or TAIL
        shift_head = random_bool(rng)
    elseif (head_idx != -1)
        # only head present
        shift_head = true
    else 
        # only tail present
        shift_head = false
    end
    # Save the kink index of the end that will be shifted
    if (shift_head)
        worm_end_idx = head_idx
    else 
        worm_end_idx = tail_idx
    end
    # Extract worm end attributes
    tau = paths[worm_end_idx+begin].tau
    n = paths[worm_end_idx+begin].n
    prev = paths[worm_end_idx+begin].prev
    next = paths[worm_end_idx+begin].next
    # Diagonal energy difference in simplified form
    dV::Float64 = timeshift_energy_difference(sim_state.model, sim_state, n, shift_head)
    # Determine the lower and upper bounds of the worm end to be timeshifted
    tau_next = (next == -1) ? sim_state.beta : paths[next+begin].tau 
    tau_prev = paths[prev+begin].tau
    # Sample the new time of the worm end from truncated exponential dist.  
    tau_new, Z, plausible = sample_truncated_exponential_from_a(rng, tau_prev, tau_next, dV) 
    !plausible && return
    # Add to PROPOSAL counter
    if (shift_head) 
        if (tau_new > tau)
            step_counters.advance_head_attempts += 1
        else
            step_counters.recede_head_attempts += 1
        end 
    else  # shift tail
        if (tau_new > tau)
            step_counters.advance_tail_attempts += 1
        else
            step_counters.recede_tail_attempts += 1
        end
    end
    # Determine the length of path to be modified
    l_path = tau_new - tau
    # Determine the total particle change based on wormend to be shifted
    dN = shift_head ? l_path/sim_state.beta : -l_path/sim_state.beta 
    # Canonical simulations: Restrict updates to interval N:(N-1,N+1)
    if (sim_state.canonical)
        if ((N_tracker + dN) < (N-1) || (N_tracker + dN) > (N+1))
            return
        end
    end 
    # Add to ACCEPTANCE counter
    if (shift_head)
        if (tau_new > tau)
            step_counters.advance_head_accepts += 1
        else
            step_counters.recede_head_accepts += 1
        end
    else 
        # shift tail
        if (tau_new > tau)
            step_counters.advance_tail_accepts += 1
        else
            step_counters.recede_tail_accepts += 1
        end
    end 
    # Modify the worm end time
    paths[worm_end_idx+begin].tau = tau_new 
    # Modify total particle number tracker
    sim_tracker.Ns[replica_index] += dN

    return  
end

@inline function insert_kink_before_head_new!(rng::AbstractRNG, sim_state::SimState, replica_index::Int64, sim_tracker:: SimTracker)::Nothing
    # Update only possible if worm head present
    if (sim_state.head_idx[replica_index] == -1)
        return
    end 
    # Extract some parameters  
    paths::Path = sim_state.paths[replica_index]
    step_counters::MCStepCounters = sim_tracker.step_counters 
    head_idx::Int64 = sim_state.head_idx[replica_index] 
    adjacency_matrix::Adjacency_Matrix = sim_state.adjacency_matrix 
    # Add to proposal counter
    step_counters.ikbh_attempts += 1
    # Extract the worm head site and replica
    i = paths[head_idx+begin].src
    # Randomly choose a nearest neighbor site
    j = get_random_neighbor(rng, adjacency_matrix, i)
    p_site = 1.0/get_neighbor_count(adjacency_matrix, i)
    # Retrieve the time of the worm head
    tau_h = paths[head_idx+begin].tau
    # Determine index of lower/upper kink of flat where head is (site i)
    prev_i = paths[head_idx+begin].prev 
    # Determine index of lower kink of flat where head jumps to (site j) 
    prev_j = find_kink_below_tau_on_site(sim_state, replica_index, tau_h, j)
    # Determine upper,lower bound times on both sites (upper time not needed)
    tau_prev_i = paths[prev_i+begin].tau
    tau_prev_j = paths[prev_j+begin].tau
    # Determine lowest time at which kink could've been inserted
    tau_min = (tau_prev_i > tau_prev_j) ? tau_prev_i : tau_prev_j 
    # Extract no. of particles in the flats adjacent to the new kink
    n_wi = paths[prev_i+begin].n
    n_i = n_wi - 1
    n_j = paths[prev_j+begin].n
    n_wj = n_j + 1                 # "w": segment with the extra particle
    # Calculate the diagonal energy difference on both sites
    dV = add_kink_energy_difference(sim_state.model, sim_state, n_wi, n_i, n_wj, n_j)
    # Truncated Sampling
    tau_kink, Z, plausible = sample_truncated_exponential_from_b(rng, tau_min, tau_h, dV)
    !plausible && return
    # Metropolis ratio 
    p_dkbh = 0.5
    p_ikbh = 0.5
    R = sim_state.t * n_wj * (p_dkbh/p_ikbh) * Z / p_site
    # Metropolis Sampling
    if (rand(rng) < R) 
        # Accept 
        # Add to acceptance counter
        step_counters.ikbh_accepts += 1   
        # Add kink pair and move head from i to j 
        move_kink_to_other_site!(sim_state,replica_index,sim_state.head_idx[replica_index],n_j,prev_j) 
        insert_pair_of_kinks!(sim_state,replica_index,n_i,n_wj,tau_kink,prev_i,prev_j)
        return    
    else # Reject
        return
    end
end

@inline function delete_kink_before_head_new!(rng::AbstractRNG, sim_state::SimState, replica_index::Int64, sim_tracker:: SimTracker)::Nothing
    # Update only possible if worm head present
    if (sim_state.head_idx[replica_index] == -1)
        return
    end 
    # Extract some parameters  
    paths::Path = sim_state.paths[replica_index]
    step_counters::MCStepCounters = sim_tracker.step_counters  
    head_idx::Int64 = sim_state.head_idx[replica_index]  
    adjacency_matrix::Adjacency_Matrix = sim_state.adjacency_matrix 
    # There has to be a regular kink before the worm head
    if (paths[paths[head_idx+begin].prev+begin].src == paths[paths[head_idx+begin].prev+begin].dest)
        return
    end  
    # Indices of: upper bound kink, kink before head, lower bound kink ; site j 
    kink_idx_j = paths[head_idx+begin].prev 
    prev_j = paths[kink_idx_j+begin].prev  
    # Times of: worm head, kink before head, lower bound kink; site j
    tau_h = paths[head_idx+begin].tau  
    tau_prev_j = paths[prev_j+begin].tau  
    # Only kinks in which the particle hops from i TO j can be deleted
    if ( paths[kink_idx_j+begin].n - paths[prev_j+begin].n < 0 )
        return
    end 
    # Retrieve worm head connecting site (i) 
    i = paths[kink_idx_j+begin].dest  
    # Determine index of lower/upper bounds of flat where kink connects to (i)  
    kink_idx_i = paths[kink_idx_j+begin].partner 
    prev_i = paths[kink_idx_i+begin].prev
    next_i = paths[kink_idx_i+begin].next 
    # Retrieve time of lower,upper bounds on connecting site (i)
    tau_prev_i = paths[prev_i+begin].tau 
    tau_next_i = (next_i != -1) ? paths[next_i+begin].tau : sim_state.beta 
    # Deletion cannot interfere w/ kinks on other site
    (tau_h >= tau_next_i) && return 
    # Add to proposal counter
    step_counters.dkbh_attempts += 1  
    # Determine lowest time at which kink could've been inserted
    tau_min = (tau_prev_i > tau_prev_j) ? tau_prev_i : tau_prev_j 
    # Probability of inverse move (ikbh) of choosing site where worm end is
    p_site = 1.0/get_neighbor_count(adjacency_matrix, i)  
    # Extract no. of particles in the flats adjacent to the new kink
    n_wi = paths[prev_i+begin].n 
    n_i = n_wi - 1
    n_j = paths[prev_j+begin].n 
    n_wj = n_j + 1                   # "w": segment with the extra particle 
    # Calculate the diagonal energy difference on both sites 
    dV::Float64 = add_kink_energy_difference(sim_state.model, sim_state, n_wi, n_i, n_wj, n_j)
    # inverse move (insert kink before head) truncated sampling
    Z = Z_single(tau_min, tau_h, dV)
    # Build the Metropolis ratio (R)
    p_dkbh = 0.5 
    p_ikbh = 0.5 
    R = sim_state.t * n_wj * (p_dkbh/p_ikbh) * Z / p_site
    R = 1.0/R 
    # Metropolis Sampling 
    if (rand(rng) < R)
        # Accept
        # Add to acceptance counter
        step_counters.dkbh_accepts += 1
        # move head over and delete the kinks 
        move_kink_to_other_site!(sim_state,replica_index,head_idx,n_i,kink_idx_i)
        delete_kink_pair!(sim_state,replica_index,kink_idx_i,kink_idx_j) 
        return 
    else 
        # Reject
        return 
    end
end

@inline function insert_kink_after_head_new!(rng::AbstractRNG, sim_state::SimState, replica_index::Int64, sim_tracker:: SimTracker)::Nothing
    # Update only possible if worm head present
    if (sim_state.head_idx[replica_index] == -1)
        return
    end  
    # Extract some parameters  
    paths::Path = sim_state.paths[replica_index]
    step_counters::MCStepCounters = sim_tracker.step_counters 
    head_idx::Int64 = sim_state.head_idx[replica_index] 
    adjacency_matrix::Adjacency_Matrix = sim_state.adjacency_matrix 
    # Add to proposal counter
    step_counters.ikah_attempts += 1
    # Extract the worm head site
    i = paths[head_idx+begin].src  
    # Randomly choose a nearest neighbor site 
    j = get_random_neighbor(rng, adjacency_matrix, i)
    p_site = 1.0/get_neighbor_count(adjacency_matrix, i)  
    # Retrieve the time of the worm head
    tau_h = paths[head_idx+begin].tau  
    # Determine index of lower/upper kinks of flat where head is (site i)
    prev_i = paths[head_idx+begin].prev 
    next_i = paths[head_idx+begin].next 
    # Determine index of lower/upper kinks of flat where head jumps to (site j) 
    prev_j = find_kink_below_tau_on_site(sim_state, replica_index, tau_h, j)
    next_j = paths[prev_j+begin].next
    # Determine upper,lower bound times on both sites
    tau_next_i = (next_i != -1) ? paths[next_i+begin].tau : sim_state.beta 
    tau_next_j = (next_j != -1) ? paths[next_j+begin].tau : sim_state.beta  
    # Determine highest time at which kink could've been inserted
    tau_max = (tau_next_i < tau_next_j) ? tau_next_i : tau_next_j 
    # Extract no. of particles in the flats adjacent to the new kink
    n_wi = paths[prev_i+begin].n 
    n_i = n_wi - 1 
    n_wj = paths[prev_j+begin].n 
    n_j = n_wj - 1                    # "w": segment with the extra particle 
    # Update not possible if no particles on destination site (j)
    (n_wj == 0) && return  
    # Calculate the diagonal energy difference on both sites (note, first j then i for relative minus sign,  dV_i - dV_j) 
    dV::Float64 = add_kink_energy_difference(sim_state.model, sim_state, n_wj, n_j, n_wi, n_i)
    # Truncated Sampling
    tau_kink, Z, plausible = sample_truncated_exponential_from_a(rng, tau_h, tau_max, dV)
    !plausible && return
    # Build the Metropolis ratio (R)
    p_dkah = 0.5 
    p_ikah = 0.5 
    R = sim_state.t * n_wj * (p_dkah/p_ikah) * Z / p_site
    # Metropolis Sampling
    if (rand(rng) < R) 
        # Accept  
        move_kink_to_other_site!(sim_state,replica_index,head_idx,n_j,prev_j)
        insert_pair_of_kinks!(sim_state, replica_index, n_i, n_wj, tau_kink, prev_i, sim_state.head_idx[replica_index])
        # Add to acceptance counter
        step_counters.ikah_accepts += 1  
        return
    else 
        # Reject
        return 
    end
end

@inline function delete_kink_after_head_new!(rng::AbstractRNG, sim_state::SimState, replica_index::Int64, sim_tracker:: SimTracker)::Nothing
    # Update only possible if worm head present
    if sim_state.head_idx[replica_index] == -1
        return
    end
    # Extract some parameters  
    paths::Path = sim_state.paths[replica_index]
    step_counters::MCStepCounters = sim_tracker.step_counters  
    head_idx::Int64 = sim_state.head_idx[replica_index]
    tail_idx::Int64 = sim_state.tail_idx[replica_index] 
    adjacency_matrix::Adjacency_Matrix = sim_state.adjacency_matrix
    # There has to be a regular kink after the worm head
    if (paths[head_idx+begin].next == tail_idx || paths[head_idx+begin].next == -1)
        return
    end  
    # Indices of: upper bound kink, kink before head, lower bound kink ; site j
    kink_idx_j = paths[head_idx+begin].next 
    next_j = paths[kink_idx_j+begin].next 
    prev_j = paths[head_idx+begin].prev  
    # Times of: worm head, kink before head, lower bound kink; site j
    tau_next_j = (next_j != -1) ? paths[next_j+begin].tau : sim_state.beta  
    tau_h = paths[head_idx+begin].tau  
    # Only kinks in which the particle hops from i TO j can be deleted
    if (paths[kink_idx_j+begin].n - paths[head_idx+begin].n < 0)
        return
    end 
    # Retrieve worm head connecting site (i) 
    i = paths[kink_idx_j+begin].dest  
    # Determine index of lower/upper bounds of flat where kink connects to (i)  
    kink_idx_i = paths[kink_idx_j+begin].partner 
    prev_i = paths[kink_idx_i+begin].prev
    next_i = paths[kink_idx_i+begin].next  
    # Retrieve time of lower,upper bounds on connecting site (i)
    tau_prev_i = paths[prev_i+begin].tau 
    tau_next_i = (next_i != -1) ? paths[next_i+begin].tau : sim_state.beta 
    # Deletion cannot interfere w/ kinks on other site
    (tau_h <= tau_prev_i) && return
    # Add to proposal counter
    step_counters.dkah_attempts += 1 
    # Determine highest time at which kink could've been inserted
    tau_max = (tau_next_i < tau_next_j) ? tau_next_i : tau_next_j 
    # Probability of inverse move (ikah) choosing site where worm end is
    p_site = 1.0/get_neighbor_count(adjacency_matrix, i)  
    # Extract no. of particles in the flats adjacent to the new kink
    n_wi = paths[prev_i+begin].n
    n_i = n_wi - 1
    n_wj = paths[prev_j+begin].n
    n_j = n_wj - 1                   # "w": segment with the extra particle
    # Calculate the diagonal energy difference on both sites (note, first j then i for dV_i-dV_j) 
    dV::Float64 = add_kink_energy_difference(sim_state.model, sim_state, n_wj, n_j, n_wi, n_i)  
    # inverse move (insert kink after head) tuncated sampling
    Z = Z_single(tau_h, tau_max, dV)
    # Build the Metropolis ratio (R)
    p_dkah = 0.5
    p_ikah = 0.5
    R =  sim_state.t * n_wj * (p_dkah/p_ikah) * Z / p_site
    R = 1.0/R
    # Metropolis Sampling
    if (rand(rng) < R) 
        # Accept 
        move_kink_to_other_site!(sim_state,replica_index,sim_state.head_idx[replica_index],n_i,prev_i)
        delete_kink_pair!(sim_state,replica_index,kink_idx_i,kink_idx_j)
        # Add to acceptance counter
        step_counters.dkah_accepts += 1 

        return  
    else # Reject
        return
    end 
end

@inline function insert_kink_before_tail_new!(rng::AbstractRNG, sim_state::SimState, replica_index::Int64, sim_tracker:: SimTracker)::Nothing 
    # Update only possible if worm tail present
    if (sim_state.tail_idx[replica_index] == -1)
        return
    end 
    # Extract some parameters  
    paths::Path = sim_state.paths[replica_index]
    step_counters::MCStepCounters = sim_tracker.step_counters
    tail_idx::Int64 = sim_state.tail_idx[replica_index]
    adjacency_matrix::Adjacency_Matrix = sim_state.adjacency_matrix
    # Extract the worm tail site
    i = paths[tail_idx+begin].src   
    # Randomly choose a nearest neighbor site 
    j = get_random_neighbor(rng, adjacency_matrix, i)
    p_site = 1.0/get_neighbor_count(adjacency_matrix, i) 
    # Retrieve the time of the worm tail
    tau_t = paths[tail_idx+begin].tau  
    # Determine index of lower/upper kinks of flat where tail is (site i)
    prev_i = paths[tail_idx+begin].prev  
    # Determine index of lower/upper kinks of flat where tail jumps to (site j) 
    prev_j = find_kink_below_tau_on_site(sim_state, replica_index, tau_t, j)
    # Determine upper,lower bound times on both sites
    tau_prev_i = paths[prev_i+begin].tau
    tau_prev_j = paths[prev_j+begin].tau
    # Determine lowest time at which kink could've been inserted
    tau_min = (tau_prev_i > tau_prev_j) ? tau_prev_i : tau_prev_j 
    # Extract no. of particles in the flats adjacent to the new kink
    n_i = paths[prev_i+begin].n 
    n_wi = n_i + 1 
    n_wj = paths[prev_j+begin].n 
    n_j = n_wj - 1                    # "w": segment with the extra particle
    # Update not possible if no particles on destinaton site (j)
    (n_wj == 0) && return
    # Add to proposal counter
    step_counters.ikbt_attempts += 1  
    # Calculate the diagonal energy difference on both sites (dV_i - dV_j) 
    dV = add_kink_energy_difference(sim_state.model, sim_state, n_wj, n_j, n_wi, n_i)
    # Truncated Sampling
    tau_kink, Z, plausible = sample_truncated_exponential_from_b(rng, tau_min, tau_t, dV)
    !plausible && return
    # Build the Metropolis ratio (R)
    p_dkbt = 0.5 
    p_ikbt = 0.5 
    R = sim_state.t * n_wj * (p_dkbt/p_ikbt) * Z / p_site 
    # Metropolis Sampling
    if (rand(rng) < R)
        # Accept
        move_kink_to_other_site!(sim_state,replica_index,tail_idx,n_wj,prev_j)
        insert_pair_of_kinks!(sim_state,replica_index,n_wi,n_j,tau_kink,prev_i,prev_j)
        # Add to acceptance counter
        step_counters.ikbt_accepts += 1 
        
        return 
 
    else # Reject
        return 
    end
end

@inline function delete_kink_before_tail_new!(rng::AbstractRNG, sim_state::SimState, replica_index::Int64, sim_tracker:: SimTracker)::Nothing
    # Update only possible if worm tail present
    if ( sim_state.tail_idx[replica_index] == -1 )
        return
    end 
    # Extract some parameters  
    paths::Path = sim_state.paths[replica_index]
    step_counters::MCStepCounters = sim_tracker.step_counters  
    head_idx::Int64 = sim_state.head_idx[replica_index]
    tail_idx::Int64 = sim_state.tail_idx[replica_index] 
    adjacency_matrix::Adjacency_Matrix = sim_state.adjacency_matrix 
    # There has to be a regular kink after the worm tail
    if (paths[tail_idx+begin].prev == head_idx || paths[paths[tail_idx+begin].prev+begin].tau == 0)
        return
    end  
    # Indices of: upper bound kink, kink before tail, lower bound kink ; site j 
    kink_idx_j = paths[tail_idx+begin].prev 
    prev_j = paths[kink_idx_j+begin].prev  
    # Times of: worm tail, kink before tail, lower bound kink; site j
    tau_t = paths[tail_idx+begin].tau  
    tau_prev_j = paths[prev_j+begin].tau  
    # Only kinks in which the particle hops from j TO i can be deleted
    if (paths[kink_idx_j+begin].n - paths[prev_j+begin].n > 0)
        return
    end 
    # Retrieve worm tail connecting site (i) 
    i = paths[kink_idx_j+begin].dest   
    # Determine index of lower/upper bounds of flat where kink connects to (i)  
    kink_idx_i = paths[kink_idx_j+begin].partner
    prev_i = paths[kink_idx_i+begin].prev
    next_i = paths[kink_idx_i+begin].next 
    # Retrieve time of lower,upper bounds on connecting site (i)
    tau_prev_i = paths[prev_i+begin].tau 
    tau_next_i = (next_i == -1) ? sim_state.beta : paths[next_i+begin].tau  
    # Deletion cannot interfere w/ kinks on other site
    (tau_t >= tau_next_i) && return 
    # Add to proposal counter
    step_counters.dkbt_attempts += 1  
    # Determine lowest time at which kink could've been inserted
    tau_min = (tau_prev_i > tau_prev_j) ? tau_prev_i : tau_prev_j 
    # Probability of inverse move (ikbt) choosing site where worm end is
    p_site = 1.0/get_neighbor_count(adjacency_matrix, i)  
    # Extract no. of particles in the flats adjacent to the new kink
    n_i = paths[prev_i+begin].n 
    n_wi = n_i + 1 
    n_wj = paths[prev_j+begin].n 
    n_j = n_wj - 1                   # "w": segment with the extra particle 
    # Calculate the diagonal energy difference on both sites (dV_i - dV_j ) 
    dV = add_kink_energy_difference(sim_state.model, sim_state, n_wj, n_j, n_wi, n_i)
    # inverse move (insert kink before tail) truncated exponential sampling
    Z = Z_single(tau_min, tau_t, dV)
    # Build the Metropolis ratio (R)
    p_dkbt = 0.5 
    p_ikbt = 0.5 
    R = sim_state.t * n_wj * (p_dkbt/p_ikbt) * Z /p_site
    R = 1.0/R  
    # Metropolis Sampling 
    if (rand(rng) < R)
        # Accept
        move_kink_to_other_site!(sim_state,replica_index,tail_idx,n_wi,kink_idx_i)
        delete_kink_pair!(sim_state,replica_index,kink_idx_i,kink_idx_j)
        # Add to acceptance counter
        step_counters.dkbt_accepts += 1   
        
        return  
    else # Reject
        return 
    end
end

@inline function insert_kink_after_tail_new!(rng::AbstractRNG, sim_state::SimState, replica_index::Int64, sim_tracker:: SimTracker)::Nothing
    # Update only possible if worm tail present
    if (sim_state.tail_idx[replica_index] == -1)
        return
    end 
    # Extract some parameters  
    paths::Path = sim_state.paths[replica_index]
    step_counters::MCStepCounters = sim_tracker.step_counters 
    tail_idx::Int64 = sim_state.tail_idx[replica_index] 
    adjacency_matrix::Adjacency_Matrix = sim_state.adjacency_matrix 
    # Add to proposal counter
    step_counters.ikat_attempts += 1  
    # Extract the worm tail site
    i = paths[tail_idx+begin].src  
    # Randomly choose a nearest neighbor site 
    j = get_random_neighbor(rng, adjacency_matrix, i)
    p_site = 1.0/get_neighbor_count(adjacency_matrix, i)  
    # Retrieve the time of the worm tail
    tau_t = paths[tail_idx+begin].tau 
    # Determine index of lower/upper kinks of flat where tail is (site i)
    prev_i = paths[tail_idx+begin].prev 
    next_i = paths[tail_idx+begin].next  
    # Determine index of lower/upper kinks of flat where tail jumps to (site j)   
    prev_j = find_kink_below_tau_on_site(sim_state, replica_index, tau_t, j)
    next_j = paths[prev_j+begin].next
    # Determine upper,lower bound times on both sites
    tau_next_i = (next_i != -1) ? paths[next_i+begin].tau : sim_state.beta 
    tau_next_j = (next_j != -1) ? paths[next_j+begin].tau : sim_state.beta 
    # Determine highest time at which kink could've been inserted
    tau_max = (tau_next_i < tau_next_j) ? tau_next_i : tau_next_j 
    # Extract no. of particles in the flats adjacent to the new kink
    n_i = paths[prev_i+begin].n 
    n_wi = n_i + 1 
    n_j = paths[prev_j+begin].n 
    n_wj = n_j + 1                    # "w": segment with the extra particle 
    # Calculate the diagonal energy difference on both sites 
    dV = add_kink_energy_difference(sim_state.model, sim_state, n_wi, n_i, n_wj, n_j)  
    # Truncated Sampling
    tau_kink, Z, plausible = sample_truncated_exponential_from_a(rng, tau_t, tau_max, dV)
    !plausible && return
    # Build the Metropolis ratio (R)
    p_dkat = 0.5 
    p_ikat = 0.5 
    R = sim_state.t * n_wj * (p_dkat/p_ikat) * Z / p_site
    # Metropolis Sampling
    if (rand(rng) < R) 
        # Accept 
        move_kink_to_other_site!(sim_state,replica_index,tail_idx,n_wj,prev_j)
        insert_pair_of_kinks!(sim_state,replica_index,n_wi,n_j,tau_kink,prev_i,sim_state.tail_idx[replica_index])
        # Add to acceptance counter
        step_counters.ikat_accepts += 1   
        
        return  
    else # Reject
        return 
    end
end

@inline function delete_kink_after_tail_new!(rng::AbstractRNG, sim_state::SimState, replica_index::Int64, sim_tracker:: SimTracker)::Nothing
    # Update only possible if worm tail present
    if (sim_state.tail_idx[replica_index] == -1)
        return
    end
    # Extract some parameters  
    paths::Path = sim_state.paths[replica_index]
    step_counters::MCStepCounters = sim_tracker.step_counters 
    head_idx::Int64 = sim_state.head_idx[replica_index]
    tail_idx::Int64 = sim_state.tail_idx[replica_index] 
    adjacency_matrix::Adjacency_Matrix = sim_state.adjacency_matrix 
    # There has to be a regular kink after the worm tail
    if (paths[tail_idx+begin].next == head_idx || paths[tail_idx+begin].next == -1)
        return
    end 
    # Indices of: upper bound kink, kink before tail, lower bound kink ; site j
    kink_idx_j = paths[tail_idx+begin].next
    next_j = paths[kink_idx_j+begin].next
    prev_j = paths[tail_idx+begin].prev
    # Times of: worm tail, kink before tail, lower bound kink; site j
    tau_next_j = (next_j != -1) ? paths[next_j+begin].tau : sim_state.beta  
    tau_t = paths[tail_idx+begin].tau  
    # Only kinks in which the particle hops from j TO i can be deleted
    if ((paths[kink_idx_j+begin].n - paths[tail_idx+begin].n) > 0) 
        return # check if ever met ????
    end
    # Retrieve worm tail site (j) and connecting site (i) 
    i = paths[kink_idx_j+begin].dest   
    # Determine index of lower/upper bounds of flat where kink connects to (i)  
    kink_idx_i = paths[kink_idx_j+begin].partner 
    prev_i = paths[kink_idx_i+begin].prev
    next_i = paths[kink_idx_i+begin].next 
    # Retrieve time of lower,upper bounds on connecting site (i)
    tau_prev_i = paths[prev_i+begin].tau 
    tau_next_i = (next_i == -1) ? sim_state.beta : paths[next_i+begin].tau   
    # Deletion cannot interfere w/ kinks on other site 
    (tau_t <= tau_prev_i) && return 
    # Add to proposal counter
    step_counters.dkat_attempts += 1  
    # Determine highest time at which kink could've been inserted 
    tau_next_min = (tau_next_i < tau_next_j) ? tau_next_i : tau_next_j 
    # Probability of inverse move (ikah) choosing site where worm end is
    p_site = 1.0/get_neighbor_count(adjacency_matrix, i)  
    # Extract no. of particles in the flats adjacent to the new kink
    n_i = paths[prev_i+begin].n 
    n_wi = n_i + 1 
    n_j = paths[prev_j+begin].n 
    n_wj = n_j + 1                    # "w": segment with the extra particle 
    # Calculate the diagonal energy difference on both sites
    dV = add_kink_energy_difference(sim_state.model, sim_state, n_wi, n_i, n_wj, n_j)  
    # inverse move (insert kink after tail) truncated sampling
    Z = Z_single(tau_t, tau_next_min, dV)
    # Build the Metropolis ratio (R)
    p_dkat = 0.5 
    p_ikat = 0.5  
    R = sim_state.t * n_wj * (p_dkat/p_ikat) * Z / p_site
    R = 1.0/R  
    # Metropolis Sampling 
    if (rand(rng) < R)
        # Accept
        # Add to acceptance counter
        step_counters.dkat_accepts += 1
        # 1: move the worm tail over from j to i 
	


        move_kink_to_other_site!(sim_state, replica_index, tail_idx, n_wi, prev_i, kink_idx_i, i) 
        # 2: delete the two kinks at tau_kink  
        delete_kink_pair!(sim_state, replica_index, kink_idx_i, kink_idx_j) 
        return  
    else # Reject
        return 
    end
end
  
