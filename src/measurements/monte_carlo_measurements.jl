function conventional_measurement!(
    sim_tracker::SimTracker,
    sim_state::SimState,
    sim_options::SimOptions,
    writing_ctr::Vector{Int64},
    collect_corr_mat::Bool=true)

    r::Int64 = 1 # TEMPORARY (eventually might loop over 1,2)  

    if sim_state.head_idx[r]==-1 && sim_state.tail_idx[r]==-1 
        sim_tracker.N_sum[r] += sim_tracker.Ns[r] 
        sim_tracker.Z_ctr[r] += 1 

        # canonical measurement 
        if sim_tracker.N_zero[r] == sim_state.N && sim_tracker.N_beta[r] == sim_state.N && writing_ctr[1] != sim_options.bin_size
            # Get fock state at desired measurement center
            update_fock_state!(sim_state, r, sim_tracker) 

            # Measure and accumulate <K>
            sim_tracker.kinetic_energy += pimc_kinetic_energy(sim_state,sim_tracker,r)
                
            # Measure and accumulate <V>
            sim_tracker.diagonal_energy += pimc_diagonal_energy(sim_state,sim_tracker)
                
            if sim_options.measure_tau_resolved_estimators 
                tau_resolved_kinetic_energy!(sim_tracker, sim_state, r) 
                tau_resolved_diagonal_energy!(sim_tracker, sim_state, r)  
            end

            # Measure <n> and <n^2>
            if sim_options.measure_n
                n_measurement!(sim_tracker, sim_state, sim_options)
            end

            # Record <n_i> and <n^2_i>
            if sim_options.measure_density
                sim_tracker.density += sim_state.fock_state_at_slice.data
                sim_tracker.density_squared += sim_state.fock_state_at_slice.data .^2
            end

            if sim_options.measure_corr
                corr_measurement!(sim_tracker, sim_state, sim_options)
            end

            writing_ctr[1] += 1
        end

    elseif sim_options.measure_corr_mat && collect_corr_mat
        if sim_state.head_idx[r] != -1 && sim_state.tail_idx[r] != -1
            # If both worm ends are present, attempt to measure one-body correlation
            writing_ctr[2] = measure_corr_mat!(sim_state, sim_tracker, r, writing_ctr[2])
        end
    end
    return writing_ctr
end


function reset_conventional_measurement!(sim_tracker::SimTracker,sim_options::SimOptions)
    sim_tracker.kinetic_energy = 0.0
    sim_tracker.diagonal_energy = 0.0
    if sim_options.measure_tau_resolved_estimators
        fill!(sim_tracker.tr_kinetic_energy, 0.0)
        fill!(sim_tracker.tr_diagonal_energy, 0.0)
    end  
    if sim_options.measure_n
        fill!(sim_tracker.n_A_accum, 0.0)
        fill!(sim_tracker.n_A_squared_accum, 0.0)
    end

    if sim_options.measure_density
	fill!(sim_tracker.density, 0)
	fill!(sim_tracker.density_squared, 0)
    end

    if sim_options.measure_corr
        fill!(sim_tracker.corr_accum, 0.0)
        fill!(sim_tracker.sigma2_accum, 0.0)
    end

    # Reset correlation matrix accumulators for next bin
    if sim_options.measure_corr_mat
        fill!(sim_tracker.corr_mat, 0.0)
        fill!(sim_tracker.Z_ctr, 0)
    end
end

function measure_corr_mat!(sim_state::SimState, sim_tracker::SimTracker, r::Int64, corr_ctr::Int64)
    tail_idx = sim_state.tail_idx[r]
    head_idx = sim_state.head_idx[r]

    if tail_idx == -1 || head_idx == -1
        return corr_ctr
    end

    paths = sim_state.paths[r]
    M = sim_state.M
    measurement_center = sim_tracker.measurement_center

    tau_head = paths[head_idx+begin].tau
    tau_tail = paths[tail_idx+begin].tau

    # Only antiworms (head precedes tail in imaginary time)
    tau_head > tau_tail && return corr_ctr

    # Antiworm must span the measurement center for PIGS ground-state estimator
    !(tau_head <= measurement_center <= tau_tail) && return corr_ctr

    if sim_state.canonical
        # Exclude antiworms whose head originates from tau=0 (insert_zero) or whose tail originates from tau=beta (insert_beta)
        # These carry trial-state weight factors and must not enter the bulk estimator
        paths[paths[head_idx+begin].prev+begin].tau < DELTA_TAU/2 && return corr_ctr
        paths[tail_idx+begin].next == -1 && return corr_ctr
    end

    # C_ij = ⟨b^†_i b_j⟩: tail carries b^†, head carries b
    i = paths[tail_idx+begin].src
    j = paths[head_idx+begin].src
    n_i_before = paths[tail_idx+begin].n - 1
    n_j_before = paths[head_idx+begin].n + 1

    if i < 0 || i >= M || j < 0 || j >= M
        @warn "Invalid site indices: i=$i, j=$j (M=$M) in measure_corr_mat!" maxlog=5
        return corr_ctr
    end

    sim_tracker.corr_mat[i+begin, j+begin] += sqrt(n_j_before * (n_i_before + 1)) / sim_state.eta
    corr_ctr += 1

    return corr_ctr
end

function n_measurement!(sim_tracker::SimTracker, sim_state::SimState, sim_options::SimOptions)
    for REP = 1:sim_state.num_replicas 
        fill!(sim_tracker.n_A[REP],0.0) 
        update_fock_state!(sim_state.fock_state_at_half_plus[REP],sim_state.M,sim_state.paths[REP],sim_state.beta/2.0) 
        n_A_last = 0 # tracks subsystem n
        for m_A_primed = 1:sim_state.m_A 
            n_A_last += sim_state.fock_state_at_half_plus[REP][sim_state.sub_sites[m_A_primed-1+begin]+begin] 
            sim_tracker.n_A[REP][m_A_primed-1+begin] = n_A_last # needed to eventually compare if both replicas are on same local particle number sector
        end
    end
    for m_A_primed = 1:sim_state.m_A  
        sim_tracker.n_A_accum[m_A_primed-1+begin] += sim_tracker.n_A[begin][m_A_primed-1+begin]
        sim_tracker.n_A_squared_accum[m_A_primed-1+begin] += (sim_tracker.n_A[begin][m_A_primed-1+begin] * sim_tracker.n_A[begin][m_A_primed-1+begin]) 
    end
    return nothing
end

function corr_measurement!(sim_tracker::SimTracker, sim_state::SimState, sim_options::SimOptions)
    for REP = 1:sim_state.num_replicas   
        fill!(sim_tracker.n_i[REP],0.0) 
        update_fock_state!(sim_state.fock_state_at_half_plus[REP],sim_state.M,sim_state.paths[REP],sim_state.beta/2.0)  
        for i_primed = 0:sim_state.L-1 
            sim_tracker.n_i[REP][i_primed+begin] = sim_state.fock_state_at_half_plus[REP][i_primed+begin] 
        end
    end
    for r = 0:floor(Int64,(sim_state.L+1)/2)  
        for i = 0:sim_state.L-1
            sim_tracker.corr_accum[r+begin] += sim_tracker.n_i[begin][i+begin] * sim_tracker.n_i[begin][((r+i)%sim_state.L) + begin] * 1.0/sim_state.L 
        end
    end

    for li = 1:sim_state.m_A 
        for i = 0:sim_state.L-1
            sim_tracker.sigma2_accum[li-1+begin] += li * sim_tracker.n_i[begin][i+begin] * sim_tracker.n_i[begin][((i)%sim_state.L)+begin] * 1.0/sim_state.L
        end
    end

    for li = 2:sim_state.m_A 
        for r = 1:li-1  
            for i = 0:sim_state.L-1 
                sim_tracker.sigma2_accum[li-1+begin] += 2*(li-r) * sim_tracker.n_i[begin][i+begin] * sim_tracker.n_i[begin][((r+i)%sim_state.L)+begin] * 1.0/sim_state.L
            end
        end
    end 
    return nothing

end

function swap_measurement!(sim_tracker::SimTracker, sim_state::SimState, sim_options::SimOptions, writing_ctr::Int64) 
    N::Int64 = sim_state.N
    num_swaps::Int64 = sim_state.num_swaps 


    if (sim_state.head_idx[0+begin] == -1 && sim_state.head_idx[1+begin] == -1 && sim_state.tail_idx[0+begin] == -1 && sim_state.tail_idx[1+begin] == -1)
        
        if (sim_tracker.N_zero[0+begin] == N && sim_tracker.N_beta[0+begin] == N && sim_tracker.N_zero[1+begin] == N && sim_tracker.N_beta[1+begin] == N)
            # Add count to histogram of number of swapped sites
            sim_tracker.SWAP_histogram[num_swaps+begin] += 1
            
            if (sim_options.no_accessible)
                writing_ctr += 1
            end
            
            if (!sim_options.no_accessible)
                # Build subsystem particle number distribution P(n)
                if num_swaps == 0
                    for REP = 1:sim_state.num_replicas
                        sim_tracker.n_A[REP] = repeat([0], sim_state.m_A)

                        update_fock_state!(
                            sim_state.fock_state_at_half_plus[REP],
                            sim_state.M,
                            sim_state.paths[REP],
                            sim_state.beta/2.0) 

                        n_A_last::Int64 = 0 # tracks subsystem n
                        for m_A_primed = 1:sim_state.m_A 
                            n_A_last += sim_state.fock_state_at_half_plus[REP][sim_state.sub_sites[m_A_primed-1+begin]+begin]
                            sim_tracker.n_A[REP][m_A_primed-1+begin] = n_A_last # needed to eventually compare if both replicas are on same local particle number sector
                            sim_tracker.Pn[m_A_primed-1+begin][n_A_last + begin] += 1
                        end
                        
                        # Energies measurement ?
                        
                        # Get Fock state at measurement center
                        update_fock_state!(
                            sim_state.fock_state_at_slice,
                            sim_state.M,
                            sim_state.paths[REP],
                            sim_tracker.measurement_center)  

                    end
                    
                    # Build P(n)^2
                    # Joint Prob Dist of both replicas having same n
                    # and no SWAP
                    for m_A_primed = 1:sim_state.m_A
                        if sim_tracker.n_A[1][m_A_primed-1+begin] == sim_tracker.n_A[2][m_A_primed-1+begin]  
                            sim_tracker.Pn_squared[m_A_primed-1+begin][sim_tracker.n_A[1][m_A_primed-1+begin]+begin] += 1
                        end
                    end
                     
                
                else  # num_swaps>0
                
                    # Get total local particle number for partitions of
                    # sizes m_A_primed=0 up to m_A_primed=m_A_max
                    
                    # Add count to swapped sites histogram of n-sector
                    for REP = 1:sim_state.num_replicas #/ THIS LOOP IS ACTUALLY NOT NECESSARY. If we made it here, n[0]==n[1].
                        sim_tracker.n_A[REP] = repeat([0], sim_state.m_A)

                        update_fock_state!(
                            sim_state.fock_state_at_half_plus[REP],
                            sim_state.M,
                            sim_state.paths[REP],
                            sim_state.beta/2.0)  

                        n_A_last::Int64 = 0  # tracks subsystem n
                        for i = 0:sim_state.num_swaps-1
                            n_A_last += sim_state.fock_state_at_half_plus[REP][sim_state.sub_sites[i+begin]+begin];
                            sim_tracker.n_A[REP][i+begin] = n_A_last 
                        end
                    end
                    if (sim_tracker.n_A[0+begin][num_swaps-1+begin] == sim_tracker.n_A[1+begin][num_swaps-1+begin]) # Not necessary. When there are SWAPs, n0 and n1 are the same.
                        sim_tracker.SWAPn_histograms[num_swaps-1+begin][sim_tracker.n_A[0+begin][num_swaps-1+begin]+begin] += 1
                        if num_swaps == sim_state.m_A 
                            writing_ctr += 1
                        end
                        # SWAPn_histograms[num_swaps-1][number of particles in the subregion]+=1; 
                    else
                        error("Despite SWAP, sim_tracker.n_A[1][num_swaps] == sim_tracker.n_A[1][num_swaps] is not true. This is not expected.") 
                    end
                end
            end # accessible entanglement if statement
        end
    end 
    return writing_ctr
end

function reset_swap_measurement!(sim_tracker::SimTracker,sim_options::SimOptions) 
    fill!(sim_tracker.SWAP_histogram, 0)
    sim_tracker.Pn = init_Pn(sim_state.m_A, sim_state.N)
    sim_tracker.Pn_squared = init_Pn_squared(sim_state.m_A, sim_state.N)
    sim_tracker.SWAPn_histograms = init_SWAPn_histograms(sim_state.m_A, sim_state.N)
end

function pimc_kinetic_energy(sim_state::SimState,sim_tracker::SimTracker,r::Int64) :: Float64  
    return pimc_kinetic_energy(
        sim_state.paths[r],
        sim_state.num_kinks[r],
        sim_tracker.measurement_center,
        sim_tracker.measurement_plus_minus,
        sim_state.M,
        sim_state.t,
        sim_state.beta) 
end

function pimc_kinetic_energy(
    paths::Path,
    num_kinks::Int64,
    measurement_center::Float64,
    measurement_plus_minus::Float64,
    M::Int64,
    t::Float64,
    beta::Float64) :: Float64 
     
    kinks_in_window::Int64 = 0
    
    for k = 1:num_kinks
        if (paths[k].tau >= measurement_center-measurement_plus_minus && paths[k].tau <= measurement_center+measurement_plus_minus)
            kinks_in_window += 1 
        end
    end
    
    return (-kinks_in_window/2.0)/(2.0*measurement_plus_minus)
end

function pimc_diagonal_energy(sim_state::SimState,sim_tracker::SimTracker) :: Float64
    return pimc_diagonal_energy(
        sim_state.fock_state_at_slice,
        sim_state.M,
        sim_state.canonical,
        sim_state.U,
        sim_state.mu)
end

function pimc_diagonal_energy(
    fock_state_at_slice::FockState,
    M::Int64,
    canonical::Bool,
    U::Float64,
    mu::Float64) :: Float64
    
    diagonal_energy::Float64 = 0.0 
    for i = 1:M
        n_i = fock_state_at_slice[i] 
        if (canonical)
            diagonal_energy += (U/2.0*n_i*(n_i-1)) 
        else
            diagonal_energy += (U/2.0*n_i*(n_i-1)-mu*n_i) 
        end
    end
    return diagonal_energy 
end

function tau_resolved_kinetic_energy!(sim_tracker::SimTracker, sim_state::SimState, r::Int64) 
    tau_resolved_kinetic_energy!( 
        sim_tracker.tr_kinetic_energy[r],
        sim_state.paths[r],
        sim_state.num_kinks[r],
        sim_state.M,
        sim_state.t,
        sim_state.beta,
        sim_tracker.measurement_centers)
end

function tau_resolved_kinetic_energy!(
    tr_kinetic_energy::Vector{Float64},
    paths::Path,
    num_kinks::Int64,
    M::Int64,
    t::Float64,
    beta::Float64,
    measurement_centers::Vector{Float64})  

    window_width::Float64 = measurement_centers[2+begin]-measurement_centers[1+begin];
    
    for i = M:num_kinks-1 # Note: the tau=0 kinks not counted
        tau::Float64 = paths[i+begin].tau 

        for j = 0:length(measurement_centers)-1
            measurement_center = measurement_centers[j+begin] 

            if ( tau >= measurement_center-window_width/2.0 && tau < measurement_center+window_width/2.0) 
                # add kink to bin
                tr_kinetic_energy[j+begin] += (-1.0/(2.0*window_width)) 
                break 
            end
        end
    end 
end

function tau_resolved_diagonal_energy!(sim_tracker::SimTracker, sim_state::SimState, r::Int64)  
    tau_resolved_diagonal_energy!(
                    sim_tracker.tr_diagonal_energy[r],
                    sim_state.paths[r],
                    sim_state.num_kinks[r],
                    sim_state.M,
                    sim_state.canonical,
                    sim_state.U,
                    sim_state.mu,
                    sim_state.beta,
                    sim_tracker.measurement_centers)  
end

function tau_resolved_diagonal_energy!(
    tr_diagonal_energy::Vector{Float64},
    paths::Path,
    num_kinks::Int64,
    M::Int64,
    canonical::Bool,
    U::Float64,
    mu::Float64,
    beta::Float64,
    measurement_centers::Vector{Float64}) 
 
    for i = 0:M-1
        current = i 
        tau = paths[current+begin].tau 
        n_i = paths[current+begin].n 

        for j = 0:length(measurement_centers)-1
            measurement_center = measurement_centers[j+begin] 

            while (tau <= measurement_center && current != -1) 
                n_i = paths[current+begin].n 
                
                current = paths[current+begin].next 
                if current != -1  
                    tau = paths[current+begin].tau 
                end
            end
            if canonical 
                tr_diagonal_energy[j+begin] += (U/2.0*n_i*(n_i-1.0)) 
            else
                tr_diagonal_energy[j+begin] += (U/2.0*n_i*(n_i-1.0)-mu*n_i)  
            end
        end
    end
end
