function update_fock_state!(
    fock_state::FockState, 
    M::Int64, 
    paths::Path, 
    measurement_center::Float64)::Nothing 
    
    for i in 0:M-1
        current = i
        tau = paths[current+begin].tau
        while tau < measurement_center + 1.0E-12 && current != -1
            n_i = paths[current+begin].n
            fock_state[i+begin] = n_i
            
            current = paths[current+begin].next
            if current != -1
                tau = paths[current+begin].tau
            end
        end
    end
    return nothing
end 

function update_fock_state!(sim_state::SimState, r::Int64, sim_tracker::SimTracker)
    update_fock_state!(
                sim_state.fock_state_at_slice,
                sim_state.M,
                sim_state.paths[r], 
                sim_tracker.measurement_center)
end