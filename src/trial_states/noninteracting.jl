struct NonInteractingTrialState <: TrialState 
end

function get_trial_state_factor(
    n_tail::Int64, 
    is_worm::Bool,
    ::NonInteractingTrialState, 
    ::SimState,
    sim_tracker::SimTracker, 
    replica_index::Int64,
    update_name::String 
    )::Float64

    if update_name == "delete_beta"
        if is_worm 
            return sqrt((sim_tracker.N_beta[replica_index] - 1) / n_tail)
        end 
        return sqrt(n_tail / (sim_tracker.N_beta[replica_index] + 1))
    elseif update_name == "insert_zero"
        if is_worm 
            return sqrt((sim_tracker.N_zero[replica_index] + 1) / n_tail)
        end 
        return sqrt(n_tail / (sim_tracker.N_zero[replica_index]))
    elseif update_name == "insert_beta"
        if is_worm 
            return sqrt((sim_tracker.N_beta[replica_index] + 1) / n_tail)
        end 
        return sqrt(n_tail / (sim_tracker.N_beta[replica_index]))
    elseif update_name == "delete_zero"
        if is_worm 
            return sqrt((sim_tracker.N_zero[replica_index] - 1) / n_tail)
        end 
        return sqrt(n_tail / (sim_tracker.N_zero[replica_index] + 1))
    end
    error("update_name not recognized")
end
 