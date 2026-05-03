struct GutzwillerTrialState <: TrialState
    kappa::Float64
end

function get_trial_state_factor(
    n_tail::Int64, 
    is_worm::Bool,
    trial_state::GutzwillerTrialState, 
    ::SimState,
    ::SimTracker, 
    ::Int64,
    ::String
    )::Float64

    if is_worm 
        return exp((-trial_state.kappa/2.0)*(1+2*(n_tail-1)))/sqrt(n_tail)
    end

    return sqrt(n_tail) * exp((-trial_state.kappa/2.0)*(1-2*n_tail)) 
end
 