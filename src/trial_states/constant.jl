struct ConstantTrialState <: TrialState
end

function get_trial_state_factor(
    ::Int64, 
    ::Bool,
    ::ConstantTrialState, 
    ::SimState,
    ::SimTracker,
    ::Int64,
    ::String
    )::Float64

    return 1.0
end