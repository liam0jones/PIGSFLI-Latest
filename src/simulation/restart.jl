function restart(  
    state_fh::SnapshotHandler)::Tuple{AbstractRNG, SimState, SimTracker}

    # load from file 
    loaded_rng::AbstractRNG = read(state_fh, "rng")
    loaded_sim_state::SimState = read(state_fh, "state")
    loaded_sim_tracker::SimTracker = read(state_fh, "tracker") 

    return loaded_rng, loaded_sim_state, loaded_sim_tracker 
end