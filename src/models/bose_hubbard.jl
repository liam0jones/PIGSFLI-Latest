abstract type BoseHubbard <: ModelSystem end

"""Energy difference for inserting a worm (is_worm=true) or a anti_worm (is_worm=false). 

This function is expected to dispatch on the ModelSystem (first argument). Here, we provide the implementation for the Bose-Hubbard model."""
@inline function worm_energy_difference(::Type{BoseHubbard}, sim_state::SimState, n_tail::Int64, n_head::Int64, is_worm::Bool)  
    dV = (sim_state.U/2.0) * (n_tail*(n_tail-1)-n_head*(n_head-1)) - sim_state.mu*(n_tail-n_head)
    if !is_worm
        dV *= -1.0
    end
    return dV
end

"""Energy difference for timeshift."""
@inline function timeshift_energy_difference(::Type{BoseHubbard}, sim_state::SimState, n::Int64, shift_head::Bool)  
    dV = sim_state.U * (n - !shift_head) - sim_state.mu
    if (!shift_head)
        dV *= -1.0  
    end 
    return dV
end


"""Energy difference for adding kink."""
@inline function add_kink_energy_difference(::Type{BoseHubbard}, sim_state::SimState, n_wi::Int64, n_i::Int64, n_wj::Int64, n_j::Int64)  
    dV_i = worm_energy_difference(BoseHubbard, sim_state, n_wi, n_i, true) 
    dV_j = worm_energy_difference(BoseHubbard, sim_state, n_wj, n_j, true) 
    return dV_j - dV_i
end