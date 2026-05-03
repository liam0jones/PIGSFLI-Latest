struct SimOptions  
    restart::Bool 
    num_replicas::Int64
    sweeps_pre::Int64
    sweeps::Int64
    sweep::Int64
    max_blocks::Int64
    measurement_frequency::Int64
    measure_tau_resolved_estimators::Bool 
    measure_n::Bool
    measure_density::Bool
    measure_corr_mat::Bool
    measure_corr::Bool
    no_accessible::Bool
    bins_wanted::Int64
    bin_size::Int64
    Z::Float64
    dZ::Float64
    norelax_mu_preeq::Bool
    norelax_eta_preeq::Bool
    save_state::Bool
    out_folder::String
end

function SimOptions(
    sweeps_pre::Int64,
    sweeps::Int64,
    sweep::Int64,
    max_blocks::Int64,
    measurement_frequency::Int64, 
    measure_n::Bool,
    measure_density::Bool,
    measure_corr_mat::Bool,
    measure_corr::Bool,
    no_accessible::Bool,
    bins_wanted::Int64,
    bin_size::Int64,
    Z::Float64,
    dZ::Float64; 
    restart::Bool=false,
    num_replicas::Int64=1,
    measure_tau_resolved_estimators::Bool=false,
    norelax_mu_preeq::Bool=false,
    norelax_eta_preeq::Bool=false,
    save_state::Bool,
    out_folder::String="./out")

    SimOptions(restart, num_replicas, sweeps_pre, sweeps, sweep, max_blocks, measurement_frequency, measure_tau_resolved_estimators, measure_n, measure_density, measure_corr_mat, measure_corr, no_accessible, bins_wanted, bin_size, Z, dZ, norelax_mu_preeq, norelax_eta_preeq, save_state, out_folder)
end