module Pigsfli
 
pigsfli_str = """        
       _            __ _ _     _  _
      (_)          / _| (_)   (_)| |
 _ __  _  __ _ ___| |_| |_     _ | |
| '_ \\| |/ _` / __|  _| | |   | || |
| |_) | | (_| \\__ \\ | | | | _ | || |
| .__/|_|\\__, |___/_| |_|_|(_)| ||_|
| |       __/ |             __/ | 
|_|      |___/             |___/ 

Path-Integral Ground State (Monte Carlo) For Lattice Implementations
""" 
     # External Modules 
     using ArgParse
     using Printf
     using JLD2 
     # ReLambertW 
     include("ReLambertW/ReLambertW.jl")
     # Exports 
     include("exports.jl") 
     # Imports 
     include("imports.jl") 
     # Global Variables 
     include("global_variables.jl")

     # utils/*
     include("utils/abstract_types.jl")  
     # file_handler/*
     include("file_handler/write_output_handler.jl")
     include("file_handler/snapshot_handler.jl") 
     # random/*
     include("random/rng.jl")
     include("random/xoshiro256pp.jl")
     # path/* 
     include("path/fockstate.jl") 
     include("path/kink.jl")
     include("path/path.jl")
     # geometry/*
     include("geometry/adjacency_matrix.jl")
     include("geometry/sparse_adjacency_helpers.jl")
     include("geometry/sites.jl")      
     # simulation/*
     include("simulation/sim_options.jl")
     include("simulation/sim_state.jl")
     include("simulation/sim_trackers.jl") 
     # validation 
     include("utils/validation.jl")
     # debug plots
     include("utils/plot.jl")  
     # others from simulation/*     
     include("simulation/monte_carlo.jl")
     include("simulation/pre_equilibrate.jl")
     include("simulation/restart.jl")
     # measurements/*
     include("measurements/measurement_centers.jl")
     include("measurements/monte_carlo_measurements.jl")
     include("measurements/pre_equillibration_measurements.jl") 
     # trial_states/*
     include("trial_states/constant.jl")
     include("trial_states/gutzwiller.jl")
     include("trial_states/noninteracting.jl")
     # models/*
     include("models/bose_hubbard.jl")
     # updates/* 
     include("updates/truncated_exponential_sampling.jl")
     include("updates/kink_operations.jl")
     include("updates/update_fockstate.jl") 
     include("updates/updates.jl") 
     include("updates/swap_updates.jl")
     
end