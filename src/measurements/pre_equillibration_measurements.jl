"""
    Z_frac_measurement!(N_data, head_idx, tail_idx, N_beta) → Z_frac::Float64

Measure whether the current MC sweep is in the Z-sector (closed world-line configuration).

The PIMC worm algorithm has two sectors:
  • Z-sector: head_idx == tail_idx == -1 (no worm endpoints, closed worldlines)
  • Worm sector: head_idx ≠ -1 or tail_idx ≠ -1 (open worldlines with worm)

When the system is in the Z-sector, we record the particle number N_beta for
later density histogram analysis, and return Z_frac = 1.0. Otherwise, no
measurement is recorded and Z_frac = 0.0.

This per-sweep signal is summed over a block to obtain the fraction of MC sweeps
spent in the Z-sector: Z_frac_block = Σ(Z_frac_i) / n_sweeps.

Returns: 1.0 if in Z-sector (closed worldline), 0.0 otherwise.
"""
@inline function Z_frac_measurement!(N_data::Vector{Int64}, head_idx::Int64,
                                    tail_idx::Int64, N_beta::Int64)::Float64
    if (head_idx == -1 && tail_idx == -1)
        push!(N_data, N_beta)
        return 1.0
    end
    return 0.0
end

"""
    Z_frac_measurement!(sim_tracker, sim_state) → Z_frac::Float64

Wrapper that extracts worm indices from sim_state and delegates to the core measurement.
"""
@inline function Z_frac_measurement!(sim_tracker::SimTracker, sim_state::SimState)::Float64
    return Z_frac_measurement!(sim_tracker.N_data,
                              sim_state.head_idx[0 + begin],
                              sim_state.tail_idx[0 + begin],
                              sim_tracker.N_beta[0 + begin])
end

"""
    histogram_measurement(N_data, N_target) → (success, N_bins, P_N, N_mean, peak_idx, N_target_idx, N_min)

Construct a histogram of particle numbers measured in Z-sector sweeps during pre-equilibration.

During pre-equilibration, every sweep that visits the Z-sector records the instantaneous
particle number N. This function builds a normalized probability distribution P(N) from
the accumulated measurements. The histogram is only meaningful if sufficient Z-sector
measurements have been collected (if N_data is empty, returns a failure flag).

Arguments:
  N_data::Vector{Int64}       particle numbers recorded in Z-sector
  N_target::Int64             target particle number for density control

Returns: (success, N_bins, P_N, N_mean, peak_idx, N_target_idx, N_min)
  success::Bool               true if N_target appears in the histogram support
  N_bins::Vector{Int64}       particle numbers on histogram support [N_min, ..., N_max]
  P_N::Vector{Float64}        normalized probability P(N) for each bin
  N_mean::Float64             mean particle number ⟨N⟩ = Σ N·P(N)
  peak_idx::Int64             index (0-based) of the mode (peak) of P(N)
  N_target_idx::Int64         index of N_target in P(N) (undefined if success==false)
  N_min::Int64                lower edge of histogram support

Physics context: In the grand-canonical ensemble at fixed μ, the density ⟨N⟩
fluctuates around a mean value determined by μ. Pre-equilibration tunes μ to
bring ⟨N⟩ → N_target. The histogram is used only for monitoring convergence
and diagnostic output; the μ update uses the running estimate ⟨N⟩_ema directly.
"""
function histogram_measurement(N_data::Vector{Int64}, N_target::Int64)::Tuple{Bool, Vector{Int64},
                                                                              Vector{Float64}, Float64,
                                                                              Int64, Int64, Int64}
    # ── Guard against empty data (insufficient Z-sector measurements) ────────────
    if isempty(N_data)
        return false, Int64[], Float64[], 0.0, 0, 0, 0
    end

    # ── Build histogram support [N_min, ..., N_max] ─────────────────────────────
    N_min = minimum(N_data)
    N_max = maximum(N_data)

    N_bins  = collect(N_min:N_max)
    N_hist  = zeros(Int64, length(N_bins))
    P_N     = zeros(Float64, length(N_bins))

    # ── Check if target N falls within the measured range ──────────────────────
    N_target_in_support = N_min ≤ N_target ≤ N_max

    # ── Accumulate histogram: count samples in each bin ───────────────────────
    for N in N_data
        bin_idx = N - N_min + 1  # 1-based indexing for arrays
        N_hist[bin_idx] += 1
    end

    # ── Normalize to probability distribution and find peak ────────────────────
    N_hist_sum = Float64(sum(N_hist))
    peak_idx = 1  # 1-based for array
    P_N_peak = 0.0

    for i in eachindex(P_N)
        P_N[i] = N_hist[i] / N_hist_sum
        if P_N[i] > P_N_peak
            peak_idx = i
            P_N_peak = P_N[i]
        end
    end

    # ── Compute mean and prepare return values ────────────────────────────────
    N_mean = sum(N_bins .* P_N)
    N_target_idx = N_target_in_support ? N_target - N_min + 1 : 0

    # Convert to 0-based indexing for consistency with return semantics
    return N_target_in_support, N_bins, P_N, N_mean, peak_idx - 1, N_target_idx - 1, N_min
end

"""
    histogram_measurement(sim_tracker, sim_state) → (success, N_bins, P_N, N_mean, peak_idx, N_target_idx, N_min)

Wrapper that delegates to the core histogram measurement function.
"""
function histogram_measurement(sim_tracker::SimTracker, sim_state::SimState)::Tuple{Bool, Vector{Int64},
                                                                                    Vector{Float64}, Float64,
                                                                                    Int64, Int64, Int64}
    return histogram_measurement(sim_tracker.N_data, sim_state.N)
end