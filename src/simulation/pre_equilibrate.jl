"""
Pre-equilibration via projected stochastic approximation (SA).

μ is tuned so that ⟨N⟩ → N_target  (fast timescale, updated every block).
η is tuned so that the fraction of MC sweeps spent in the Z-sector (closed
world-line configurations) converges to the user-specified target Z_target
(slow timescale, after a μ warm-up period).

Step-size schedule (Robbins-Monro):
  aₖ = γ_μ / (k+1)^α  for μ,   b_j = γ_η / (j+1)^α  for η,   α = SA_EXPONENT
  ⟹  Σ aₖ = ∞  and  Σ aₖ² < ∞  (α ∈ (0.5, 1)).

SA fixed points:
  μ*  satisfying  ⟨N⟩(μ*) = N_target          (∂⟨N⟩/∂μ > 0  ⟹  unique)
  η*  satisfying  Z_frac(η*) = Z_target       (∂Z_frac/∂η < 0  ⟹  unique)

where Z_frac = (sweeps in Z-sector) / (total sweeps per block).  At fixed μ*,
Z_frac is a strictly decreasing function of η (higher η → more worm insertions
→ less time in the closed Z-sector), so η* is unique within the projection set.

Z_frac directly controls worm-algorithm efficiency for both canonical and
grand-canonical PIMC:
  • Z_frac too small → worm sector dominates → closed-worldline measurements
    are rare, making the canonical main simulation effectively stalled.
  • Z_frac too large → system rarely enters worm sector → poor topological
    ergodicity, insufficient connectivity between configurations.
Targeting Z_frac = Z_target (user-specified via -Z, typically ~0.5) selects
the η that balances measurement rate against topological ergodicity for both
canonical and grand-canonical main simulation stages.

The EMA of Z_frac — not the raw noisy block estimate — is fed into the SA
update.  This suppresses per-block variance without shifting the fixed point.

Convergence criterion: both μ and η must accumulate a net stability credit of
n_stable_required.  Each block that satisfies its tolerance adds +1; each block
that does not subtracts 1 (floored at 0).  This majority-rule is robust to
occasional noisy blocks near the fixed point: a single bad block costs one
credit rather than resetting all accumulated evidence (the failure mode that
causes non-convergence after 1 M sweeps when the parameters are already frozen
by RM decay and only the block estimators are fluctuating).

Physics-informed η projection bounds (global constants):
  LOGETA_LO = −6  (η_min ≈ 2.5×10⁻³: worm insertions essentially impossible)
  LOGETA_HI =  2  (η_max ≈ 7.4: covers all practical balance points, including weakly interacting / high-T / large-lattice regimes)
"""

# ── Robbins-Monro step schedule ────────────────────────────────────────────────
const SA_EXPONENT = 0.6        # α ∈ (0.5, 1): Σaₖ=∞, Σaₖ²<∞

# ── Global projection bounds ───────────────────────────────────────────────────
const MU_BOUND  = 50.0         # fallback half-range for μ when no tighter bound applies
const LOGETA_LO = -5.0         # log(η) lower bound → η_min ≈ 2.5×10⁻³
const LOGETA_HI =  2.0         # log(η) upper bound → η_max ≈ 7.4
const ETA_MIN   = 1e-10        # hard numerical floor (well below any physical η*)

# ── SA state ───────────────────────────────────────────────────────────────────
mutable struct PreEqSAState
    block::Int64           # μ SA iteration index k  (increments every block)
    eta_updates::Int64     # η SA iteration index j  (increments every η update)
    log_eta::Float64       # log(η) maintained in log-space for numerical safety
    N_ema::Float64         # EMA of per-block ⟨N⟩ estimates
    z_frac_ema::Float64    # EMA of per-block Z-sector fraction
    n_mu_stable::Int64     # net stability credits for μ convergence
    n_eta_stable::Int64    # net stability credits for η convergence
    z_stuck_blocks::Int64  # count of consecutive blocks stuck in Z_frac = 0 or 1
end

function PreEqSAState(sim_state::SimState)
    PreEqSAState(0, 0,
                 clamp(log(sim_state.eta), LOGETA_LO, LOGETA_HI),
                 Float64(sim_state.N),
                 0.5,       # neutral prior; overwritten with measured value on first block
                 0, 0, 0)
end

# ── Robbins-Monro step size ────────────────────────────────────────────────────
@inline sa_step(γ::Float64, k::Int64) = γ / (k + 1)^SA_EXPONENT

# ── Runtime μ bounds ───────────────────────────────────────
"""
    mu_bounds(sim_state) → (mu_lo, mu_hi)

Returns the SA projection interval for μ.

Lower bound: −MU_BOUND for all fillings. In the superfluid phase of the
Bose-Hubbard model the fixed point satisfies μ* ≈ −z·t (band bottom), which
is negative regardless of filling. Restricting μ ≥ 0 based on filling alone
would exclude the true fixed point from the projection set whenever the system
is in the superfluid phase, causing the SA to stall at the boundary.

Upper bound: max(MU_BOUND, ⌈N/M⌉·U). The nth Mott lobe has its upper
phase boundary at μ ≈ n·U, so the ceiling filling times U gives a safe
physical ceiling that grows with filling and interaction strength.
MU_BOUND = 50 provides a floor for non-interacting or sub-unit-filling cases.
"""
function mu_bounds(sim_state::SimState)
    n_ceil = cld(sim_state.N, sim_state.M)
    mu_lo  = -MU_BOUND
    mu_hi  = max(MU_BOUND, Float64(n_ceil) * sim_state.U)
    return mu_lo, mu_hi
end

# ── Projected SA updates ───────────────────────────────────────────────────────

"""
    sa_update_mu!(sim_state, N_block, a_k, mu_lo, mu_hi, max_step)

μₖ₊₁ = Proj_{[mu_lo, mu_hi]}( μₖ − clamp(aₖ(N̄ₖ−N_target), −max_step, max_step) )

The per-step cap prevents a single noisy block from catapulting μ across the
projection set. The RM convergence guarantee is preserved: steps shrink to
zero via the schedule, and the cap only activates in the large-signal regime.
"""
@inline function sa_update_mu!(sim_state::SimState, N_block::Float64, a_k::Float64,
                               mu_lo::Float64, mu_hi::Float64, max_step::Float64)
    delta = clamp(-a_k * (N_block - sim_state.N), -max_step, max_step)
    sim_state.mu = clamp(sim_state.mu + delta, mu_lo, mu_hi)
end

"""
    sa_update_logeta!(sim_state, state, b_j, max_step, Z_target, Z_frac, sweeps_per_block)

log(η)_{j+1} = Proj_{[LOGETA_LO, LOGETA_HI]}( log(η)_j + clamp(b_j · s, −max_step, max_step) )

where  s = logit(Z_frac*) − logit(Z_target),  logit(p) = log(p/(1−p)).

The logit transform maps Z_frac ∈ (0,1) onto (−∞,+∞) and amplifies the
correction signal when Z_frac approaches 0% or 100%:

  Z_frac ≈ 0  →  logit(Z_frac*) → −∞  →  large negative signal  →  decrease η
  Z_frac ≈ 1  →  logit(Z_frac*) → +∞  →  large positive signal  →  increase η

Near the fixed point Z_frac* ≈ Z_target the logit is nearly linear and the
signal stays small, preserving stability. The linear signal Z_frac − Z_target
only provides |signal| ≤ 1 regardless of how extreme the imbalance is, so the
SA takes many small steps to escape from Z_frac = 0 or Z_frac = 1.  The logit
increases those corrections by a factor of 5–10× when the system is stuck in
one sector, while leaving near-equilibrium updates unchanged.

A half-sample continuity correction clips Z_frac to [0.5/n, 1−0.5/n] before
the logit to avoid log(0) when the system spends an entire block in one sector.

Uses the raw per-block Z_frac (not the EMA) as the SA signal, consistent with
how the μ update uses the raw N_block.  The EMA is reserved for convergence
checking only; feeding the EMA into the SA update introduces a lag bias that
can drive η past the fixed point when Z_frac changes rapidly.
"""
@inline function sa_update_logeta!(sim_state::SimState, state::PreEqSAState,
                                   b_j::Float64, max_step::Float64,
                                   Z_target::Float64, Z_frac::Float64,
                                   sweeps_per_block::Int64)
    state.z_stuck_blocks = (Z_frac == 0.0 || Z_frac == 1.0) ? state.z_stuck_blocks + 1 : 0
    clip  = 0.5 / sweeps_per_block
    Z_safe = clamp(Z_frac, clip, 1.0 - clip)
    signal = log(Z_safe / (1.0 - Z_safe)) - log(Z_target / (1.0 - Z_target))
    delta  = clamp(b_j * signal, -max_step, max_step)
    state.log_eta = clamp(state.log_eta + delta, LOGETA_LO, LOGETA_HI)
    sim_state.eta = max(exp(state.log_eta), ETA_MIN)
end

# ── Block measurement bookkeeping ─────────────────────────────────────────────
function clear_block!(sim_tracker::SimTracker)
    empty!(sim_tracker.N_data)
    sim_tracker.Z_ctr[1] = 0
    reset!(sim_tracker.step_counters)
end

# ── Debugging output ───────────────────────────────────────────────────────────
"""
    print_preeq_debug(k, Z_frac, N_block, state, sim_state, Z_target, 
                      z_frac_tol, mu_stable, eta_stable, sweeps_per_block)

Print per-block diagnostics during pre-equilibration. Called at strategic
points to track Z_frac evolution, μ and η trajectories, and convergence progress.
"""
function print_preeq_debug(k::Int64, Z_frac::Float64, N_block::Float64, 
                          state::PreEqSAState, sim_state::SimState,
                          mu_stable::Bool, eta_stable::Bool)
    # Format values for display
    z_frac_pct = round(Z_frac * 100.0; digits=1)
    z_frac_ema_pct = round(state.z_frac_ema * 100.0; digits=1)
    mu_fmt = round(sim_state.mu; digits=4)
    eta_fmt = round(sim_state.eta; digits=4)
    mu_status = mu_stable ? "✓" : "✗"
    eta_status = eta_stable ? "✓" : "✗"
    
    # Print diagnostic line
    println("  Block $(lpad(k,3)) | Z_frac=$(lpad(z_frac_pct,5))% (EMA=$(lpad(z_frac_ema_pct,4))%) | " *
            "N=$(lpad(round(N_block; digits=2),5)) (EMA=$(lpad(round(state.N_ema; digits=2),4))) | μ=$(lpad(mu_fmt,8)) $(mu_status) | η=$(lpad(eta_fmt,7)) $(eta_status)")
end

# ── Main pre-equilibration ─────────────────────────────────────────────────────
function pre_equilibrate!(
    rng::AbstractRNG,
    sim_state::SimState,
    sim_options::SimOptions,
    sim_tracker::SimTracker;
    silent::Bool = false,
)
    sim_tracker.sim_phase = pre_equilibrate_phase
    # Set to grand-canonical for pre-equilibration tuning, then restore at end
    sim_state.canonical = false
    # Explicitly set target particle number N for grand-canonical tuning
    target_N = sim_state.N
    norelax_mu  = sim_options.norelax_mu_preeq
    norelax_eta = sim_options.norelax_eta_preeq

    if norelax_mu && norelax_eta
        !silent && println("Pre-equilibration skipped (--no-relax-mu and --no-relax-eta both set). " *
                           "Using μ = $(sim_state.mu), η = $(sim_state.eta).")
        return
    end

    # ── Physics-informed μ bounds ──────────────────────────────────────────────
    mu_lo, mu_hi = mu_bounds(sim_state)

    # Project the starting μ onto the physical interval immediately.
    if !norelax_mu
        sim_state.mu = clamp(sim_state.mu, mu_lo, mu_hi)
    end

    # ── η target from user options ─────────────────────────────────────────────
    Z_target   = sim_options.Z      # Already a fraction (e.g., 0.48 for 48%)
    z_frac_tol = sim_options.dZ     # Already a fraction (e.g., 0.03 for 3%)

    # ── Hyper-parameters ──────────────────────────────────────────────────────
    sweeps_per_block  = sim_options.sweeps_pre  # Default: 1000000*M*β
    max_blocks        = sim_options.max_blocks
    # μ gain: normalised by N
    γ_mu              = 0.5 / max(1.0, Float64(sim_state.N))
    # η gain.  The logit-linear approximation gives logit(Z_frac) ≈ −2 log(η) + C
    # near the fixed point, so the SA Jacobian per update is b_j × 2.
    γ_eta             = 0.5
    # Per-block step caps.
    # μ: prevents a single noisy block from moving μ by an unreasonable amount.
    max_mu_step       = 2.0
    # η: reduced to 0.35 log-units to prevent overshoot and oscillation.
    max_eta_step      = 0.3
    # Warmup period: allow μ to converge first, THEN start counting η convergence.
    mu_warmup_blocks  = 5
    # EMA smoothing for convergence tracking
    ema_alpha         = 0.35
    # |N_ema − N_target| / N_target < density_tol → μ declared converged.
    density_tol       = 0.02
    # Convergence stability requirements (separate thresholds reflect physical importance):
    n_eta_stable_required = 6
    n_mu_stable_required = 3

    state = PreEqSAState(sim_state)

    !silent && println("\nStage (1/3): Pre-equilibrating...  " *
                       "(μ ∈ [$(round(mu_lo, digits=2)), $(round(mu_hi, digits=2))], " *
                       "η ∈ [$(round(exp(LOGETA_LO), digits=4)), $(round(exp(LOGETA_HI), digits=2))], " *
                       "Z_target = $(Z_target) ± $(z_frac_tol))\n")

    while state.block < max_blocks
        state.block += 1
        k = state.block

        # ── Run one block of MC sweeps ─────────────────────────────────────────
        clear_block!(sim_tracker)
        for _ = 1:sweeps_per_block
            random_mc_update!(rng, sim_state, 1, sim_tracker)
            sim_tracker.Z_ctr[1] += round(Int64, Z_frac_measurement!(sim_tracker, sim_state))
        end

        # ── Block estimators ─────────────────────────────────────────────────
        Z_frac = sim_tracker.Z_ctr[1] / sweeps_per_block
        has_z_data = sim_tracker.Z_ctr[1] > 0
        N_block = has_z_data ? sum(sim_tracker.N_data) / length(sim_tracker.N_data) : NaN

        # ── EMA update ────────────────────────────────────────────────────────
        if k == 1
            state.N_ema      = has_z_data ? N_block : Float64(sim_state.N)
            state.z_frac_ema = Z_frac
        else
            has_z_data && (state.N_ema = ema_alpha * N_block + (1 - ema_alpha) * state.N_ema)
            state.z_frac_ema = ema_alpha * Z_frac + (1 - ema_alpha) * state.z_frac_ema
        end

        # ── μ SA update (fast timescale, every block with Z-sector data) ────────
        mu_stable = norelax_mu
        if !norelax_mu
            if has_z_data
                if k ≤ mu_warmup_blocks
                    γ_mu_eff = 2.0 / max(1.0, sim_state.N)
                else
                    γ_mu_eff = γ_mu
                end
                sa_update_mu!(sim_state, N_block, sa_step(γ_mu_eff, k), mu_lo, mu_hi, max_mu_step)
                mu_stable = abs(state.N_ema - sim_state.N) / sim_state.N < density_tol
            else
                sim_state.mu = clamp(sim_state.mu + 0.01 * sign(sim_state.N - state.N_ema), mu_lo, mu_hi)
            end
        end

        # ── η SA update (slow timescale, Z-fraction targeting) ────────────────
        eta_stable = norelax_eta
        if !norelax_eta
            state.eta_updates += 1
            sa_update_logeta!(sim_state, state, sa_step(γ_eta, state.eta_updates),
                              max_eta_step, Z_target, Z_frac, sweeps_per_block)
            # Only COUNT toward convergence after warmup period
            eta_stable = (k > mu_warmup_blocks) && (abs(state.z_frac_ema - Z_target) ≤ z_frac_tol)
        end

        # ── Stability credit counters ──────────────────────────────────────────
        state.n_mu_stable  = mu_stable  ? state.n_mu_stable  + 1 : max(0, state.n_mu_stable  - 1)
        state.n_eta_stable = eta_stable ? state.n_eta_stable + 1 : max(0, state.n_eta_stable - 1)

        # ── Diagnostic output ──────────────────────────────────────────────────
        if !silent
            print_preeq_debug(k, Z_frac, N_block, state, sim_state, mu_stable, eta_stable)
        end

        # ── Convergence check ─────────────────────────────────────────────────
        if state.n_mu_stable ≥ n_mu_stable_required && state.n_eta_stable ≥ n_eta_stable_required
            if !silent
                println("\n  ✓ Pre-equilibration converged at block $(k):")
                println("    μ = $(round(sim_state.mu, digits=6))")
                println("    η = $(round(sim_state.eta, digits=6))")
                println("    Z_frac = $(round(state.z_frac_ema, digits=3)) (target $(Z_target) ± $(z_frac_tol))")
                println("    N = $(round(state.N_ema, digits=2)) (target $(sim_state.N) ± $(round(density_tol * sim_state.N, digits=2)))\n")
            end
            break
        end
    end

    if state.block > max_blocks
        if !silent
            println("\n  ✗ Pre-equilibration FAILED to converge after $(state.block) blocks.")
            println("    Final state:")
            println("      μ = $(round(sim_state.mu, digits=6))")
            println("      η = $(round(sim_state.eta, digits=6))")
            println("      Z_frac = $(round(state.z_frac_ema, digits=3)) (target $(Z_target) ± $(z_frac_tol))")
            println("      Stability credits: μ=$(state.n_mu_stable)/$n_mu_stable_required, η=$(state.n_eta_stable)/$n_eta_stable_required")
            println("    Stuck counter: Z_frac consecutive blocks at 0/1 = $(state.z_stuck_blocks)")
            println("    Diagnostics:")
            println("      - If Z_frac stays at 0%, the worm sector is too favorable (η too high)")
            println("      - If Z_frac stays at 100%, the worm sector is too unfavorable (η too low)")
            println("      - If μ is at bound, particle density control is not working")
        end
        error("Pre-equilibration failed to converge. " *
              "Try: increase --sweeps-pre, adjust --eta / --Z, or check initial μ value.")
    end
    sim_state.canonical = true
end
