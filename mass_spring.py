import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from scipy.linalg import eigh

from utils import (
    PALETTE,
    SAVE_DIR,
    calculate_fixed_hook_length_ms,
    draw_spring_with_hook,
    draw_wall,
    estimate_normal_mode_frequencies,
    format_frequency,
    save_animation,
    solve_ode,
    style_dark_subplot,
)


def two_mass_spring_derivatives(t, y, m1, m2, k1, k2, k3):
    """
    Compute the derivatives for the two-mass three-spring system.

    System: Left Wall —[k1]— m1 —[k2]— m2 —[k3]— Right Wall

    Equations of motion (from theory.md):
    m₁ẍ₁ + (k₁ + k₂)x₁ - k₂x₂ = 0
    m₂ẍ₂ + (k₂ + k₃)x₂ - k₂x₁ = 0

    Parameters
    ----------
    t : float
        Time (not used explicitly, required by solve_ivp)
    y : array-like
        State vector [x₁, v₁, x₂, v₂] where v = dx/dt
    m1, m2 : float
        Masses
    k1, k2, k3 : float
        Spring constants

    Returns
    -------
    list
        Derivatives [dx₁/dt, dv₁/dt, dx₂/dt, dv₂/dt]
    """
    x1, v1, x2, v2 = y

    # Accelerations from equations of motion
    a1 = (-(k1 + k2) * x1 + k2 * x2) / m1
    a2 = (k2 * x1 - (k2 + k3) * x2) / m2

    return [v1, a1, v2, a2]


def th_normal_modes_mass_spring_system(m1, m2, k1, k2, k3):
    """
    Calculate the theoretical normal mode frequencies for the two-mass system.

    Solves the generalized eigenvalue problem: K*v = ω²*M*v

    From theory.md:
    ω² = [(m₂(k₁+k₂) + m₁(k₂+k₃)) ± √Δ] / (2m₁m₂)
    where Δ = [m₂(k₁+k₂) + m₁(k₂+k₃)]² - 4m₁m₂[(k₁+k₂)(k₂+k₃) - k₂²]

    Parameters
    ----------
    m1, m2 : float
        Masses
    k1, k2, k3 : float
        Spring constants

    Returns
    -------
    tuple
        (omega1, omega2, f1, f2, mode1, mode2)
        Angular frequencies (rad/s), frequencies (Hz), and mode shapes
    """
    # Mass matrix
    M = np.diag([m1, m2])

    # Stiffness matrix
    K = np.array([[k1 + k2, -k2], [-k2, k2 + k3]])

    # Solve generalized eigenvalue problem
    omega_sq, modes = eigh(K, M)

    # Sort by frequency
    idx = np.argsort(omega_sq)
    omega_sq = omega_sq[idx]
    modes = modes[:, idx]

    omega_sq = np.maximum(omega_sq, 0.0)
    omega1 = np.sqrt(omega_sq[0])
    omega2 = np.sqrt(omega_sq[1])

    # Convert to Hz
    f1 = omega1 / (2 * np.pi)
    f2 = omega2 / (2 * np.pi)

    # Mode shapes (normalized)
    mode1 = modes[:, 0]
    mode2 = modes[:, 1]

    return omega1, omega2, f1, f2, mode1, mode2


def est_freqs_mass_spring_system(t, x1, x2, m1, m2, k1, k2, k3, use_hann_window=True):
    """
    Estimate the normal mode frequencies from simulation data using FFT.

    Projects the simulation data onto the normal mode coordinates and
    performs FFT to extract dominant frequencies.

    Parameters
    ----------
    t : array
        Time array
    x1, x2 : array
        Position arrays for both masses
    m1, m2 : float
        Masses
    k1, k2, k3 : float
        Spring constants
    use_hann_window : bool
        Whether to apply Hann window to reduce spectral leakage

    Returns
    -------
    tuple
        (f1_estimated, f2_estimated) in Hz
    """
    M = np.diag([m1, m2]).astype(float)
    K = np.array([[k1 + k2, -k2], [-k2, k2 + k3]], dtype=float)
    x = np.vstack([np.asarray(x1), np.asarray(x2)])
    f_est = estimate_normal_mode_frequencies(t, x, M, K, use_hann_window)
    return f_est[0], f_est[1]


def coupled_mass_spring_system_animation_with_plots(
    x1_init=0.5,
    x2_init=-0.5,
    v1_init=0.0,
    v2_init=0.0,
    m1=1.0,
    m2=1.0,
    k1=10.0,
    k2=10.0,
    k3=10.0,
    system_length=12.0,
    simulation_time=20.0,
    fps=30,
    save_format="gif",
    save_anim=False,
    filename=None,
    *,
    return_data=False,
    precomputed_data=None,
    show=True,
    trace_length=None,
):
    """
    Simulate and animate a coupled mass-spring system.

    System: Left Wall —[k1]— m1 —[k2]— m2 —[k3]— Right Wall

    Parameters
    ----------
    x1_init, x2_init : float
        Initial displacements from equilibrium
    v1_init, v2_init : float
        Initial velocities
    m1, m2 : float
        Masses
    k1, k2, k3 : float
        Spring constants
    system_length : float
        Distance between walls
    simulation_time : float
        Total simulation time in seconds
    fps : int
        Frames per second for animation
    save_format : str
        Format to save animation ('gif' or 'mp4')
    save_anim : bool
        Whether to save the animation
    filename : str or None
        Filename for saved animation
    return_data : bool
        Return numeric simulation data instead of building an animation.
    precomputed_data : SimulationData or None
        Reuse numeric data when building an animation in the UI.
    show : bool
        Display the Matplotlib figure when not saving. Set to False in Streamlit.
    trace_length : int or None
        Number of recent positions retained in the motion trace.

    Returns
    -------
        anim - animation object
    """

    from utils import SimulationData, validate_simulation_inputs

    validate_simulation_inputs(
        m1=m1,
        m2=m2,
        k1=k1,
        k2=k2,
        k3=k3,
        system_length=system_length,
        simulation_time=simulation_time,
        fps=fps,
    )
    if trace_length is not None and int(trace_length) < 1:
        raise ValueError("trace_length must be at least 1.")

    # =========================================================================
    # NUMERICAL SOLUTION
    # =========================================================================

    # Initial state: [x₁, v₁, x₂, v₂]
    y0 = [x1_init, v1_init, x2_init, v2_init]

    # Time span
    t_span = (0, simulation_time)
    n_frames = max(2, int(simulation_time * fps) + 1)
    t_eval = np.linspace(0, simulation_time, n_frames)

    # Solving the ODE
    if precomputed_data is None:
        y = solve_ode(
            two_mass_spring_derivatives, t_span, y0, t_eval, args=(m1, m2, k1, k2, k3)
        )
    else:
        t_eval = np.asarray(precomputed_data.time, dtype=float)
        y = np.asarray(precomputed_data.state, dtype=float)
        if y.shape != (4, t_eval.size):
            raise ValueError("precomputed_data has an incompatible state shape.")
    x1 = y[0]
    v1 = y[1]
    x2 = y[2]
    v2 = y[3]

    # =========================================================================
    # PHYSICAL QUANTITIES
    # =========================================================================

    # Equilibrium positions
    eq_x1 = system_length / 3.0
    eq_x2 = 2.0 * system_length / 3.0

    # Actual positions
    pos_x1 = eq_x1 + x1
    pos_x2 = eq_x2 + x2

    # Energies
    KE_all = 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2
    PE_spring_all = 0.5 * k1 * x1**2 + 0.5 * k2 * (x2 - x1) ** 2 + 0.5 * k3 * x2**2
    total_E_all = KE_all + PE_spring_all

    # =========================================================================
    # FREQUENCY ANALYSIS
    # =========================================================================

    # Theoretical normal modes
    omega1_theory, omega2_theory, f1_theory, f2_theory, mode1, mode2 = (
        th_normal_modes_mass_spring_system(m1, m2, k1, k2, k3)
    )

    # Numerical frequency estimation
    f1_num, f2_num = est_freqs_mass_spring_system(t_eval, x1, x2, m1, m2, k1, k2, k3)

    data = SimulationData(
        time=np.array(t_eval, copy=True),
        state=np.array(y, copy=True),
        energy=np.array(total_E_all, copy=True),
        theoretical_frequencies=(float(f1_theory), float(f2_theory)),
        estimated_frequencies=(float(f1_num), float(f2_num)),
        metadata={
            "system": "mass_spring",
            "parameters": {
                "x1_init": x1_init,
                "x2_init": x2_init,
                "v1_init": v1_init,
                "v2_init": v2_init,
                "m1": m1,
                "m2": m2,
                "k1": k1,
                "k2": k2,
                "k3": k3,
                "system_length": system_length,
                "simulation_time": simulation_time,
                "fps": fps,
            },
            "ke": np.array(KE_all, copy=True),
            "pe": np.array(PE_spring_all, copy=True),
            "v1": np.array(v1, copy=True),
            "v2": np.array(v2, copy=True),
        },
    )
    if return_data:
        return data

    f1_str = format_frequency(f1_num, decimals=4)
    f2_str = format_frequency(f2_num, decimals=4)

    print("\nNormal Mode Frequencies:")
    print(
        f"  Theoretical: omega1 = {omega1_theory:.4f} rad/s (f1 = {f1_theory:.4f} Hz)"
    )
    print(f"              omega2 = {omega2_theory:.4f} rad/s (f2 = {f2_theory:.4f} Hz)")
    print(f"  Numerical:   f1 ~= {f1_str}, f2 ~= {f2_str}")

    # =========================================================================
    # ANIMATION SETUP
    # =========================================================================

    # Geometry parameters
    left_wall_x = 0.0
    right_wall_x = system_length
    y_position = 0.0
    wall_width = 0.4
    wall_height = 1.5

    # Spring parameters
    relaxed_spring_length = system_length / 3.0
    hook_length = calculate_fixed_hook_length_ms(relaxed_spring_length)
    spring_radius = 0.15
    spring_num_coils = 8

    mass_size_ref = 25
    avg_mass = (m1 + m2) / 2.0
    mass_1_size = max(15, mass_size_ref * np.cbrt(m1 / avg_mass))
    mass_2_size = max(15, mass_size_ref * np.cbrt(m2 / avg_mass))

    # ── Dark-theme palette ────────────────────────────────────────────────
    COLOR_MASS1 = PALETTE["MASS1"]
    COLOR_MASS2 = PALETTE["MASS2"]
    COLOR_SPRING = PALETTE["SPRING"]
    COLOR_GUIDE = PALETTE["GUIDE"]
    COLOR_GRID = PALETTE["GRID"]
    COLOR_BG_MAIN = PALETTE["BG_MAIN"]
    COLOR_BG_X1 = PALETTE["BG_TH1"]
    COLOR_BG_X2 = PALETTE["BG_TH3"]
    COLOR_BG_PH = PALETTE["BG_PHASE"]

    # =========================================================================
    # ANIMATION SETUP
    # =========================================================================

    # Geometry parameters
    left_wall_x = 0.0
    right_wall_x = system_length
    y_position = 0.0
    wall_width = 0.4
    wall_height = 1.5

    # Spring parameters
    relaxed_spring_length = system_length / 3.0
    hook_length = calculate_fixed_hook_length_ms(relaxed_spring_length)
    spring_radius = 0.15
    spring_num_coils = 8

    mass_size_ref = 25
    avg_mass = (m1 + m2) / 2.0
    mass_1_size = max(15, mass_size_ref * np.cbrt(m1 / avg_mass))
    mass_2_size = max(15, mass_size_ref * np.cbrt(m2 / avg_mass))

    # ── Dark-theme palette ────────────────────────────────────────────────
    COLOR_MASS1 = PALETTE["MASS1"]
    COLOR_MASS2 = PALETTE["MASS2"]
    COLOR_SPRING = PALETTE["SPRING"]
    COLOR_GUIDE = PALETTE["GUIDE"]
    COLOR_GRID = PALETTE["GRID"]
    COLOR_BG_MAIN = PALETTE["BG_MAIN"]
    COLOR_BG_X1 = PALETTE["BG_TH1"]
    COLOR_BG_X2 = PALETTE["BG_TH3"]
    COLOR_BG_PH = PALETTE["BG_PHASE"]

    # Figure setup
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(11, 8))
    fig.set_facecolor("black")
    gs = GridSpec(
        2,
        3,
        left=0.08,
        right=0.92,
        top=0.9,
        bottom=0.06,
        height_ratios=[1.5, 1],
        hspace=0.28,
        wspace=0.3,
    )

    # Animation subplot
    ax = fig.add_subplot(gs[0, :])
    ax.set_facecolor(COLOR_BG_MAIN)

    # Time domain subplots
    ax_x1 = fig.add_subplot(gs[1, 0])
    ax_x2 = fig.add_subplot(gs[1, 1])
    ax_phase = fig.add_subplot(gs[1, 2])
    ax_x1.set_facecolor(COLOR_BG_X1)
    ax_x2.set_facecolor(COLOR_BG_X2)
    ax_phase.set_facecolor(COLOR_BG_PH)

    padding = 0.5
    x_min = left_wall_x - padding
    x_max = right_wall_x + padding
    y_min = y_position - wall_height / 2 - padding
    y_max = y_position + wall_height / 2 + padding

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.1, color=COLOR_GRID)
    for sp in ax.spines.values():
        sp.set_edgecolor(PALETTE["GUIDE"])

    if abs(x1_init - x2_init) < 1e-6:
        motion_type = "In-Phase Mode (Both move together)"
    elif abs(x1_init + x2_init) < 1e-6:
        motion_type = "Out-of-Phase Mode (Opposite directions)"
    elif x1_init == 0.0 or x2_init == 0.0:
        motion_type = "Energy Transfer (Beats Phenomenon)"
    else:
        motion_type = "Mixed Mode (Superposition)"

    title_text = (
        f"Coupled Mass-Spring System: {motion_type}\n"
        f"Normal Modes: $\\omega_1$={omega1_theory:.3f} rad/s, $\\omega_2$={omega2_theory:.3f} rad/s"
    )
    ax.set_title(title_text, fontsize=14, fontweight="bold", pad=10, color="white")

    draw_wall(ax, left_wall_x, y_position, wall_width, wall_height)
    draw_wall(ax, right_wall_x, y_position, wall_width, wall_height)

    ax.axvline(eq_x1, color=COLOR_GUIDE, linestyle="--", lw=1, alpha=0.6, zorder=0)
    ax.axvline(eq_x2, color=COLOR_GUIDE, linestyle="--", lw=1, alpha=0.6, zorder=0)
    ax.axhline(y_position, color=COLOR_GUIDE, linestyle=":", lw=1, alpha=0.4, zorder=0)

    (spring1_line,) = ax.plot([], [], color=COLOR_SPRING, lw=2.0, zorder=2, alpha=0.95)
    (spring2_line,) = ax.plot([], [], color=COLOR_SPRING, lw=2.0, zorder=2, alpha=0.95)
    (spring3_line,) = ax.plot([], [], color=COLOR_SPRING, lw=2.0, zorder=2, alpha=0.95)

    (mass1_plot,) = ax.plot(
        [],
        [],
        "o",
        color=COLOR_MASS1,
        markersize=mass_1_size,
        zorder=10,
        markeredgecolor=PALETTE["EDGE_MASS1"],
        markeredgewidth=1.5,
    )
    (mass2_plot,) = ax.plot(
        [],
        [],
        "o",
        color=COLOR_MASS2,
        markersize=mass_2_size,
        zorder=10,
        markeredgecolor=PALETTE["EDGE_MASS2"],
        markeredgewidth=1.5,
    )

    trace_length = (
        int(trace_length) if trace_length is not None else min(300, n_frames // 2)
    )
    trace1_x, trace1_y = [], []
    trace2_x, trace2_y = [], []
    (trace1_line,) = ax.plot([], [], color=COLOR_MASS1, alpha=0.35, lw=1.2, zorder=1)
    (trace2_line,) = ax.plot([], [], color=COLOR_MASS2, alpha=0.35, lw=1.2, zorder=1)

    # =========================================================================
    # SUBPLOTS SETUP
    # =========================================================================

    # x1 vs t
    ax_x1.set_xlim(0, 1.15 * simulation_time)
    ax_x1.set_ylim(min(x1) * 1.1, max(x1) * 1.1)
    style_dark_subplot(ax_x1, "Time (s)", "$x_1$ (m)", "Displacement $x_1$ vs Time")
    (line_x1,) = ax_x1.plot([], [], color=COLOR_MASS1, lw=1.5)
    (marker_x1,) = ax_x1.plot([], [], "o", color=COLOR_MASS1, markersize=8)

    # x2 vs t
    ax_x2.set_xlim(0, 1.15 * simulation_time)
    ax_x2.set_ylim(min(x2) * 1.1, max(x2) * 1.1)
    style_dark_subplot(ax_x2, "Time (s)", "$x_2$ (m)", "Displacement $x_2$ vs Time")
    (line_x2,) = ax_x2.plot([], [], color=COLOR_MASS2, lw=1.5)
    (marker_x2,) = ax_x2.plot([], [], "o", color=COLOR_MASS2, markersize=8)

    # Phase Space
    x_range = min(min(x1), min(x2)) * 1.2, max(max(x1), max(x2)) * 1.2
    v_range = min(min(v1), min(v2)) * 1.2, max(max(v1), max(v2)) * 1.2
    ax_phase.set_xlim(x_range)
    ax_phase.set_ylim(v_range)
    style_dark_subplot(
        ax_phase,
        "Position (m)",
        "Velocity (m/s)",
        "Phase Space ($\\dot{x}$ vs $x$)",
    )
    (line_phase1,) = ax_phase.plot(
        [], [], color=COLOR_MASS1, lw=1.5, label="Mass 1", alpha=0.7
    )
    (line_phase2,) = ax_phase.plot(
        [], [], color=COLOR_MASS2, lw=1.5, label="Mass 2", alpha=0.7
    )
    (marker_phase1,) = ax_phase.plot([], [], "o", color=COLOR_MASS1, markersize=8)
    (marker_phase2,) = ax_phase.plot([], [], "o", color=COLOR_MASS2, markersize=8)
    ax_phase.legend(
        loc="upper right",
        fontsize=8,
        facecolor=PALETTE["BG_MAIN"],
        edgecolor=PALETTE["GUIDE"],
        labelcolor="white",
    )

    # =========================================================================
    # ANIMATION FUNCTION
    # =========================================================================

    def init():
        """Initialize animation"""
        spring1_line.set_data([], [])
        spring2_line.set_data([], [])
        spring3_line.set_data([], [])
        mass1_plot.set_data([], [])
        mass2_plot.set_data([], [])
        trace1_line.set_data([], [])
        trace2_line.set_data([], [])

        line_x1.set_data([], [])
        line_x2.set_data([], [])
        marker_x1.set_data([], [])
        marker_x2.set_data([], [])

        line_phase1.set_data([], [])
        line_phase2.set_data([], [])
        marker_phase1.set_data([], [])
        marker_phase2.set_data([], [])

        return (
            spring1_line,
            spring2_line,
            spring3_line,
            mass1_plot,
            mass2_plot,
            trace1_line,
            trace2_line,
            line_x1,
            line_x2,
            marker_x1,
            marker_x2,
            line_phase1,
            line_phase2,
            marker_phase1,
            marker_phase2,
        )

    def update(frame):
        """Update animation for each frame"""
        x1_curr = pos_x1[frame]
        x2_curr = pos_x2[frame]

        # Spring 1: Left wall to mass 1
        x_s1, y_s1 = draw_spring_with_hook(
            (left_wall_x + wall_width / 2, y_position),
            (x1_curr, y_position),
            spring_num_coils,
            spring_radius,
            hook_length,
        )
        spring1_line.set_data(x_s1, y_s1)

        # Spring 2: Mass 1 to mass 2
        x_s2, y_s2 = draw_spring_with_hook(
            (x1_curr, y_position),
            (x2_curr, y_position),
            spring_num_coils,
            spring_radius,
            hook_length,
        )
        spring2_line.set_data(x_s2, y_s2)

        # Spring 3: Mass 2 to right wall
        x_s3, y_s3 = draw_spring_with_hook(
            (x2_curr, y_position),
            (right_wall_x - wall_width / 2, y_position),
            spring_num_coils,
            spring_radius,
            hook_length,
        )
        spring3_line.set_data(x_s3, y_s3)

        mass1_plot.set_data([x1_curr], [y_position])
        mass2_plot.set_data([x2_curr], [y_position])

        trace1_x.append(x1_curr)
        trace1_y.append(y_position)
        trace2_x.append(x2_curr)
        trace2_y.append(y_position)

        if len(trace1_x) > trace_length:
            trace1_x.pop(0)
            trace1_y.pop(0)
            trace2_x.pop(0)
            trace2_y.pop(0)

        trace1_line.set_data(trace1_x, trace1_y)
        trace2_line.set_data(trace2_x, trace2_y)

        current_t = t_eval[: frame + 1]
        current_x1 = x1[: frame + 1]
        current_x2 = x2[: frame + 1]

        line_x1.set_data(current_t, current_x1)
        line_x2.set_data(current_t, current_x2)

        marker_x1.set_data([t_eval[frame]], [x1[frame]])
        marker_x2.set_data([t_eval[frame]], [x2[frame]])

        current_v1 = v1[: frame + 1]
        current_v2 = v2[: frame + 1]

        line_phase1.set_data(current_x1, current_v1)
        line_phase2.set_data(current_x2, current_v2)

        marker_phase1.set_data([x1[frame]], [v1[frame]])
        marker_phase2.set_data([x2[frame]], [v2[frame]])

        return (
            spring1_line,
            spring2_line,
            spring3_line,
            mass1_plot,
            mass2_plot,
            trace1_line,
            trace2_line,
            line_x1,
            line_x2,
            marker_x1,
            marker_x2,
            line_phase1,
            line_phase2,
            marker_phase1,
            marker_phase2,
        )

    print("Creating animation...")
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(t_eval),
        interval=int(1000 / fps),
        blit=False,
        repeat=False,
    )
    if save_anim:
        if filename is None:
            filename = (
                f"mass_spring_m1={m1}_m2={m2}_k1={k1}_k2={k2}_k3={k3}.{save_format}"
            )
        save_animation(anim, SAVE_DIR / filename, fps, save_format)
        plt.close(fig)
    elif show:
        plt.show()

    return anim


def main_mass_spring_simulation_with_plots():
    """Main function to run mass-spring simulation with user-defined parameters"""
    print("\n" + "=" * 60)
    print("COUPLED MASS-SPRING SYSTEM SIMULATION WITH PLOTS")
    print("=" * 60)

    default_params = {
        "x1_init": 0.1,
        "x2_init": 0.0,
        "v1_init": 0.0,
        "v2_init": 0.0,
        "m1": 1.0,
        "m2": 1.0,
        "k1": 10.0,
        "k2": 0.5,
        "k3": 10.0,
        "system_length": 12.0,
        "simulation_time": 15.0,
        "fps": 30,
        "save_format": "mp4",
    }

    use_defaults = input("Use default parameters? (y/n): ").strip().lower() == "y"
    if use_defaults:
        params = default_params
    else:
        params = {}
        for key, val in default_params.items():
            user_input = input(f"Enter value for {key} (default={val}): ").strip()
            if user_input == "":
                params[key] = val
            else:
                try:
                    params[key] = float(user_input)
                except ValueError:
                    print(f"Invalid input for {key}. Using default value {val}.")
                    params[key] = val

    save_animation = input("Save animation? (y/n): ").strip().lower() == "y"
    filename = None
    if save_animation:
        base_input = input(
            "Enter base filename (without extension) or press Enter for default: "
        ).strip()
        if base_input:
            from pathlib import Path

            base_input = Path(base_input).stem
            filename = f"{base_input}.{params['save_format']}"

    print("\nStarting simulation with parameters:")
    print(f"    Initial Displacement of mass 1 (x1_init): {params['x1_init']} m")
    print(f"    Initial Displacement of mass 2 (x2_init): {params['x2_init']} m")
    print(f"    Initial Velocity of mass 1 (v1_init): {params['v1_init']} m/s")
    print(f"    Initial Velocity of mass 2 (v2_init): {params['v2_init']} m/s")
    print(f"    Mass 1 (m1): {params['m1']} kg")
    print(f"    Mass 2 (m2): {params['m2']} kg")
    print(f"    Spring Constant k1: {params['k1']} N/m")
    print(f"    Spring Constant k2: {params['k2']} N/m")
    print(f"    Spring Constant k3: {params['k3']} N/m")
    print(f"    System Length: {params['system_length']} m")
    print(f"    Simulation Time: {params['simulation_time']} s")
    print(f"    Frames per Second (fps): {params['fps']}")

    anim = coupled_mass_spring_system_animation_with_plots(
        x1_init=params["x1_init"],
        x2_init=params["x2_init"],
        v1_init=params["v1_init"],
        v2_init=params["v2_init"],
        m1=params["m1"],
        m2=params["m2"],
        k1=params["k1"],
        k2=params["k2"],
        k3=params["k3"],
        system_length=params["system_length"],
        simulation_time=params["simulation_time"],
        fps=int(params["fps"]),
        save_format=params["save_format"],
        save_anim=save_animation,
        filename=filename,
    )

    return anim


if __name__ == "__main__":
    main_mass_spring_simulation_with_plots()
    print("\nSimulation complete.")
