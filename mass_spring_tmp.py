import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh

# Matplotlib visualization
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from double_pendulum import draw_spring_with_hook, SAVE_DIR


def calculate_fixed_hook_length_mass_spring(relaxed_spring_length, hook_ratio=0.12):
    """
    Calculate the fixed hook length based on the relaxed spring length.
    """
    coiled_spring_len = relaxed_spring_length / (1 + 2 * hook_ratio)
    hook_len = hook_ratio * coiled_spring_len
    return hook_len


def draw_wall(ax, x_position, y_center, wall_width, wall_height, color="#966919"):
    """
    Draw a vertical wall at the specified x position.
    """
    rect = Rectangle(
        (x_position - wall_width / 2, y_center - wall_height / 2),
        wall_width,
        wall_height,
        color=color,
        zorder=5,
    )
    ax.add_patch(rect)
    return rect


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
    t = np.asarray(t)
    dt = t[1] - t[0]
    n = t.size

    x = np.vstack([np.asarray(x1), np.asarray(x2)])

    x = x - x.mean(axis=1, keepdims=True)

    M = np.diag([m1, m2])
    K = np.array([[k1 + k2, -k2], [-k2, k2 + k3]])

    omega_sq, V = eigh(K, M)

    q = V.T @ (M @ x)

    if use_hann_window:
        window = np.hanning(n)
        q = q * window

    freqs = np.fft.rfftfreq(n, dt)
    Q = np.fft.rfft(q, axis=1)
    amp = np.abs(Q)

    rms = np.sqrt(np.mean(q**2, axis=1))
    rms_rel = rms / (np.max(rms) + 1e-30)
    f_est = []
    for i in range(2):
        if rms_rel[i] < 1e-3:
            f_est.append(0.0)
            continue

        peak_idx = np.argmax(amp[i, 1:]) + 1
        f_est.append(freqs[peak_idx])

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

    Returns
    -------
        anim - animation object
    """

    # =========================================================================
    # NUMERICAL SOLUTION
    # =========================================================================

    # Initial state: [x₁, v₁, x₂, v₂]
    y0 = [x1_init, v1_init, x2_init, v2_init]

    # Time span
    t_span = (0, simulation_time)
    n_frames = int(simulation_time * fps) + 1
    t_eval = np.linspace(0, simulation_time, n_frames)

    # Solving the ODE
    print("Solving differential equations...")
    solution = solve_ivp(
        two_mass_spring_derivatives,
        t_span,
        y0,
        args=(m1, m2, k1, k2, k3),
        method="RK45",
        rtol=1e-5,
        atol=1e-7,
        dense_output=True,
    )

    if solution.sol is None:
        print("ERROR: ODE solver failed!")
        return None, None

    y = solution.sol(t=t_eval)
    x1 = y[0]
    v1 = y[1]
    x2 = y[2]
    v2 = y[3]

    print(f"Solution computed: {len(t_eval)} time steps")

    # =========================================================================
    # PHYSICAL QUANTITIES
    # =========================================================================

    # Equilibrium positions
    eq_x1 = system_length / 3.0
    eq_x2 = 2.0 * system_length / 3.0

    # Actual positions
    pos_x1 = eq_x1 + x1
    pos_x2 = eq_x2 + x2

    # Spring extensions/compressions
    spring1_ext = x1
    spring2_ext = x2 - x1
    spring3_ext = -x2

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

    def _fmt_freq(f):
        return f"{f:.4f} Hz" if f > 0 else "N/A (Not Excited)"

    f1_str = _fmt_freq(f1_num)
    f2_str = _fmt_freq(f2_num)

    print("\nNormal Mode Frequencies:")
    print(f"  Theoretical: ω₁ = {omega1_theory:.4f} rad/s (f₁ = {f1_theory:.4f} Hz)")
    print(f"              ω₂ = {omega2_theory:.4f} rad/s (f₂ = {f2_theory:.4f} Hz)")
    print(f"  Numerical:   f₁ ≈ {f1_str}, f₂ ≈ {f2_str}")

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
    hook_length = calculate_fixed_hook_length_mass_spring(relaxed_spring_length)
    spring_radius = 0.15
    spring_num_coils = 8

    mass_size_ref = 25
    avg_mass = (m1 + m2) / 2.0
    mass_1_size = max(15, mass_size_ref * np.cbrt(m1 / avg_mass))
    mass_2_size = max(15, mass_size_ref * np.cbrt(m2 / avg_mass))

    # ── Dark-theme palette (matches triple_pendulum.py / double_pendulum.py) ────
    COLOR_MASS1 = "#FF4D6D"  # hot-pink-red   — mass 1
    COLOR_MASS2 = "#00EAFF"  # neon cyan        — mass 2
    COLOR_SPRING = "#708090"  # slate grey springs
    COLOR_GUIDE = "#555555"  # dim guide lines
    COLOR_GRID = "#444444"
    COLOR_BG_MAIN = "#091217"  # main animation panel
    COLOR_BG_X1 = "#110000"  # x1 panel
    COLOR_BG_X2 = "#00001A"  # x2 panel
    COLOR_BG_PH = "#091217"  # phase space panel

    # Figure setup
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 10))
    fig.set_facecolor("black")
    gs = GridSpec(
        2,
        3,
        left=0.08,
        right=0.92,
        top=0.92,
        bottom=0.06,
        height_ratios=[1.5, 1],
        hspace=0.77,
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
        sp.set_edgecolor("#555555")

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
        markeredgecolor="#991133",
        markeredgewidth=1.5,
    )
    (mass2_plot,) = ax.plot(
        [],
        [],
        "o",
        color=COLOR_MASS2,
        markersize=mass_2_size,
        zorder=10,
        markeredgecolor="#008899",
        markeredgewidth=1.5,
    )

    trace_length = min(300, n_frames // 2)
    trace1_x, trace1_y = [], []
    trace2_x, trace2_y = [], []
    (trace1_line,) = ax.plot([], [], color=COLOR_MASS1, alpha=0.35, lw=1.2, zorder=1)
    (trace2_line,) = ax.plot([], [], color=COLOR_MASS2, alpha=0.35, lw=1.2, zorder=1)

    # =========================================================================
    # SUBPLOTS SETUP
    # =========================================================================

    def _style_subplot(a, xlabel, ylabel, title):
        """Apply consistent dark-theme styling to a subplot."""
        a.set_xlabel(xlabel, fontsize=10, color="white")
        a.set_ylabel(ylabel, fontsize=10, color="white")
        a.set_title(title, fontsize=12, fontweight="bold", color="white")
        a.tick_params(colors="white", which="both")
        for sp in a.spines.values():
            sp.set_edgecolor("#555555")
        a.grid(True, alpha=0.3, color=COLOR_GRID)

    # x1 vs t
    ax_x1.set_xlim(0, 1.15 * simulation_time)
    ax_x1.set_ylim(min(x1) * 1.1, max(x1) * 1.1)
    _style_subplot(ax_x1, "Time (s)", "$x_1$ (m)", "Displacement $x_1$ vs Time")
    (line_x1,) = ax_x1.plot([], [], color=COLOR_MASS1, lw=1.5)
    (marker_x1,) = ax_x1.plot([], [], "o", color=COLOR_MASS1, markersize=8)

    # x2 vs t
    ax_x2.set_xlim(0, 1.15 * simulation_time)
    ax_x2.set_ylim(min(x2) * 1.1, max(x2) * 1.1)
    _style_subplot(ax_x2, "Time (s)", "$x_2$ (m)", "Displacement $x_2$ vs Time")
    (line_x2,) = ax_x2.plot([], [], color=COLOR_MASS2, lw=1.5)
    (marker_x2,) = ax_x2.plot([], [], "o", color=COLOR_MASS2, markersize=8)

    # Phase Space
    x_range = min(min(x1), min(x2)) * 1.2, max(max(x1), max(x2)) * 1.2
    v_range = min(min(v1), min(v2)) * 1.2, max(max(v1), max(v2)) * 1.2
    ax_phase.set_xlim(x_range)
    ax_phase.set_ylim(v_range)
    _style_subplot(
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
        facecolor="#1a1a2e",
        edgecolor="#555555",
        labelcolor="white",
    )

    # =========================================================================
    # TEXT ANNOTATIONS
    # =========================================================================

    # Timer
    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        color="#FF9F1C",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#1a1000",
            alpha=0.88,
            edgecolor="#CC8800",
            linewidth=1.5,
        ),
    )

    # System info box
    info_text = ax.text(
        0.01,
        -0.03,
        "",
        fontsize=9,
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        color="#AAFF00",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#1a3300",
            alpha=0.88,
            edgecolor="#77BB00",
            linewidth=1.5,
        ),
    )

    # Dynamic info box
    dynamic_text = ax.text(
        0.38,
        -0.03,
        "",
        fontsize=9,
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        color=COLOR_MASS1,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#330011",
            alpha=0.88,
            edgecolor="#CC2244",
            linewidth=1.5,
        ),
    )

    # Frequency comparison box
    freq_text = ax.text(
        0.82,
        -0.03,
        "",
        fontsize=9,
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        color=COLOR_MASS2,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#003366",
            alpha=0.88,
            edgecolor="#00AACC",
            linewidth=1.5,
        ),
    )

    # Static system info
    info_str = (
        "SYSTEM PARAMETERS\n"
        "─────────────────\n"
        f"$m_1$ = {m1:.2f} kg,"
        f"$m_2$ = {m2:.2f} kg\n"
        f"$k_1$ = {k1:.2f} N/m,"
        f"$k_2$ = {k2:.2f} N/m,"
        f"$k_3$ = {k3:.2f} N/m\n"
        f"\n"
        f"INITIAL CONDITIONS\n"
        f"──────────────────\n"
        f"$x_1$(0) = {x1_init:+.3f} m,"
        f"$x_2$(0) = {x2_init:+.3f} m\n"
        f"$\\dot{{x}}_1$(0) = {v1_init:+.3f} m/s,"
        f"$\\dot{{x}}_2$(0) = {v2_init:+.3f} m/s"
    )
    info_text.set_text(info_str)

    # Static frequency comparison
    freq_str = (
        "FREQUENCY ANALYSIS\n"
        "──────────────────\n"
        f"Mode 1 (Low):\n"
        f"  Theory: {f1_theory:.4f} Hz\n"
        f"  Simul.: {f1_str}\n"
        f"\n"
        f"Mode 2 (High):\n"
        f"  Theory: {f2_theory:.4f} Hz\n"
        f"  Simul.: {f2_str}\n"
    )
    freq_text.set_text(freq_str)

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
        line_phase1.set_data([], [])
        line_phase2.set_data([], [])
        marker_x1.set_data([], [])
        marker_x2.set_data([], [])
        marker_phase1.set_data([], [])
        marker_phase2.set_data([], [])

        time_text.set_text("")
        dynamic_text.set_text("")
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
            line_phase1,
            line_phase2,
            marker_x1,
            marker_x2,
            marker_phase1,
            marker_phase2,
            time_text,
            dynamic_text,
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
        current_v1 = v1[: frame + 1]
        current_v2 = v2[: frame + 1]

        line_x1.set_data(current_t, current_x1)
        line_x2.set_data(current_t, current_x2)
        line_phase1.set_data(current_x1, current_v1)
        line_phase2.set_data(current_x2, current_v2)

        marker_x1.set_data([t_eval[frame]], [x1[frame]])
        marker_x2.set_data([t_eval[frame]], [x2[frame]])
        marker_phase1.set_data([x1[frame]], [v1[frame]])
        marker_phase2.set_data([x2[frame]], [v2[frame]])

        time_text.set_text(f"Time: {t_eval[frame]:.2f} s")

        dynamic_str = (
            "CURRENT STATE\n"
            "─────────────\n"
            f"$x_1$ = {x1[frame]:+.3f} m, "
            f"$x_2$ = {x2[frame]:+.3f} m\n"
            f"$\\dot{{x}}_1$ = {v1[frame]:+.3f} m/s,"
            f"$\\dot{{x}}_2$ = {v2[frame]:+.3f} m/s\n"
            f"\n"
            f"SPRING EXTENSIONS\n"
            f"─────────────────\n"
            f"$\\Delta s_1$ = {spring1_ext[frame]:+.3f} m, "
            f"$\\Delta s_2$ = {spring2_ext[frame]:+.3f} m, "
            f"$\\Delta s_3$ = {spring3_ext[frame]:+.3f} m\n"
            f"\n"
            f"ENERGY\n"
            f"──────\n"
            f"KE  = {KE_all[frame]:.3f} J, "
            f"PE  = {PE_spring_all[frame]:.3f} J, "
            f"Tot = {total_E_all[frame]:.3f} J"
        )
        dynamic_text.set_text(dynamic_str)

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
            line_phase1,
            line_phase2,
            marker_x1,
            marker_x2,
            marker_phase1,
            marker_phase2,
            time_text,
            dynamic_text,
        )

    print("Creating animation...")
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=int(1000 / fps),
        blit=False,
        repeat=False,
    )

    if save_anim:
        if filename is None:
            filename = (
                f"mass_spring_m1={m1}_m2={m2}_k1={k1}_k2={k2}_k3={k3}.{save_format}"
            )

        filepath = SAVE_DIR / filename
        try:
            print(f"Saving animation to {filepath.resolve()}...")
            if save_format.lower() == "mp4":
                writer = FFMpegWriter(
                    fps=fps,
                    codec="libx264",
                    bitrate=8000,
                    extra_args=[
                        "-pix_fmt",
                        "yuv420p",
                        "-profile:v",
                        "high",
                        "-level",
                        "4.1",
                        "-movflags",
                        "+faststart",
                    ],
                )
                anim.save(filepath, writer=writer)
            else:
                anim.save(filepath, writer="ffmpeg", fps=fps, dpi=120)
            print(f"Animation saved: {filepath}")
            plt.close(fig)
        except Exception as e:
            print(f"Error saving animation: {e}")
    else:
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
    print(
        "You can modify the parameters in the 'main_mass_spring_simulation_with_plots' function to run different scenarios."
    )
