import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

from utils import (
    PALETTE,
    SAVE_DIR,
    calculate_fixed_hook_length_three_pendulum,
    calculate_pendulum_mass_positions,
    calculate_plot_limits_3,
    draw_ceiling,
    draw_pendulum_string,
    draw_spring_with_hook,
    estimate_normal_mode_frequencies,
    format_frequency,
    save_animation,
    solve_ode,
    style_dark_subplot,
)


def draw_three_coupled_pendulum(
    theta_1=0,
    theta_2=0,
    theta_3=10,
    ceiling_length=12,
    string_length=12,
    separation_ratio=0.6,
    num_coils=8,
    mass_size=25,
    figsize=(12, 8),
    padding=0.15,
):
    """
    Draw a three coupled pendulum system with two springs.
    m1 connected to m2 by spring 1, m2 connected to m3 by spring 2.

    Parameters:
    -----------
    theta_1, theta_2, theta_3 : float
        Angular displacements in degrees for the three pendulums
    ceiling_length : float
        Length of the ceiling support
    string_length : float
        Length of each pendulum string
    separation_ratio : float
        Ratio determining spacing between anchors
    num_coils : int
        Number of coils in each spring
    mass_size : float
        Size of the mass markers
    figsize : tuple
        Figure size
    padding : float
        Padding around the plot
    """
    # System dimensions
    ceiling_height = 0.05 * ceiling_length
    anchor_separation = separation_ratio * ceiling_length

    # Anchor positions (evenly spaced)
    left_anchor_x = -anchor_separation
    middle_anchor_x = 0
    right_anchor_x = anchor_separation
    anchor_y = 0

    # Mass positions
    left_mass_x, left_mass_y = calculate_pendulum_mass_positions(
        left_anchor_x, string_length, theta_1
    )
    middle_mass_x, middle_mass_y = calculate_pendulum_mass_positions(
        middle_anchor_x, string_length, theta_2
    )
    right_mass_x, right_mass_y = calculate_pendulum_mass_positions(
        right_anchor_x, string_length, theta_3
    )

    # Hook length based on relaxed state
    hook_length = calculate_fixed_hook_length_three_pendulum(
        ceiling_length, string_length, separation_ratio
    )

    # Spring radius
    spring_radius = 0.03 * ceiling_length

    # Figure
    fig, ax = plt.subplots(figsize=figsize)

    # Ceiling
    ceiling_width = 2 * anchor_separation + ceiling_length * 0.3
    draw_ceiling(ax, ceiling_width, ceiling_height)

    # Reference equilibrium lines
    ax.axvline(x=left_anchor_x, color="gray", linestyle="--", lw=1, alpha=0.7)
    ax.axvline(x=middle_anchor_x, color="gray", linestyle="--", lw=1, alpha=0.7)
    ax.axvline(x=right_anchor_x, color="gray", linestyle="--", lw=1, alpha=0.7)

    # Pendulum strings
    draw_pendulum_string(ax, left_anchor_x, anchor_y, left_mass_x, left_mass_y)
    draw_pendulum_string(ax, middle_anchor_x, anchor_y, middle_mass_x, middle_mass_y)
    draw_pendulum_string(ax, right_anchor_x, anchor_y, right_mass_x, right_mass_y)

    # First spring (m1 to m2)
    x_spring1, y_spring1 = draw_spring_with_hook(
        (left_mass_x, left_mass_y),
        (middle_mass_x, middle_mass_y),
        num_coils=num_coils,
        radius=spring_radius,
        hook_length=hook_length,
    )
    ax.plot(x_spring1, y_spring1, color="darkgrey", lw=2)

    # Second spring (m2 to m3)
    x_spring2, y_spring2 = draw_spring_with_hook(
        (middle_mass_x, middle_mass_y),
        (right_mass_x, right_mass_y),
        num_coils=num_coils,
        radius=spring_radius,
        hook_length=hook_length,
    )
    ax.plot(x_spring2, y_spring2, color="darkgrey", lw=2)

    # Masses
    ax.plot(left_mass_x, left_mass_y, "ro", ms=mass_size, label="$m_1$")
    ax.plot(middle_mass_x, middle_mass_y, "go", ms=mass_size, label="$m_2$")
    ax.plot(right_mass_x, right_mass_y, "bo", ms=mass_size, label="$m_3$")

    # Plot limits
    xlim, ylim = calculate_plot_limits_3(
        ceiling_length,
        left_mass_x,
        right_mass_x,
        left_mass_y,
        right_mass_y,
        padding,
        middle_mass_x,
        middle_mass_y,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(
        f"Three Coupled Pendulums: $\\theta_1 = {theta_1}^\\circ, \\theta_2 = {theta_2}^\\circ, \\theta_3 = {theta_3}^\\circ$"
    )
    ax.legend(loc="upper right", ncol=3, fontsize=8, markerscale=0.5)

    return fig, ax


def three_coupled_pendulum_derivatives(t, y, g, L, k1, k2, m1, m2, m3):
    """
    Compute the derivatives for the three coupled pendulum system.

    Uses the exact non-linear equations of motion:
    θ̈₁ = -(g/L)sin(θ₁) + (k₁/m₁)(sin(θ₂) - sin(θ₁))cos(θ₁)
    θ̈₂ = -(g/L)sin(θ₂) - (k₁/m₂)(sin(θ₂) - sin(θ₁))cos(θ₂) + (k₂/m₂)(sin(θ₃) - sin(θ₂))cos(θ₂)
    θ̈₃ = -(g/L)sin(θ₃) - (k₂/m₃)(sin(θ₃) - sin(θ₂))cos(θ₃)

    Parameters
    ----------
    t : float
        Time (not used explicitly, required by solve_ivp)
    y : array-like
        State vector [θ₁, ω₁, θ₂, ω₂, θ₃, ω₃] where ω = dθ/dt
    g : float
        Gravitational acceleration
    L : float
        Length of pendulum rods
    k1, k2 : float
        Spring constants (k1 connects m1-m2, k2 connects m2-m3)
    m1, m2, m3 : float
        Masses of the three pendulum bobs

    Returns
    -------
    list
        Derivatives [dθ₁/dt, dω₁/dt, dθ₂/dt, dω₂/dt, dθ₃/dt, dω₃/dt]
    """
    theta1, omega1, theta2, omega2, theta3, omega3 = y

    # Spring extension terms
    delta_sin_12 = np.sin(theta2) - np.sin(theta1)
    delta_sin_23 = np.sin(theta3) - np.sin(theta2)

    # Angular accelerations (exact non-linear equations)
    d_theta1 = omega1
    d_omega1 = -(g / L) * np.sin(theta1) + (k1 / m1) * delta_sin_12 * np.cos(theta1)

    d_theta2 = omega2
    d_omega2 = (
        -(g / L) * np.sin(theta2)
        - (k1 / m2) * delta_sin_12 * np.cos(theta2)
        + (k2 / m2) * delta_sin_23 * np.cos(theta2)
    )

    d_theta3 = omega3
    d_omega3 = -(g / L) * np.sin(theta3) - (k2 / m3) * delta_sin_23 * np.cos(theta3)

    return [d_theta1, d_omega1, d_theta2, d_omega2, d_theta3, d_omega3]


def th_normal_modes_triple_pendulum(g, L, k1, k2, m1, m2, m3):
    """
    Calculate the theoretical normal mode frequencies for the linearized three-pendulum system.

    For equal masses (m1 = m2 = m3 = m) and equal springs (k1 = k2 = k):
        ω₁² = g/L               (sloshing mode - all move together)
        ω₂² = g/L + k/m         (anti-symmetric mode)
        ω₃² = g/L + 3k/m        (breathing mode)

    For unequal masses/springs, we solve the eigenvalue problem.

    Parameters
    ----------
    g, L, k1, k2, m1, m2, m3 : float
        System parameters

    Returns
    -------
    tuple
        (omega1, omega2, omega3, f1, f2, f3) - angular frequencies and frequencies in Hz
    """
    # Construct the dynamical matrix for the linearized system
    # The linearized equations give: M*θ̈ = -K*θ
    # where θ = [θ₁, θ₂, θ₃]

    # Stiffness matrix K (from potential energy)
    omega0_sq = g / L

    K = np.array(
        [
            [omega0_sq + k1 / m1, -k1 / m1, 0],
            [-k1 / m2, omega0_sq + (k1 + k2) / m2, -k2 / m2],
            [0, -k2 / m3, omega0_sq + k2 / m3],
        ]
    )

    # Eigenvalues give ω²
    eigenvalues = np.linalg.eigvals(K)

    # Sort eigenvalues (ω² values)
    omega_sq = np.sort(np.real(eigenvalues))

    omega1 = np.sqrt(omega_sq[0])  # Lowest frequency (sloshing)
    omega2 = np.sqrt(omega_sq[1])  # Middle frequency (anti-symmetric)
    omega3 = np.sqrt(omega_sq[2])  # Highest frequency (breathing)

    # Convert to Hz
    f1 = omega1 / (2 * np.pi)
    f2 = omega2 / (2 * np.pi)
    f3 = omega3 / (2 * np.pi)

    return omega1, omega2, omega3, f1, f2, f3


def est_freqs_triple_pendulum(
    t, theta1, theta2, theta3, g, L, k1, k2, m1, m2, m3, use_hann_window=True
):
    """
    Estimate the 3 normal-mode frequencies for unequal masses/springs by:
    1) solving the *generalized* eigenproblem K v = ω² M v (small-angle linear model)
    2) projecting simulated θ(t) onto the eigenvectors (modal coordinates)
    3) FFT on each modal coordinate and picking the dominant peak

    Returns
    -------
    (f1, f2, f3) in Hz, ordered low -> high (matching ω1 <= ω2 <= ω3).
    """
    omega0_sq = g / L
    M = np.diag([m1, m2, m3]).astype(float)
    K = np.array(
        [
            [m1 * omega0_sq + k1, -k1, 0.0],
            [-k1, m2 * omega0_sq + k1 + k2, -k2],
            [0.0, -k2, m3 * omega0_sq + k2],
        ],
        dtype=float,
    )
    theta = np.vstack([np.asarray(theta1), np.asarray(theta2), np.asarray(theta3)])
    f_est = estimate_normal_mode_frequencies(t, theta, M, K, use_hann_window)
    return f_est[0], f_est[1], f_est[2]


def demo_normal_modes_three():
    """
    Demonstrate the three normal modes of the coupled pendulum system.
    """
    print("=" * 60)
    print("THREE COUPLED PENDULUM - NORMAL MODE DEMONSTRATIONS")
    print("=" * 60)

    # Parameters
    m1, m2, m3 = 1.0, 1.0, 1.0
    k1, k2 = 10.0, 10.0
    L, g = 2.0, 9.81

    # Theoretical frequencies
    omega1, omega2, omega3, f1, f2, f3 = th_normal_modes_triple_pendulum(
        g, L, k1, k2, m1, m2, m3
    )
    T1 = 1 / f1
    T2 = 1 / f2
    T3 = 1 / f3

    print("\nSystem Parameters:")
    print(f"  m₁ = m₂ = m₃ = {m1} kg")
    print(f"  k₁ = k₂ = {k1} N/m")
    print(f"  L = {L} m")
    print(f"  g = {g} m/s²")
    print("\nTheoretical Normal Modes:")
    print(f"  Mode 1 (sloshing):      ω₁ = {omega1:.4f} rad/s, T₁ = {T1:.4f} s")
    print(f"  Mode 2 (anti-symmetric): ω₂ = {omega2:.4f} rad/s, T₂ = {T2:.4f} s")
    print(f"  Mode 3 (breathing):      ω₃ = {omega3:.4f} rad/s, T₃ = {T3:.4f} s")
    print(
        f"  Frequency ratios: ω₂/ω₁ = {omega2 / omega1:.4f}, ω₃/ω₁ = {omega3 / omega1:.4f}"
    )

    return omega1, omega2, omega3


def three_coupled_pendulum_animation_with_plots(
    theta_1_init=0.0,
    theta_2_init=10.0,
    theta_3_init=0.0,
    m1=1.0,
    m2=1.0,
    m3=1.0,
    k1=5.0,
    k2=5.0,
    L=2.0,
    g=9.81,
    simulation_time=20.0,
    fps=30,
    save_format="gif",
    save_anim=False,
    filename=None,
    *,
    return_data=False,
    precomputed_data=None,
    show=True,
    trace_length=50,
):
    """
    Simulate and animate a coupled pendulum system.

    Parameters
    ----------
    theta_1_init : float
        Initial angle of pendulum 1 in degrees
    theta_2_init : float
        Initial angle of pendulum 2 in degrees
    theta_3_init : float
        Initial angle of pendulum 3 in degrees
    m1, m2, m3 : float
        Masses of pendulum bobs
    k1, k2 : float
        Spring constants
    L : float
        Length of pendulum rods
    g : float
        Gravitational acceleration
    simulation_time : float
        Total simulation time in seconds
    fps : int
        Frames per second for animation
    save_format : str
        Format to save animation ('gif' or 'mp4')
    save_anim : bool
        Whether to save the animation
    filename : str or None
        Filename for saved animation (auto-generated if None)
    return_data : bool
        Return numeric simulation data instead of building an animation.
    precomputed_data : SimulationData or None
        Reuse numeric data when building an animation in the UI.
    show : bool
        Display the Matplotlib figure when not saving. Set to False in Streamlit.
    trace_length : int
        Number of recent positions retained in the motion trace.

    Returns
    -------
        anim - Animation object
    """

    from utils import SimulationData, validate_simulation_inputs

    validate_simulation_inputs(
        m1=m1,
        m2=m2,
        m3=m3,
        k1=k1,
        k2=k2,
        L=L,
        g=g,
        simulation_time=simulation_time,
        fps=fps,
    )
    if int(trace_length) < 1:
        raise ValueError("trace_length must be at least 1.")

    # =========================================================================
    # NUMERICAL SOLUTION
    # =========================================================================

    theta1_0 = np.radians(theta_1_init)
    theta2_0 = np.radians(theta_2_init)
    theta3_0 = np.radians(theta_3_init)

    # Initial state: [θ₁, ω₁, θ₂, ω₂, θ₃, ω₃] (starting from rest)
    y0 = [theta1_0, 0.0, theta2_0, 0.0, theta3_0, 0.0]

    t_span = (0, simulation_time)
    n_frames = max(2, int(simulation_time * fps) + 1)
    t_eval = np.linspace(0, simulation_time, n_frames)

    if precomputed_data is None:
        y = solve_ode(
            three_coupled_pendulum_derivatives,
            t_span,
            y0,
            t_eval,
            args=(g, L, k1, k2, m1, m2, m3),
        )
    else:
        t_eval = np.asarray(precomputed_data.time, dtype=float)
        y = np.asarray(precomputed_data.state, dtype=float)
        if y.shape != (6, t_eval.size):
            raise ValueError("precomputed_data has an incompatible state shape.")
    theta1 = y[0]
    omega1 = y[1]
    theta2 = y[2]
    omega2 = y[3]
    theta3 = y[4]
    omega3 = y[5]

    # =========================================================================
    # SPRING EXTENSION + ENERGIES
    # =========================================================================

    theta1_deg_all = np.degrees(theta1)
    theta2_deg_all = np.degrees(theta2)
    theta3_deg_all = np.degrees(theta3)

    spring_ext_all_1 = L * (np.sin(theta2) - np.sin(theta1))
    spring_ext_all_2 = L * (np.sin(theta3) - np.sin(theta2))

    KE_all = (
        0.5 * m1 * (L * omega1) ** 2
        + 0.5 * m2 * (L * omega2) ** 2
        + 0.5 * m3 * (L * omega3) ** 2
    )
    PE_grav_all = (
        -m1 * g * L * np.cos(theta1)
        - m2 * g * L * np.cos(theta2)
        - m3 * g * L * np.cos(theta3)
    )
    PE_spring_all = 0.5 * k1 * spring_ext_all_1**2 + 0.5 * k2 * spring_ext_all_2**2
    total_E_all = KE_all + PE_grav_all + PE_spring_all

    # =========================================================================
    # FREQUENCY ANALYSIS
    # =========================================================================

    # Theoretical normal modes
    omega1_theory, omega2_theory, omega3_theory, f1_theory, f2_theory, f3_theory = (
        th_normal_modes_triple_pendulum(g, L, k1, k2, m1, m2, m3)
    )

    # Numerical frequency estimation
    f1_num, f2_num, f3_num = est_freqs_triple_pendulum(
        t_eval, theta1, theta2, theta3, g, L, k1, k2, m1, m2, m3
    )

    data = SimulationData(
        time=np.array(t_eval, copy=True),
        state=np.array(y, copy=True),
        energy=np.array(total_E_all, copy=True),
        theoretical_frequencies=(float(f1_theory), float(f2_theory), float(f3_theory)),
        estimated_frequencies=(float(f1_num), float(f2_num), float(f3_num)),
        metadata={
            "system": "triple_pendulum",
            "parameters": {
                "theta_1_init": theta_1_init,
                "theta_2_init": theta_2_init,
                "theta_3_init": theta_3_init,
                "m1": m1,
                "m2": m2,
                "m3": m3,
                "k1": k1,
                "k2": k2,
                "L": L,
                "g": g,
                "simulation_time": simulation_time,
                "fps": fps,
            },
            "ke": np.array(KE_all, copy=True),
            "pe": np.array(PE_grav_all + PE_spring_all, copy=True),
            "v1": np.array(omega1, copy=True),
            "v2": np.array(omega2, copy=True),
            "v3": np.array(omega3, copy=True),
        },
    )
    if return_data:
        return data

    f1_str = format_frequency(f1_num, decimals=4)
    f2_str = format_frequency(f2_num, decimals=4)
    f3_str = format_frequency(f3_num, decimals=4)

    print("\nNormal Mode Frequencies:")
    print(
        f"  Theoretical: omega1 = {omega1_theory:.4f} rad/s (f1 = {f1_theory:.4f} Hz)"
    )
    print(f"              omega2 = {omega2_theory:.4f} rad/s (f2 = {f2_theory:.4f} Hz)")
    print(f"              omega3 = {omega3_theory:.4f} rad/s (f3 = {f3_theory:.4f} Hz)")
    print(f"  Numerical:   f1 ~= {f1_str}, f2 ~= {f2_str}, f3 ~= {f3_str}")

    # =========================================================================
    # ANIMATION SETUP
    # =========================================================================

    # Pivot positions
    pivot_separation = 1.5
    pivot1_x = -pivot_separation
    pivot2_x = 0
    pivot3_x = pivot_separation
    pivot_y = 0

    ceiling_length = 2 * pivot_separation + 0.5
    separation_ratio = pivot_separation / ceiling_length
    spring_hook_length = calculate_fixed_hook_length_three_pendulum(
        ceiling_length, L, separation_ratio, hook_ratio=0.15
    )
    spring_radius = 0.02 * ceiling_length
    spring_num_coils = 10
    ceiling_height = 0.05 * ceiling_length

    x1_all = pivot1_x + L * np.sin(theta1)
    y1_all = pivot_y - L * np.cos(theta1)
    x2_all = pivot2_x + L * np.sin(theta2)
    y2_all = pivot_y - L * np.cos(theta2)
    x3_all = pivot3_x + L * np.sin(theta3)
    y3_all = pivot_y - L * np.cos(theta3)

    # ── Dark-theme palette
    COLOR_MASS1 = PALETTE["MASS1"]
    COLOR_MASS2 = PALETTE["MASS3"]
    COLOR_MASS3 = PALETTE["MASS2"]
    COLOR_SPRING = PALETTE["SPRING"]
    COLOR_ROD = PALETTE["ROD"]
    COLOR_PIVOT = PALETTE["PIVOT"]
    COLOR_GUIDE = PALETTE["GUIDE"]
    COLOR_GRID = PALETTE["GRID"]
    COLOR_BG_MAIN = PALETTE["BG_MAIN"]
    COLOR_BG_TH1 = PALETTE["BG_TH1"]
    COLOR_BG_TH2 = PALETTE["BG_TH2"]
    COLOR_BG_TH3 = PALETTE["BG_TH3"]

    # Figure setup with gridspec
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 8))
    fig.set_facecolor("black")
    gs = GridSpec(
        3,
        2,
        figure=fig,
        width_ratios=[1.35, 1],
        left=0.04,
        right=0.98,
        top=0.9,
        bottom=0.06,
        hspace=0.48,
        wspace=0.24,
    )

    # Main pendulum animation (left column)
    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor(COLOR_BG_MAIN)

    # Time series plots (right column)
    ax_theta1 = fig.add_subplot(gs[0, 1])
    ax_theta2 = fig.add_subplot(gs[1, 1])
    ax_theta3 = fig.add_subplot(gs[2, 1])
    ax_theta1.set_facecolor(COLOR_BG_TH1)
    ax_theta2.set_facecolor(COLOR_BG_TH2)
    ax_theta3.set_facecolor(COLOR_BG_TH3)

    fig.suptitle(
        "Three Spring-Coupled Pendulum System",
        fontsize=14,
        fontweight="bold",
        color="white",
    )

    xlim, ylim = calculate_plot_limits_3(
        ceiling_length,
        np.min(x1_all),
        np.max(x3_all),
        np.min([y1_all.min(), y2_all.min(), y3_all.min()]),
        np.min([y1_all.min(), y2_all.min(), y3_all.min()]),
        padding=0.2,
        middle_mass_x=np.mean(x2_all),
        middle_mass_y=np.mean(y2_all),
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.15, color=COLOR_GRID)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for sp in ax.spines.values():
        sp.set_edgecolor(PALETTE["GUIDE"])

    # Motion type
    if (
        abs(theta_1_init - theta_2_init) < 1e-6
        and abs(theta_2_init - theta_3_init) < 1e-6
    ):
        motion_type = "Sloshing Mode (Normal Mode 1 - All in phase)"
    elif abs(theta_1_init + theta_3_init) < 1e-6 and abs(theta_2_init) < 1e-6:
        motion_type = "Anti-Symmetric Mode (Normal Mode 2 - Middle at rest)"
    elif (
        abs(theta_1_init - theta_3_init) < 1e-6
        and abs(theta_2_init + 2 * theta_1_init) < 1e-6
    ):
        motion_type = "Breathing Mode (Normal Mode 3 - Maximum spring stretch)"
    else:
        motion_type = "Mixed-Mode Oscillation (Superposition)"

    title_text = motion_type

    ax.set_title(title_text, fontsize=12, fontweight="bold", color="white")

    draw_ceiling(ax, ceiling_length, ceiling_height)

    ax.plot(pivot1_x, pivot_y, marker="o", color=COLOR_PIVOT, markersize=15, zorder=6)
    ax.plot(pivot2_x, pivot_y, marker="o", color=COLOR_PIVOT, markersize=15, zorder=6)
    ax.plot(pivot3_x, pivot_y, marker="o", color=COLOR_PIVOT, markersize=15, zorder=6)

    ax.axvline(
        pivot1_x, ymin=0, ymax=1, color=COLOR_GUIDE, ls="--", lw=1, zorder=0, alpha=0.6
    )
    ax.axvline(
        pivot2_x, ymin=0, ymax=1, color=COLOR_GUIDE, ls="--", lw=1, zorder=0, alpha=0.6
    )
    ax.axvline(
        pivot3_x, ymin=0, ymax=1, color=COLOR_GUIDE, ls="--", lw=1, zorder=0, alpha=0.6
    )

    (rod1_line,) = ax.plot([], [], color=COLOR_ROD, lw=2.5, zorder=3)
    (rod2_line,) = ax.plot([], [], color=COLOR_ROD, lw=2.5, zorder=3)
    (rod3_line,) = ax.plot([], [], color=COLOR_ROD, lw=2.5, zorder=3)

    (spring1_line,) = ax.plot([], [], color=COLOR_SPRING, lw=2.0, zorder=2, alpha=0.95)
    (spring2_line,) = ax.plot([], [], color=COLOR_SPRING, lw=2.0, zorder=2, alpha=0.95)

    mass_size_min = 20
    mass_size_ref = 25
    avg_mass = (m1 + m2 + m3) / 3.0 if (m1 + m2 + m3) > 0 else 1.0
    mass_1_size = max(mass_size_min, mass_size_ref * np.cbrt(m1 / avg_mass))
    mass_2_size = max(mass_size_min, mass_size_ref * np.cbrt(m2 / avg_mass))
    mass_3_size = max(mass_size_min, mass_size_ref * np.cbrt(m3 / avg_mass))

    (mass1_plot,) = ax.plot(
        [],
        [],
        marker="o",
        color=COLOR_MASS1,
        markersize=mass_1_size,
        zorder=4,
        label=f"$m_1$={m1} kg",
        markeredgecolor=PALETTE["EDGE_MASS1"],
        markeredgewidth=1.5,
    )
    (mass2_plot,) = ax.plot(
        [],
        [],
        marker="o",
        color=COLOR_MASS2,
        markersize=mass_2_size,
        zorder=4,
        label=f"$m_2$={m2} kg",
        markeredgecolor=PALETTE["EDGE_MASS3"],
        markeredgewidth=1.5,
    )
    (mass3_plot,) = ax.plot(
        [],
        [],
        marker="o",
        color=COLOR_MASS3,
        markersize=mass_3_size,
        zorder=4,
        label=f"$m_3$={m3} kg",
        markeredgecolor=PALETTE["EDGE_MASS2"],
        markeredgewidth=1.5,
    )

    trace1_x, trace1_y = [], []
    trace2_x, trace2_y = [], []
    trace3_x, trace3_y = [], []
    (trace1_line,) = ax.plot([], [], color=COLOR_MASS1, lw=1.2, alpha=0.55, zorder=1)
    (trace2_line,) = ax.plot([], [], color=COLOR_MASS2, lw=1.2, alpha=0.55, zorder=1)
    (trace3_line,) = ax.plot([], [], color=COLOR_MASS3, lw=1.2, alpha=0.55, zorder=1)

    (trace1_marker,) = ax.plot(
        [], [], "o", color=COLOR_MASS1, markersize=8, zorder=5, alpha=0.9
    )
    (trace2_marker,) = ax.plot(
        [], [], "o", color=COLOR_MASS2, markersize=8, zorder=5, alpha=0.9
    )
    (trace3_marker,) = ax.plot(
        [], [], "o", color=COLOR_MASS3, markersize=8, zorder=5, alpha=0.9
    )

    # Setup time series plots
    # Theta1 vs time
    ax_theta1.set_xlim(0, 1.1 * simulation_time)
    ax_theta1.set_ylim(np.degrees(theta1).min() - 5, np.degrees(theta1).max() + 5)
    style_dark_subplot(ax_theta1, "Time (s)", "$\\theta_1$ (deg)", "$\\theta_1$ vs $t$")
    (theta1_time_line,) = ax_theta1.plot(
        [], [], color=COLOR_MASS1, lw=2, label="$\\theta_1$"
    )
    (theta1_current_point,) = ax_theta1.plot(
        [], [], "o", color=COLOR_MASS1, markersize=8, zorder=5
    )
    ax_theta1.legend(
        loc="upper right",
        fontsize=9,
        facecolor=PALETTE["BG_MAIN"],
        edgecolor=PALETTE["GUIDE"],
        labelcolor="white",
    )

    # Theta2 vs time
    ax_theta2.set_xlim(0, 1.1 * simulation_time)
    ax_theta2.set_ylim(np.degrees(theta2).min() - 5, np.degrees(theta2).max() + 5)
    style_dark_subplot(ax_theta2, "Time (s)", "$\\theta_2$ (deg)", "$\\theta_2$ vs $t$")
    (theta2_time_line,) = ax_theta2.plot(
        [], [], color=COLOR_MASS2, lw=2, label="$\\theta_2$"
    )
    (theta2_current_point,) = ax_theta2.plot(
        [], [], "o", color=COLOR_MASS2, markersize=8, zorder=5
    )
    ax_theta2.legend(
        loc="upper right",
        fontsize=9,
        facecolor=PALETTE["BG_MAIN"],
        edgecolor=PALETTE["GUIDE"],
        labelcolor="white",
    )

    # Theta3 vs time
    ax_theta3.set_xlim(0, 1.1 * simulation_time)
    ax_theta3.set_ylim(np.degrees(theta3).min() - 5, np.degrees(theta3).max() + 5)
    style_dark_subplot(ax_theta3, "Time (s)", "$\\theta_3$ (deg)", "$\\theta_3$ vs $t$")
    (theta3_time_line,) = ax_theta3.plot(
        [], [], color=COLOR_MASS3, lw=2, label="$\\theta_3$"
    )
    (theta3_current_point,) = ax_theta3.plot(
        [], [], "o", color=COLOR_MASS3, markersize=8, zorder=5
    )
    ax_theta3.legend(
        loc="upper right",
        fontsize=9,
        facecolor=PALETTE["BG_MAIN"],
        edgecolor=PALETTE["GUIDE"],
        labelcolor="white",
    )

    # Time series data
    time_history = []
    theta1_history = []
    theta2_history = []
    theta3_history = []

    for annotation in ax.texts:
        annotation.set_visible(False)

    # =========================================================================
    # ANIMATION FUNCTIONS
    # =========================================================================

    def init():
        """Initialize animation."""
        rod1_line.set_data([], [])
        rod2_line.set_data([], [])
        rod3_line.set_data([], [])
        spring1_line.set_data([], [])
        spring2_line.set_data([], [])
        mass1_plot.set_data([], [])
        mass2_plot.set_data([], [])
        mass3_plot.set_data([], [])
        trace1_line.set_data([], [])
        trace2_line.set_data([], [])
        trace3_line.set_data([], [])
        trace1_marker.set_data([], [])
        trace2_marker.set_data([], [])
        trace3_marker.set_data([], [])
        theta1_time_line.set_data([], [])
        theta2_time_line.set_data([], [])
        theta3_time_line.set_data([], [])
        theta1_current_point.set_data([], [])
        theta2_current_point.set_data([], [])
        theta3_current_point.set_data([], [])
        return (
            rod1_line,
            rod2_line,
            rod3_line,
            spring1_line,
            spring2_line,
            mass1_plot,
            mass2_plot,
            mass3_plot,
            trace1_line,
            trace2_line,
            trace3_line,
            trace1_marker,
            trace2_marker,
            trace3_marker,
            theta1_time_line,
            theta2_time_line,
            theta3_time_line,
            theta1_current_point,
            theta2_current_point,
            theta3_current_point,
        )

    def animate(frame):
        """Update animation for each frame."""
        current_time = t_eval[frame]
        theta1_deg = theta1_deg_all[frame]
        theta2_deg = theta2_deg_all[frame]
        theta3_deg = theta3_deg_all[frame]

        x1, y1 = x1_all[frame], y1_all[frame]
        x2, y2 = x2_all[frame], y2_all[frame]
        x3, y3 = x3_all[frame], y3_all[frame]

        rod1_line.set_data([pivot1_x, x1], [pivot_y, y1])
        rod2_line.set_data([pivot2_x, x2], [pivot_y, y2])
        rod3_line.set_data([pivot3_x, x3], [pivot_y, y3])

        spring1_x, spring1_y = draw_spring_with_hook(
            start_pos=(x1, y1),
            end_pos=(x2, y2),
            num_coils=spring_num_coils,
            radius=spring_radius,
            hook_length=spring_hook_length,
        )
        spring1_line.set_data(spring1_x, spring1_y)

        spring2_x, spring2_y = draw_spring_with_hook(
            start_pos=(x2, y2),
            end_pos=(x3, y3),
            num_coils=spring_num_coils,
            radius=spring_radius,
            hook_length=spring_hook_length,
        )
        spring2_line.set_data(spring2_x, spring2_y)

        mass1_plot.set_data([x1], [y1])
        mass2_plot.set_data([x2], [y2])
        mass3_plot.set_data([x3], [y3])

        trace1_x.append(x1)
        trace1_y.append(y1)
        trace2_x.append(x2)
        trace2_y.append(y2)
        trace3_x.append(x3)
        trace3_y.append(y3)

        max_trace = int(trace_length)
        if len(trace1_x) > max_trace:
            trace1_x.pop(0)
            trace1_y.pop(0)
            trace2_x.pop(0)
            trace2_y.pop(0)
            trace3_x.pop(0)
            trace3_y.pop(0)

        trace1_line.set_data(trace1_x, trace1_y)
        trace2_line.set_data(trace2_x, trace2_y)
        trace3_line.set_data(trace3_x, trace3_y)

        if len(trace1_x) > 0:
            trace1_marker.set_data([trace1_x[-1]], [trace1_y[-1]])
            trace2_marker.set_data([trace2_x[-1]], [trace2_y[-1]])
            trace3_marker.set_data([trace3_x[-1]], [trace3_y[-1]])

        time_history.append(current_time)
        theta1_history.append(theta1_deg)
        theta2_history.append(theta2_deg)
        theta3_history.append(theta3_deg)

        theta1_time_line.set_data(time_history, theta1_history)
        theta2_time_line.set_data(time_history, theta2_history)
        theta3_time_line.set_data(time_history, theta3_history)
        theta1_current_point.set_data([current_time], [theta1_deg])
        theta2_current_point.set_data([current_time], [theta2_deg])
        theta3_current_point.set_data([current_time], [theta3_deg])

        return (
            rod1_line,
            rod2_line,
            rod3_line,
            spring1_line,
            spring2_line,
            mass1_plot,
            mass2_plot,
            mass3_plot,
            trace1_line,
            trace2_line,
            trace3_line,
            trace1_marker,
            trace2_marker,
            trace3_marker,
            theta1_time_line,
            theta2_time_line,
            theta3_time_line,
            theta1_current_point,
            theta2_current_point,
            theta3_current_point,
        )

    print("\nCreating animation...")
    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(t_eval),
        interval=1000 / fps,
        blit=True,
        repeat=False,
    )

    if save_anim:
        if filename is None:
            filename = (
                f"triple_pendulum_m1_{m1}_m2_{m2}_m3_{m3}_k1_{k1}_k2_{k2}_L_{L}_"
                f"theta1_{theta_1_init}_theta2_{theta_2_init}_theta3_{theta_3_init}"
                f".{save_format}"
            )
        save_animation(anim, SAVE_DIR / filename, fps, save_format)
        plt.close(fig)
    elif show:
        plt.show()

    return anim


def main():
    """Main function to run simulations and demonstrations."""
    header = "STARTING SPRING-COUPLED TRIPLE PENDULUM SIMULATION"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    default_params = {
        "theta_1_init": 0.0,
        "theta_2_init": 0.0,
        "theta_3_init": 10.0,
        "m1": 1.0,
        "m2": 1.0,
        "m3": 1.0,
        "k1": 0.5,
        "k2": 0.5,
        "L": 2.0,
        "g": 9.81,
        "simulation_time": 10.0,
        "fps": 30,
        "save_format": "gif",
    }

    use_default = input("Use default simulation parameters? (y/n): ").strip().lower()
    if use_default == "y":
        params = default_params.copy()
    else:
        params = {}
        try:
            params["theta_1_init"] = float(
                input(
                    f"Enter initial angle for pendulum 1 (degrees)[{default_params['theta_1_init']}] degrees): "
                )
                or default_params["theta_1_init"]
            )
            params["theta_2_init"] = float(
                input(
                    f"Enter initial angle for pendulum 2 (degrees)[{default_params['theta_2_init']}] degrees): "
                )
                or default_params["theta_2_init"]
            )
            params["theta_3_init"] = float(
                input(
                    f"Enter initial angle for pendulum 3 (degrees)[{default_params['theta_3_init']}] degrees): "
                )
                or default_params["theta_3_init"]
            )
            params["m1"] = float(
                input(
                    f"Enter mass of the first pendulum (kg)[{default_params['m1']}] kg): "
                )
                or default_params["m1"]
            )
            params["m2"] = float(
                input(
                    f"Enter mass of the second pendulum (kg)[{default_params['m2']}] kg): "
                )
                or default_params["m2"]
            )
            params["m3"] = float(
                input(
                    f"Enter mass of the third pendulum (kg)[{default_params['m3']}] kg): "
                )
                or default_params["m3"]
            )
            params["k1"] = float(
                input(f"Enter spring constant k1 (N/m)[{default_params['k1']}] N/m): ")
                or default_params["k1"]
            )
            params["k2"] = float(
                input(f"Enter spring constant k2 (N/m)[{default_params['k2']}] N/m): ")
                or default_params["k2"]
            )
            params["L"] = float(
                input(
                    f"Enter length of the pendulum rods (m)[{default_params['L']}] m): "
                )
                or default_params["L"]
            )
            params["g"] = float(
                input(
                    f"Enter gravitational acceleration (m/s²)[{default_params['g']}] m/s²): "
                )
                or default_params["g"]
            )
            params["simulation_time"] = float(
                input(
                    f"Enter simulation time (seconds)[{default_params['simulation_time']}] seconds): "
                )
                or default_params["simulation_time"]
            )
            params["fps"] = int(
                input(
                    f"Enter frames per second for animation [{default_params['fps']}]: "
                )
                or default_params["fps"]
            )
            params["save_format"] = (
                input(
                    f"Enter animation save format ('gif' or 'mp4')[{default_params['save_format']}]: "
                )
                .strip()
                .lower()
                or default_params["save_format"]
            )
            if params["save_format"] not in ["gif", "mp4"]:
                print("Invalid format. Using default 'gif'.")
                params["save_format"] = "gif"
        except ValueError:
            print("Invalid input. Using default parameters.")
            params = default_params.copy()

    save_anim = input("Save animation to file? (y/n): ").strip().lower() == "y"
    filename = None
    if save_anim:
        base_input = input(
            "Enter base filename (without extension) or press Enter for default: "
        ).strip()
        if base_input:
            # Strip any extension the user may have accidentally typed
            from pathlib import Path

            base_input = Path(base_input).stem
            filename = f"{base_input}.{params['save_format']}"

    print("\nRunning simulation with parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    demo_normal_modes_three()

    animation = three_coupled_pendulum_animation_with_plots(
        theta_1_init=params["theta_1_init"],
        theta_2_init=params["theta_2_init"],
        theta_3_init=params["theta_3_init"],
        m1=params["m1"],
        m2=params["m2"],
        m3=params["m3"],
        k1=params["k1"],
        k2=params["k2"],
        L=params["L"],
        g=params["g"],
        simulation_time=params["simulation_time"],
        fps=params["fps"],
        save_format=params["save_format"],
        save_anim=save_anim,
        filename=filename,
    )

    return animation


if __name__ == "__main__":
    animation = main()
    print("\nSimulation completed.")
