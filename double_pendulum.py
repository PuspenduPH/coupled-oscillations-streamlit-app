import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

from utils import (
    PALETTE,
    SAVE_DIR,
    calculate_fixed_hook_length,
    calculate_plot_limits,
    draw_ceiling,
    draw_spring_with_hook,
    estimate_normal_mode_frequencies,
    format_frequency,
    save_animation,
    solve_ode,
    style_dark_subplot,
)


def coupled_pendulum_derivatives(t, y, g, L, k, m1, m2):
    """
    Compute the derivatives for the coupled pendulum system.

    Uses the exact non-linear equations of motion:
    θ̈₁ = -(g/L)sin(θ₁) + (k/m₁)(sin(θ₂) - sin(θ₁))cos(θ₁)
    θ̈₂ = -(g/L)sin(θ₂) - (k/m₂)(sin(θ₂) - sin(θ₁))cos(θ₂)

    Parameters
    ----------
    t : float
        Time (not used explicitly, required by solve_ivp)
    y : array-like
        State vector [θ₁, ω₁, θ₂, ω₂] where ω = dθ/dt
    g : float
        Gravitational acceleration
    L : float
        Length of pendulum rods
    k : float
        Spring constant
    m1, m2 : float
        Masses of the two pendulum bobs

    Returns
    -------
    list
        Derivatives [dθ₁/dt, dω₁/dt, dθ₂/dt, dω₂/dt]
    """
    theta1, omega1, theta2, omega2 = y

    # Spring extension term
    delta_sin = np.sin(theta2) - np.sin(theta1)

    # Angular accelerations (exact non-linear equations)
    d_theta1 = omega1
    d_omega1 = -(g / L) * np.sin(theta1) + (k / m1) * delta_sin * np.cos(theta1)

    d_theta2 = omega2
    d_omega2 = -(g / L) * np.sin(theta2) - (k / m2) * delta_sin * np.cos(theta2)

    return [d_theta1, d_omega1, d_theta2, d_omega2]


def th_normal_modes_double_pendulum(g, L, k, m1, m2):
    """
    Calculate the theoretical normal mode frequencies for the linearized system.

    For equal masses (m1 = m2 = m):
        ω₁² = g/L           (in-phase mode)
        ω₂² = g/L + 2k/m    (out-of-phase mode)

    For unequal masses, we solve the eigenvalue problem of the linearized system.

    Parameters
    ----------
    g, L, k, m1, m2 : float
        System parameters

    Returns
    -------
    tuple
        (omega1, omega2, f1, f2) - angular frequencies and frequencies in Hz
    """

    a11 = g / L + k / m1
    a12 = -k / m1
    a21 = -k / m2
    a22 = g / L + k / m2

    A = np.array([[a11, a12], [a21, a22]])
    eigenvalues = np.linalg.eigvals(A)
    omega_sq = np.sort(np.real(eigenvalues))

    omega1 = np.sqrt(omega_sq[0])  # Lower frequency (in-phase)
    omega2 = np.sqrt(omega_sq[1])  # Higher frequency (out-of-phase)

    # Convert to Hz
    f1 = omega1 / (2 * np.pi)
    f2 = omega2 / (2 * np.pi)

    return omega1, omega2, f1, f2


def est_freqs_double_pendulum(t, theta1, theta2, g, L, k, m1, m2, use_hann_window=True):
    """
    Estimate the 2 normal-mode frequencies for unequal masses/springs by:
    1) solving the *generalized* eigenproblem K v = ω² M v (small-angle linear model)
    2) projecting simulated θ(t) onto the eigenvectors (modal coordinates)
    3) FFT on each modal coordinate and picking the dominant peak

    Parameters
    ----------
    t : array
        Time array
    theta1, theta2 : array
        Angle arrays from simulation
    g, L, k, m1, m2 : float
        System parameters
    use_hann_window : bool
        Whether to apply a Hann window before FFT

    Returns
    -------
    tuple
        (f1_est, f2_est) - Estimated frequencies in Hz
    """
    omega0_sq = g / L
    M = np.diag([m1, m2]).astype(float)
    K = np.array(
        [
            [m1 * omega0_sq + k, -k],
            [-k, m2 * omega0_sq + k],
        ],
        dtype=float,
    )
    theta = np.vstack([np.asarray(theta1), np.asarray(theta2)])
    f_est = estimate_normal_mode_frequencies(t, theta, M, K, use_hann_window)
    return f_est[0], f_est[1]


def demo_normal_modes():
    """
    Demonstrate the two normal modes of the coupled pendulum.
    """
    print("=" * 60)
    print("NORMAL MODE DEMONSTRATIONS")
    print("=" * 60)

    # Parameters
    m1, m2, k, L, g = 1.0, 1.0, 10.0, 2.0, 9.81

    # Theoretical frequencies
    omega1, omega2, f1, f2 = th_normal_modes_double_pendulum(g, L, k, m1, m2)
    T1 = 1 / f1
    T2 = 1 / f2

    print("\nSystem Parameters:")
    print(f"  m₁ = m₂ = {m1} kg")
    print(f"  k = {k} N/m")
    print(f"  L = {L} m")
    print(f"  g = {g} m/s²")
    print("\nTheoretical Normal Modes:")
    print(f"  Mode 1 (in-phase):     ω₁ = {omega1:.4f} rad/s, T₁ = {T1:.4f} s")
    print(f"  Mode 2 (out-of-phase): ω₂ = {omega2:.4f} rad/s, T₂ = {T2:.4f} s")
    print(f"  Frequency ratio: ω₂/ω₁ = {omega2 / omega1:.4f}")

    return omega1, omega2


def double_pendulum_animation_with_plots(
    theta_1_init=0.0,
    theta_2_init=10.0,
    m1=1.0,
    m2=1.0,
    k=5.0,
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
    m1, m2 : float
        Masses of pendulum bobs
    k : float
        Spring constant
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
    tuple
        (fig, anim) - Figure and animation objects
    """

    from utils import SimulationData, validate_simulation_inputs

    validate_simulation_inputs(
        m1=m1,
        m2=m2,
        k=k,
        L=L,
        g=g,
        simulation_time=simulation_time,
        fps=fps,
    )
    if int(trace_length) < 1:
        raise ValueError("trace_length must be at least 1.")

    theta1_0 = np.radians(theta_1_init)
    theta2_0 = np.radians(theta_2_init)

    y0 = [theta1_0, 0.0, theta2_0, 0.0]

    t_span = (0, simulation_time)
    n_frames = max(2, int(simulation_time * fps) + 1)
    t = np.linspace(0, simulation_time, n_frames)

    if precomputed_data is None:
        y = solve_ode(
            coupled_pendulum_derivatives, t_span, y0, t, args=(g, L, k, m1, m2)
        )
    else:
        t = np.asarray(precomputed_data.time, dtype=float)
        y = np.asarray(precomputed_data.state, dtype=float)
        if y.shape != (4, t.size):
            raise ValueError("precomputed_data has an incompatible state shape.")
    theta1 = y[0]
    omega1 = y[1]
    theta2 = y[2]
    omega2 = y[3]

    theta1_deg_all = np.degrees(theta1)
    theta2_deg_all = np.degrees(theta2)

    # Spring extensions and energies
    spring_ext_all = L * (np.sin(theta2) - np.sin(theta1))
    KE_all = 0.5 * m1 * (L * omega1) ** 2 + 0.5 * m2 * (L * omega2) ** 2
    PE_grav_all = -m1 * g * L * np.cos(theta1) - m2 * g * L * np.cos(theta2)
    PE_spring_all = 0.5 * k * spring_ext_all**2
    total_E_all = KE_all + PE_grav_all + PE_spring_all

    # Theoretical normal modes
    omega1_theory, omega2_theory, f1_theory, f2_theory = (
        th_normal_modes_double_pendulum(g, L, k, m1, m2)
    )
    # Numerical frequency estimation
    f1_num, f2_num = est_freqs_double_pendulum(t, theta1, theta2, g, L, k, m1, m2)

    data = SimulationData(
        time=np.array(t, copy=True),
        state=np.array(y, copy=True),
        energy=np.array(total_E_all, copy=True),
        theoretical_frequencies=(float(f1_theory), float(f2_theory)),
        estimated_frequencies=(float(f1_num), float(f2_num)),
        metadata={
            "system": "double_pendulum",
            "parameters": {
                "theta_1_init": theta_1_init,
                "theta_2_init": theta_2_init,
                "m1": m1,
                "m2": m2,
                "k": k,
                "L": L,
                "g": g,
                "simulation_time": simulation_time,
                "fps": fps,
            },
            "ke": np.array(KE_all, copy=True),
            "pe": np.array(PE_grav_all + PE_spring_all, copy=True),
            "v1": np.array(omega1, copy=True),
            "v2": np.array(omega2, copy=True),
        },
    )
    if return_data:
        return data

    f1_str = format_frequency(f1_num)
    f2_str = format_frequency(f2_num)

    print("\nNormal Mode Frequencies:")
    print(
        f"  Theoretical: omega1 = {omega1_theory:.3f} rad/s (f1 = {f1_theory:.3f} Hz)"
    )
    print(f"              omega2 = {omega2_theory:.3f} rad/s (f2 = {f2_theory:.3f} Hz)")
    print(f"  Numerical:   f1 ~= {f1_str}, f2 ~= {f2_str}")

    # Pivot positions
    pivot_separation = 1.5
    pivot1_x = -pivot_separation / 2
    pivot2_x = pivot_separation / 2
    pivot_y = 0

    ceiling_length = pivot_separation + 1.0
    separation_ratio = pivot_separation / ceiling_length
    spring_hook_length = calculate_fixed_hook_length(
        ceiling_length, L, separation_ratio, hook_ratio=0.1
    )
    spring_radius = 0.03 * ceiling_length
    spring_num_coils = 10
    ceiling_height = 0.05 * ceiling_length

    # Mass positions
    x1_all = pivot1_x + L * np.sin(theta1)
    y1_all = pivot_y - L * np.cos(theta1)
    x2_all = pivot2_x + L * np.sin(theta2)
    y2_all = pivot_y - L * np.cos(theta2)

    # ── Dark-theme palette ────────────────────────────────────────────────
    COLOR_MASS1 = PALETTE["MASS1"]
    COLOR_MASS2 = PALETTE["MASS2"]
    COLOR_SPRING = PALETTE["SPRING"]
    COLOR_ROD = PALETTE["ROD"]
    COLOR_PIVOT = PALETTE["PIVOT"]
    COLOR_GUIDE = PALETTE["GUIDE"]
    COLOR_GRID = PALETTE["GRID"]
    COLOR_BG_MAIN = PALETTE["BG_MAIN"]
    COLOR_BG_TH1 = PALETTE["BG_TH3"]
    COLOR_BG_TH2 = PALETTE["BG_TH2"]

    # Figure setup with gridspec
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 7))
    fig.set_facecolor("black")
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.35, 1],
        left=0.02,
        right=0.98,
        top=0.86,
        bottom=0.08,
        hspace=0.36,
        wspace=0.24,
    )

    # Main pendulum animation (left column)
    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor(COLOR_BG_MAIN)

    # Time series plots (right column)
    ax_theta1 = fig.add_subplot(gs[0, 1])
    ax_theta2 = fig.add_subplot(gs[1, 1])
    ax_theta1.set_facecolor(COLOR_BG_TH1)
    ax_theta2.set_facecolor(COLOR_BG_TH2)

    fig.suptitle(
        "Spring-Coupled Double Pendulum Motion",
        fontsize=14,
        fontweight="bold",
        color="white",
    )

    xlim, ylim = calculate_plot_limits(
        ceiling_length,
        np.min(x1_all),
        np.max(x2_all),
        np.min(y1_all),
        np.min(y2_all),
        padding=0.2,
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

    # Motion type based on initial conditions
    if abs(theta_1_init - theta_2_init) < 1e-6:
        motion_type = "In-Phase Oscillation (Normal Mode 1)"
    elif abs(theta_1_init + theta_2_init) < 1e-6:
        motion_type = "Out-of-Phase Oscillation (Normal Mode 2)"
    elif theta_1_init == 0 or theta_2_init == 0:
        motion_type = "Energy Transfer (Beats Phenomenon)"
    else:
        motion_type = "Mixed-Mode Oscillation (Superposition)"

    title_text = f"Coupled Pendulum System\n{motion_type}"

    ax.set_title(title_text, fontsize=12, fontweight="bold", pad=10, color="white")

    draw_ceiling(ax, ceiling_length, ceiling_height)
    ax.plot(pivot1_x, pivot_y, "o", color=COLOR_PIVOT, markersize=15, zorder=6)
    ax.plot(pivot2_x, pivot_y, "o", color=COLOR_PIVOT, markersize=15, zorder=6)

    ax.axvline(
        pivot1_x, ymin=0, ymax=1, color=COLOR_GUIDE, ls="--", lw=1, zorder=0, alpha=0.6
    )
    ax.axvline(
        pivot2_x, ymin=0, ymax=1, color=COLOR_GUIDE, ls="--", lw=1, zorder=0, alpha=0.6
    )

    (rod1_line,) = ax.plot([], [], color=COLOR_ROD, lw=2.5, zorder=3)
    (rod2_line,) = ax.plot([], [], color=COLOR_ROD, lw=2.5, zorder=3)
    (spring_line,) = ax.plot([], [], color=COLOR_SPRING, lw=2.0, zorder=2, alpha=0.95)

    mass_radius_min = 20
    mass_radius_ref = 25
    avg_mass = (m1 + m2) / 2.0 if (m1 + m2) > 0 else 1.0
    mass_1_size = max(mass_radius_min, mass_radius_ref * np.cbrt(m1 / avg_mass))
    mass_2_size = max(mass_radius_min, mass_radius_ref * np.cbrt(m2 / avg_mass))

    (mass1_plot,) = ax.plot(
        [],
        [],
        "o",
        color=COLOR_MASS1,
        markersize=mass_1_size,
        zorder=4,
        label=f"Mass 1 ($m_1$={m1} kg)",
        markeredgecolor=PALETTE["EDGE_MASS1"],
        markeredgewidth=1.5,
    )
    (mass2_plot,) = ax.plot(
        [],
        [],
        "o",
        color=COLOR_MASS2,
        markersize=mass_2_size,
        zorder=4,
        label=f"Mass 2 ($m_2$={m2} kg)",
        markeredgecolor=PALETTE["EDGE_MASS2"],
        markeredgewidth=1.5,
    )

    ax.legend(
        loc="lower right",
        fontsize=8,
        markerscale=0.5,
        ncol=2,
        fancybox=True,
        facecolor=PALETTE["BG_MAIN"],
        edgecolor=PALETTE["GUIDE"],
        labelcolor="white",
        labelspacing=1.0,
    )
    trace1_x, trace1_y = [], []
    trace2_x, trace2_y = [], []
    (trace1_line,) = ax.plot([], [], color=COLOR_MASS1, lw=1.2, alpha=0.55, zorder=1)
    (trace2_line,) = ax.plot([], [], color=COLOR_MASS2, lw=1.2, alpha=0.55, zorder=1)

    (trace1_marker,) = ax.plot(
        [], [], "o", color=COLOR_MASS1, markersize=8, zorder=5, alpha=0.9
    )
    (trace2_marker,) = ax.plot(
        [], [], "o", color=COLOR_MASS2, markersize=8, zorder=5, alpha=0.9
    )

    # Time series plots
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

    # Time series data
    time_history = []
    theta1_history = []
    theta2_history = []

    def init():
        """Initialize animation."""
        rod1_line.set_data([], [])
        rod2_line.set_data([], [])
        spring_line.set_data([], [])
        mass1_plot.set_data([], [])
        mass2_plot.set_data([], [])
        trace1_line.set_data([], [])
        trace2_line.set_data([], [])
        trace1_marker.set_data([], [])
        trace2_marker.set_data([], [])
        theta1_time_line.set_data([], [])
        theta2_time_line.set_data([], [])
        theta1_current_point.set_data([], [])
        theta2_current_point.set_data([], [])
        return (
            rod1_line,
            rod2_line,
            spring_line,
            mass1_plot,
            mass2_plot,
            trace1_line,
            trace2_line,
            trace1_marker,
            trace2_marker,
            theta1_time_line,
            theta2_time_line,
            theta1_current_point,
            theta2_current_point,
        )

    def animate(frame):
        """Update animation for each frame."""
        current_time = t[frame]
        theta1_deg = theta1_deg_all[frame]
        theta2_deg = theta2_deg_all[frame]

        x1, y1 = x1_all[frame], y1_all[frame]
        x2, y2 = x2_all[frame], y2_all[frame]

        rod1_line.set_data([pivot1_x, x1], [pivot_y, y1])
        rod2_line.set_data([pivot2_x, x2], [pivot_y, y2])

        spring_x, spring_y = draw_spring_with_hook(
            start_pos=(x1, y1),
            end_pos=(x2, y2),
            num_coils=spring_num_coils,
            radius=spring_radius,
            hook_length=spring_hook_length,
        )
        spring_line.set_data(spring_x, spring_y)

        mass1_plot.set_data([x1], [y1])
        mass2_plot.set_data([x2], [y2])

        trace1_x.append(x1)
        trace1_y.append(y1)
        trace2_x.append(x2)
        trace2_y.append(y2)

        max_trace = int(trace_length)
        if len(trace1_x) > max_trace:
            trace1_x.pop(0)
            trace1_y.pop(0)
            trace2_x.pop(0)
            trace2_y.pop(0)

        trace1_line.set_data(trace1_x, trace1_y)
        trace2_line.set_data(trace2_x, trace2_y)

        if len(trace1_x) > 0:
            trace1_marker.set_data([trace1_x[-1]], [trace1_y[-1]])
            trace2_marker.set_data([trace2_x[-1]], [trace2_y[-1]])

        time_history.append(current_time)
        theta1_history.append(theta1_deg)
        theta2_history.append(theta2_deg)

        theta1_time_line.set_data(time_history, theta1_history)
        theta2_time_line.set_data(time_history, theta2_history)

        theta1_current_point.set_data([current_time], [theta1_deg])
        theta2_current_point.set_data([current_time], [theta2_deg])

        return (
            rod1_line,
            rod2_line,
            spring_line,
            mass1_plot,
            mass2_plot,
            trace1_line,
            trace2_line,
            trace1_marker,
            trace2_marker,
            theta1_time_line,
            theta2_time_line,
            theta1_current_point,
            theta2_current_point,
        )

    # Animation
    print("\nCreating animation...")
    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(t),
        interval=1000 / fps,
        blit=True,
        repeat=False,
    )

    if save_anim:
        if filename is None:
            ext = ".gif" if save_format == "gif" else ".mp4"
            filename = (
                f"time_series_m1_{m1}_m2_{m2}_k_{k}_L_{L}_"
                f"theta1_{theta_1_init}_theta2_{theta_2_init}{ext}"
            )
        save_animation(anim, SAVE_DIR / filename, fps, save_format)
        plt.close(fig)
    elif show:
        plt.show()

    return anim


def main():
    """Main function to run simulations and demonstrations."""
    header = "STARTING SPRING-COUPLED DOUBLE PENDULUM SIMULATION"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    default_params = {
        "theta_1_init": 0.0,
        "theta_2_init": 10.0,
        "m1": 1.0,
        "m2": 1.0,
        "k": 0.5,
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
            params["k"] = float(
                input(f"Enter spring constant (N/m)[{default_params['k']}] N/m): ")
                or default_params["k"]
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
            from pathlib import Path

            base_input = Path(base_input).stem
            filename = f"{base_input}.{params['save_format']}"

    print("\nRunning simulation with parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    demo_normal_modes()

    animation = double_pendulum_animation_with_plots(
        theta_1_init=params["theta_1_init"],
        theta_2_init=params["theta_2_init"],
        m1=params["m1"],
        m2=params["m2"],
        k=params["k"],
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
