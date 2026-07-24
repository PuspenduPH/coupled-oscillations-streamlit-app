"""
utils.py - Shared Utility and Helper Functions
==============================================
Common utilities used across the coupled-oscillation simulation scripts:
    * double_pendulum.py
    * triple_pendulum.py
    * mass_spring.py

Sections
--------
1. Drawing Helpers        -- draw_spring_with_hook, draw_pendulum_string,
                             draw_ceiling, draw_wall
2. Geometry & Layout      -- calculate_pendulum_mass_positions,
                             calculate_plot_limits, calculate_plot_limits_3,
                             calculate_fixed_hook_length,
                             calculate_fixed_hook_length_ms
3. ODE Solving            -- solve_ode
4. Frequency Analysis     -- estimate_normal_mode_frequencies, format_frequency
5. Animation Saving       -- save_animation
6. Subplot Styling        -- style_dark_subplot
7. Colour Palette         -- PALETTE

"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.patches import Rectangle
from scipy.integrate import solve_ivp
from scipy.linalg import eigh


@dataclass(frozen=True)
class SimulationData:
    """Pickle-friendly numeric output shared by the Streamlit studio.

    The animation objects deliberately do not live here: they contain Matplotlib
    state and are expensive to serialize.  This makes instances safe to cache
    with ``st.cache_data`` while the UI builds a fresh animation from them.
    """

    time: np.ndarray
    state: np.ndarray
    energy: np.ndarray
    theoretical_frequencies: tuple[float, ...]
    estimated_frequencies: tuple[float, ...]
    metadata: dict[str, Any]


def validate_simulation_inputs(**values: float) -> None:
    """Raise a clear ``ValueError`` for non-physical or unusable inputs."""

    for name, value in values.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero.")


# ===========================================================================
# 7.  COLOUR PALETTE
# ===========================================================================

#: Shared dark-theme colour constants used across all simulation scripts.
PALETTE: dict = {
    # Mass / pendulum bob colours
    "MASS1": "#FF4D6D",  # hot-pink-red   - mass / pendulum 1
    "MASS2": "#00EAFF",  # neon cyan       - mass / pendulum 2
    "MASS3": "#AAFF00",  # neon lime-green - mass / pendulum 3
    # Structure colours
    "SPRING": "#708090",  # slate grey  - springs
    "ROD": "#CCCCCC",  # light silver - pendulum rods
    "PIVOT": "#FF9F1C",  # amber        - pivot / anchor points
    # Guide & grid
    "GUIDE": "#555555",  # dim guide lines
    "GRID": "#444444",  # grid lines
    # Background panels
    "BG_MAIN": "#091217",  # main animation panel
    "BG_TH1": "#110000",  # theta1 / x1 time-series panel
    "BG_TH2": "#001100",  # theta2 / x2 time-series panel
    "BG_TH3": "#00001A",  # theta3 time-series panel (triple)
    "BG_PHASE": "#091217",  # phase-space panel
    # Text / annotation colours
    "TIME": "#FF9F1C",  # timer readout (amber)
    "INFO_MASS1": "#FF4D6D",  # per-frame info box  - mass 1
    "INFO_FREQ": "#00EAFF",  # frequency comparison - neon cyan
    "INFO_SYS": "#AAFF00",  # system parameters    - neon lime
    # Edge colours for mass markers
    "EDGE_MASS1": "#991133",
    "EDGE_MASS2": "#008899",
    "EDGE_MASS3": "#668800",
    # Bounding-box face / edge pairs
    "BOX_TIME_FACE": "#1a1000",
    "BOX_TIME_EDGE": "#CC8800",
    "BOX_INFO1_FACE": "#330011",
    "BOX_INFO1_EDGE": "#CC2244",
    "BOX_FREQ_FACE": "#003366",
    "BOX_FREQ_EDGE": "#00AACC",
    "BOX_SYS_FACE": "#1a3300",
    "BOX_SYS_EDGE": "#77BB00",
}


# ===========================================================================
# 1.  DRAWING HELPERS
# ===========================================================================


def draw_spring_with_hook(start_pos, end_pos, num_coils, radius, hook_length):
    """
    Compute (x, y) coordinates for a coiled spring with straight hook
    segments at each end.

    The spring runs from start_pos to end_pos along their connecting line.
    hook_length is consumed at each end as a straight segment; the rest is
    filled with num_coils helical coils of the given radius.

    Parameters
    ----------
    start_pos   : (float, float) -- starting attachment point
    end_pos     : (float, float) -- ending attachment point
    num_coils   : int            -- number of full coils
    radius      : float          -- perpendicular amplitude of each coil
    hook_length : float          -- length of each straight hook

    Returns
    -------
    x_full, y_full : np.ndarray
    """
    start_x, start_y = start_pos
    end_x, end_y = end_pos

    dx = end_x - start_x
    dy = end_y - start_y
    current_dist = np.sqrt(dx**2 + dy**2)

    if current_dist < 1e-9:
        return np.array([start_x]), np.array([start_y])

    angle = np.arctan2(dy, dx)

    spring_len = current_dist - 2 * hook_length
    if spring_len < 0:
        spring_len = 0
        actual_hook_len = current_dist / 2
    else:
        actual_hook_len = hook_length

    t = np.linspace(0, num_coils * 2 * np.pi, 300)
    s = np.linspace(0, 1, 300)

    start_coil_x = start_x + actual_hook_len * np.cos(angle)
    start_coil_y = start_y + actual_hook_len * np.sin(angle)

    coil_dx = spring_len * np.cos(angle)
    coil_dy = spring_len * np.sin(angle)

    x_base = start_coil_x + s * coil_dx
    y_base = start_coil_y + s * coil_dy

    perp_x = -np.sin(angle) * radius * np.sin(t)
    perp_y = np.cos(angle) * radius * np.sin(t)

    x_spring = x_base + perp_x
    y_spring = y_base + perp_y

    hook_start_x = np.linspace(start_x, start_coil_x, 50)
    hook_start_y = np.linspace(start_y, start_coil_y, 50)

    end_coil_x = start_coil_x + coil_dx
    end_coil_y = start_coil_y + coil_dy
    hook_end_x = np.linspace(end_coil_x, end_x, 50)
    hook_end_y = np.linspace(end_coil_y, end_y, 50)

    x_full = np.concatenate([hook_start_x, x_spring, hook_end_x])
    y_full = np.concatenate([hook_start_y, y_spring, hook_end_y])

    return x_full, y_full


def draw_pendulum_string(ax, x_anchor, y_anchor, x_mass, y_mass, color="#CCCCCC", lw=2):
    """Draw a straight pendulum rod from pivot to mass on ax."""
    ax.plot([x_anchor, x_mass], [y_anchor, y_mass], color=color, lw=lw)


def draw_ceiling(ax, ceiling_length, ceiling_height, color="#4A2E00"):
    """
    Draw a rectangular ceiling bar centred at (0, 0) on ax.

    Returns the Rectangle patch added to ax.
    """
    rect = Rectangle(
        (-ceiling_length / 2, -ceiling_height / 2),
        ceiling_length,
        ceiling_height,
        color=color,
        zorder=5,
    )
    ax.add_patch(rect)
    return rect


def draw_wall(ax, x_position, y_center, wall_width, wall_height, color="#966919"):
    """
    Draw a vertical wall rectangle on ax (for the mass-spring system).

    Returns the Rectangle patch added to ax.
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


# ===========================================================================
# 2.  GEOMETRY AND LAYOUT
# ===========================================================================


def calculate_pendulum_mass_positions(x_initial, string_length, theta_degrees):
    """
    Return the (x, y) position of a pendulum bob.

    Pivot is at (x_initial, 0). Positive theta displaces bob to the right.
    y is always <= 0 (below the pivot).

    Parameters
    ----------
    x_initial     : float -- horizontal pivot position
    string_length : float
    theta_degrees : float -- angle from vertical (degrees)

    Returns
    -------
    x_position, y_position : float
    """
    theta_rad = np.radians(theta_degrees)
    x_position = x_initial + string_length * np.sin(theta_rad)
    y_position = -string_length * np.cos(theta_rad)
    return x_position, y_position


def calculate_fixed_hook_length_three_pendulum(
    ceiling_length, string_length, separation_ratio, hook_ratio=0.12
):
    """
    Calculate the fixed hook length for three-pendulum system based on the relaxed state (theta=0).
    Returns hook length based on the distance between adjacent masses.
    """
    # Geometry in relaxed state
    anchor_separation = separation_ratio * ceiling_length
    left_anchor_x = -anchor_separation
    middle_anchor_x = 0

    # Positions at rest (theta = 0)
    left_mass_x, _ = calculate_pendulum_mass_positions(left_anchor_x, string_length, 0)
    middle_mass_x, _ = calculate_pendulum_mass_positions(
        middle_anchor_x, string_length, 0
    )

    # Distance between adjacent masses at rest
    relaxed_dist = np.abs(middle_mass_x - left_mass_x)

    # Hook length
    relaxed_spring_len = relaxed_dist / (1 + 2 * hook_ratio)
    hook_len = hook_ratio * relaxed_spring_len

    return hook_len


def calculate_plot_limits(
    ceiling_length, left_mass_x, right_mass_x, left_mass_y, right_mass_y, padding=0.15
):
    """
    Compute (xlim, ylim) axis limits for a two-pendulum animation panel.

    Encompasses the ceiling extent and extreme mass positions, with fractional
    padding on every side.

    Parameters
    ----------
    ceiling_length             : float
    left_mass_x, right_mass_x : float -- extreme x positions of both masses
    left_mass_y, right_mass_y : float -- extreme y positions of both masses
    padding                    : float (default 0.15)

    Returns
    -------
    xlim, ylim : (float, float)
    """
    x_min = min(-ceiling_length / 2, left_mass_x, right_mass_x)
    x_max = max(ceiling_length / 2, left_mass_x, right_mass_x)
    y_min = min(left_mass_y, right_mass_y)
    y_max = 0.0

    x_rng = x_max - x_min
    y_rng = y_max - y_min

    xlim = (x_min - x_rng * padding, x_max + x_rng * padding)
    ylim = (y_min - y_rng * padding, y_max + y_rng * padding)
    return xlim, ylim


def calculate_plot_limits_3(
    ceiling_length,
    left_mass_x,
    right_mass_x,
    left_mass_y,
    right_mass_y,
    padding=0.15,
    middle_mass_x=None,
    middle_mass_y=None,
):
    """
    Compute (xlim, ylim) for a two- or three-pendulum animation panel.

    Optionally includes a middle mass in the bounds calculation.

    Parameters
    ----------
    ceiling_length               : float
    left_mass_x, right_mass_x   : float
    left_mass_y, right_mass_y   : float
    padding                      : float (default 0.15)
    middle_mass_x, middle_mass_y : float or None

    Returns
    -------
    xlim, ylim : (float, float)
    """
    xs = [left_mass_x, right_mass_x]
    ys = [left_mass_y, right_mass_y]

    if middle_mass_x is not None and middle_mass_y is not None:
        xs.append(middle_mass_x)
        ys.append(middle_mass_y)

    x_min = min(-ceiling_length / 2, *xs)
    x_max = max(ceiling_length / 2, *xs)
    y_min = min(ys)
    y_max = 0.0

    x_rng = x_max - x_min
    y_rng = y_max - y_min

    xlim = (x_min - x_rng * padding, x_max + x_rng * padding)
    ylim = (y_min - y_rng * padding, y_max + y_rng * padding)
    return xlim, ylim


def calculate_fixed_hook_length(
    ceiling_length, string_length, separation_ratio, hook_ratio=0.12
):
    """
    Hook length for coupled-pendulum springs, derived from relaxed geometry
    (theta = 0).

    At rest: total_dist = relaxed_spring_len + 2 * hook_len,
    with hook_len = hook_ratio * relaxed_spring_len.

    Parameters
    ----------
    ceiling_length   : float
    string_length    : float
    separation_ratio : float  -- anchor_separation / ceiling_length
    hook_ratio       : float  (default 0.12)

    Returns
    -------
    hook_len : float
    """
    anchor_sep = separation_ratio * ceiling_length
    left_x = -anchor_sep / 2
    right_x = anchor_sep / 2

    lx, _ = calculate_pendulum_mass_positions(left_x, string_length, 0)
    rx, _ = calculate_pendulum_mass_positions(right_x, string_length, 0)

    relaxed_dist = abs(rx - lx)
    relaxed_spring_len = relaxed_dist / (1 + 2 * hook_ratio)
    return hook_ratio * relaxed_spring_len


def calculate_fixed_hook_length_ms(relaxed_spring_length, hook_ratio=0.12):
    """
    Hook length for mass-spring system springs.

    Parameters
    ----------
    relaxed_spring_length : float -- natural length of spring incl. hooks
    hook_ratio            : float (default 0.12)

    Returns
    -------
    hook_len : float
    """
    coiled = relaxed_spring_length / (1 + 2 * hook_ratio)
    return hook_ratio * coiled


# ===========================================================================
# 3.  ODE SOLVING
# ===========================================================================


def solve_ode(
    derivatives, t_span, y0, t_eval, args=(), rtol=1e-5, atol=1e-7, method="RK45"
):
    """
    Thin wrapper around scipy.integrate.solve_ivp with dense output.

    Prints progress messages compatible with the simulation scripts.

    Parameters
    ----------
    derivatives : callable  -- f(t, y, *args)
    t_span      : (float, float)
    y0          : array-like
    t_eval      : np.ndarray -- output time points
    args        : tuple      -- extra arguments for derivatives
    rtol, atol  : float
    method      : str        (default "RK45")

    Returns
    -------
    y : np.ndarray, shape (n_states, len(t_eval))

    Raises
    ------
    RuntimeError if dense solution unavailable.
    """
    print("Solving differential equations...")
    sol = solve_ivp(
        derivatives,
        t_span,
        list(y0),
        args=args,
        method=method,
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )

    if sol.sol is None:
        raise RuntimeError(
            "solve_ivp did not return a dense solution (sol.sol is None)."
        )

    y = sol.sol(t=t_eval)
    print(f"Solution computed: {len(t_eval)} time steps")
    return y


# ===========================================================================
# 4.  FREQUENCY ANALYSIS (FFT)
# ===========================================================================


def estimate_normal_mode_frequencies(
    t, coords, M, K, use_hann_window=True, rms_threshold=1e-2
):
    """
    Estimate normal-mode frequencies from simulated time-domain data.

    Steps
    -----
    1. Solve generalised eigenproblem  K v = omega^2 M v  -> modal matrix V.
    2. Project coordinates onto modes: q(t) = V^T M coords(t).
    3. Optionally apply a Hann window to reduce spectral leakage.
    4. FFT each modal coordinate; pick the dominant peak (skipping DC).

    Parameters
    ----------
    t             : np.ndarray, shape (n,)       -- uniform time array
    coords        : np.ndarray, shape (n_dof, n) -- one DOF per row
    M             : np.ndarray (n_dof x n_dof)   -- symmetric mass matrix
    K             : np.ndarray (n_dof x n_dof)   -- symmetric stiffness matrix
    use_hann_window : bool  (default True)
    rms_threshold   : float -- modes below this relative RMS -> 0.0 Hz
                               (default 1e-2)

    Returns
    -------
    f_est : list[float]  -- Hz per mode, ordered low-to-high;
                            0.0 means the mode was not excited.
    """
    t = np.asarray(t)
    dt = t[1] - t[0]
    n = t.size

    coords = np.asarray(coords, dtype=float)
    coords -= coords.mean(axis=1, keepdims=True)  # removes DC per channel

    omega_sq, V = eigh(K, M)
    omega_sq = np.maximum(omega_sq, 0.0)  # guards tiny negatives

    q = V.T @ (M @ coords)  # modal coordinates

    if use_hann_window:
        q *= np.hanning(n)

    freqs = np.fft.rfftfreq(n, dt)
    amp = np.abs(np.fft.rfft(q, axis=1))

    rms = np.sqrt(np.mean(amp**2, axis=1))
    rms_rel = rms / (rms.max() + 1e-30)

    f_est = []
    for i in range(q.shape[0]):
        if rms_rel[i] < rms_threshold:
            f_est.append(0.0)
        else:
            idx = int(np.argmax(amp[i, 1:])) + 1  # skips DC
            f_est.append(float(freqs[idx]))

    return f_est


def format_frequency(f, decimals=3):
    """
    Format a frequency (Hz) for display in animation annotations.

    Returns "X.XXX Hz" for f > 0, otherwise "N/A (Not Excited)".
    """
    return f"{f:.{decimals}f} Hz" if f > 0 else "N/A (Not Excited)"


# ===========================================================================
# 5.  ANIMATION SAVING
# ===========================================================================

SCRIPT_DIR = Path(__file__).parent
SAVE_DIR = SCRIPT_DIR / "outputs"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def save_animation(anim, save_path, fps, save_format="gif", dpi=120):
    """
    Save a matplotlib FuncAnimation as GIF or MP4.

    For MP4, uses FFMpegWriter with H.264 + yuv420p for broad compatibility.

    Parameters
    ----------
    anim        : matplotlib.animation.FuncAnimation
    save_path   : str or pathlib.Path  -- full output path incl. extension
    fps         : int
    save_format : str  "gif" or "mp4"  (default "gif")
    dpi         : int  (default 120)
    """
    from matplotlib.animation import FFMpegWriter, PillowWriter

    save_path = Path(save_path)
    print(f"Saving animation to {save_path.resolve()} ...")
    try:
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
            anim.save(save_path, writer=writer)
        else:
            anim.save(save_path, writer=PillowWriter(fps=fps), dpi=dpi)
        print("Animation saved successfully!")
    except Exception as exc:
        print(f"Error saving animation: {exc}")


# ===========================================================================
# 6.  SUBPLOT STYLING
# ===========================================================================


def style_dark_subplot(
    ax,
    xlabel,
    ylabel,
    title,
    grid_color=PALETTE["GRID"],
    grid_alpha=0.3,
    label_fontsize=10,
    title_fontsize=12,
):
    """
    Apply the shared dark-theme style to a Matplotlib Axes object.

    Configures labels, title, tick colours, spine edge colours, and grid to
    match the dark colour palette used across all simulation scripts.

    Parameters
    ----------
    ax             : matplotlib.axes.Axes
    xlabel, ylabel : str
    title          : str
    grid_color     : str   (default PALETTE["GRID"] = "#444444")
    grid_alpha     : float (default 0.3)
    label_fontsize : int   (default 10)
    title_fontsize : int   (default 12)
    """
    ax.set_xlabel(xlabel, fontsize=label_fontsize, color="white")
    ax.set_ylabel(ylabel, fontsize=label_fontsize, color="white")
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", color="white")
    ax.tick_params(colors="white", which="both")
    for sp in ax.spines.values():
        sp.set_edgecolor(PALETTE["GUIDE"])
    ax.grid(True, alpha=grid_alpha, color=grid_color)


# ===========================================================================
# Public API
# ===========================================================================

__all__ = [
    "SimulationData",
    "validate_simulation_inputs",
    "draw_spring_with_hook",
    "draw_pendulum_string",
    "draw_ceiling",
    "draw_wall",
    "calculate_pendulum_mass_positions",
    "calculate_plot_limits",
    "calculate_plot_limits_3",
    "calculate_fixed_hook_length",
    "calculate_fixed_hook_length_ms",
    "solve_ode",
    "estimate_normal_mode_frequencies",
    "format_frequency",
    "SAVE_DIR",
    "save_animation",
    "style_dark_subplot",
    "PALETTE",
]
