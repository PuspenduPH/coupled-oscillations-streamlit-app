"""Coupled Oscillators Studio.

The app is intentionally a thin orchestration layer over the existing physics
modules. This app calls the existing modules and displays the results in the form
of animations and plots.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.animation import HTMLWriter

import double_pendulum
import mass_spring
import triple_pendulum
from utils import (
    PALETTE,
    SAVE_DIR,
    SimulationData,
    calculate_fixed_hook_length,
    calculate_fixed_hook_length_ms,
    calculate_fixed_hook_length_three_pendulum,
    calculate_pendulum_mass_positions,
    calculate_plot_limits,
    calculate_plot_limits_3,
    draw_ceiling,
    draw_pendulum_string,
    draw_spring_with_hook,
    draw_wall,
)


def draw_coupled_pendulum(
    theta_1=0,
    theta_2=5,
    ceiling_length=5,
    string_length=10,
    separation_ratio=0.5,
    num_coils=8,
    mass_size=15,
    figsize=(8, 6),
    padding=0.06,
):
    """
    Draws a coupled pendulum system.
    """
    # System dimensions
    ceiling_height = 0.15 * ceiling_length
    anchor_separation = separation_ratio * ceiling_length

    # Anchor positions
    left_anchor_x = -anchor_separation / 2
    right_anchor_x = anchor_separation / 2
    anchor_y = 0

    # Mass positions
    left_mass_x, left_mass_y = calculate_pendulum_mass_positions(
        left_anchor_x, string_length, theta_1
    )
    right_mass_x, right_mass_y = calculate_pendulum_mass_positions(
        right_anchor_x, string_length, theta_2
    )

    # Hook length based on relaxed state
    hook_length = calculate_fixed_hook_length(
        ceiling_length, string_length, separation_ratio
    )

    # Spring radius
    spring_radius = 0.1 * ceiling_length

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PALETTE["BG_MAIN"])
    ax.set_facecolor(PALETTE["BG_MAIN"])

    # Ceiling
    draw_ceiling(ax, ceiling_length, ceiling_height)

    # Pendulum strings
    draw_pendulum_string(ax, left_anchor_x, anchor_y, left_mass_x, left_mass_y)
    draw_pendulum_string(ax, right_anchor_x, anchor_y, right_mass_x, right_mass_y)

    # Spring
    x_spring, y_spring = draw_spring_with_hook(
        (left_mass_x, left_mass_y),
        (right_mass_x, right_mass_y),
        num_coils=num_coils,
        radius=spring_radius,
        hook_length=hook_length,
    )
    ax.plot(x_spring, y_spring, color=PALETTE["SPRING"], lw=2)
    ax.text(
        0,
        (left_mass_y + right_mass_y) / 2 + 0.1 * string_length,
        "$k$",
        color=PALETTE["MASS2"],
        ha="center",
    )

    # Masses
    ax.plot(left_mass_x, left_mass_y, "o", color=PALETTE["MASS1"], ms=mass_size)
    ax.plot(right_mass_x, right_mass_y, "o", color=PALETTE["MASS2"], ms=mass_size)

    ax.text(
        left_mass_x,
        left_mass_y - 0.18 * string_length,
        "$m_1$",
        color=PALETTE["MASS1"],
        ha="center",
        fontweight="bold",
    )
    ax.text(
        right_mass_x,
        right_mass_y - 0.18 * string_length,
        "$m_2$",
        color=PALETTE["MASS2"],
        ha="center",
        fontweight="bold",
    )

    xlim, ylim = calculate_plot_limits(
        ceiling_length, left_mass_x, right_mass_x, left_mass_y, right_mass_y, padding
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis("off")
    plt.tight_layout()

    return fig, ax


def draw_three_coupled_pendulum(
    theta_1=0,
    theta_2=0,
    theta_3=10,
    ceiling_length=12,
    string_length=12,
    separation_ratio=0.6,
    num_coils=8,
    mass_size=15,
    figsize=(8, 6),
    padding=0.15,
):
    """
    Draws a three coupled pendulum system with two springs.
    m1 connected to m2 by spring 1, m2 connected to m3 by spring 2.
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
    spring_radius = 0.06 * ceiling_length

    # Figure
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PALETTE["BG_MAIN"])
    ax.set_facecolor(PALETTE["BG_MAIN"])

    # Ceiling
    ceiling_width = 2 * anchor_separation + ceiling_length * 0.3
    draw_ceiling(ax, ceiling_width, ceiling_height)

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
    ax.plot(x_spring1, y_spring1, color=PALETTE["SPRING"], lw=2)

    # Second spring (m2 to m3)
    x_spring2, y_spring2 = draw_spring_with_hook(
        (middle_mass_x, middle_mass_y),
        (right_mass_x, right_mass_y),
        num_coils=num_coils,
        radius=spring_radius,
        hook_length=hook_length,
    )
    ax.plot(x_spring2, y_spring2, color=PALETTE["SPRING"], lw=2)

    # Masses
    ax.plot(left_mass_x, left_mass_y, "o", color=PALETTE["MASS1"], ms=mass_size)
    ax.plot(middle_mass_x, middle_mass_y, "o", color=PALETTE["MASS2"], ms=mass_size)
    ax.plot(right_mass_x, right_mass_y, "o", color=PALETTE["MASS3"], ms=mass_size)

    ax.text(
        left_mass_x,
        left_mass_y - 0.15 * string_length,
        "$m_1$",
        color=PALETTE["MASS1"],
        ha="center",
        fontweight="bold",
    )
    ax.text(
        middle_mass_x,
        middle_mass_y - 0.15 * string_length,
        "$m_2$",
        color=PALETTE["MASS2"],
        ha="center",
        fontweight="bold",
    )
    ax.text(
        right_mass_x,
        right_mass_y - 0.15 * string_length,
        "$m_3$",
        color=PALETTE["MASS3"],
        ha="center",
        fontweight="bold",
    )

    ax.text(
        left_anchor_x / 2,
        (left_mass_y + middle_mass_y) / 2 + 0.1 * string_length,
        "$k_1$",
        color=PALETTE["MASS2"],
        ha="center",
    )
    ax.text(
        right_anchor_x / 2,
        (middle_mass_y + right_mass_y) / 2 + 0.1 * string_length,
        "$k_2$",
        color=PALETTE["MASS2"],
        ha="center",
    )

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
    ax.axis("off")
    plt.tight_layout()

    return fig, ax


def draw_coupled_mass_spring_system(
    x1=0.0,
    x2=0.0,
    system_length=12.0,
    y_position=0.0,
    num_coils=8,
    mass_size=18,
    padding=0.5,
):
    """
    Draw a coupled mass-spring system with horizontal motion only.
    Configuration: Left Wall → Spring1 → Mass1 → Spring2 → Mass2 → Spring3 → Right Wall

    Masses are in equilibrium at L/3 and 2L/3.
    x1 and x2 are DISPLACEMENTS from equilibrium.
    """
    # Wall positions
    left_wall_x = 0.0
    right_wall_x = system_length

    # Wall dimensions
    wall_width = 0.4
    wall_height = 1.5

    # Equilibrium positions
    eq_x1 = system_length / 3.0
    eq_x2 = 2.0 * system_length / 3.0

    # Actual positions based on displacement
    pos_x1 = eq_x1 + x1
    pos_x2 = eq_x2 + x2

    # Relaxed spring length (distance between equilibrium points)
    relaxed_spring_length = system_length / 3.0

    # Hook length based on relaxed spring length
    hook_length = calculate_fixed_hook_length_ms(relaxed_spring_length)

    # Spring radius
    spring_radius = 0.15

    # Figure
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(PALETTE["BG_MAIN"])
    ax.set_facecolor(PALETTE["BG_MAIN"])

    # Left wall
    draw_wall(ax, left_wall_x, y_position, wall_width, wall_height)

    # Right wall
    draw_wall(ax, right_wall_x, y_position, wall_width, wall_height)

    # Spring 1: Left Wall to Mass 1
    spring1_start = (left_wall_x + wall_width / 2, y_position)
    spring1_end = (pos_x1, y_position)
    x_spring1, y_spring1 = draw_spring_with_hook(
        spring1_start,
        spring1_end,
        num_coils=num_coils,
        radius=spring_radius,
        hook_length=hook_length,
    )
    ax.plot(x_spring1, y_spring1, color=PALETTE["SPRING"], lw=2)

    # Spring 2: Mass 1 to Mass 2
    spring2_start = (pos_x1, y_position)
    spring2_end = (pos_x2, y_position)
    x_spring2, y_spring2 = draw_spring_with_hook(
        spring2_start,
        spring2_end,
        num_coils=num_coils,
        radius=spring_radius,
        hook_length=hook_length,
    )
    ax.plot(x_spring2, y_spring2, color=PALETTE["SPRING"], lw=2)

    # Spring 3: Mass 2 to Right Wall
    spring3_start = (pos_x2, y_position)
    spring3_end = (right_wall_x - wall_width / 2, y_position)
    x_spring3, y_spring3 = draw_spring_with_hook(
        spring3_start,
        spring3_end,
        num_coils=num_coils,
        radius=spring_radius,
        hook_length=hook_length,
    )
    ax.plot(x_spring3, y_spring3, color=PALETTE["SPRING"], lw=2)

    # Masses
    ax.plot(pos_x1, y_position, "o", color=PALETTE["MASS1"], ms=mass_size, zorder=10)
    ax.plot(pos_x2, y_position, "o", color=PALETTE["MASS2"], ms=mass_size, zorder=10)

    ax.text(
        pos_x1,
        y_position - wall_height * 0.3,
        "$m_1$",
        color=PALETTE["MASS1"],
        ha="center",
        fontweight="bold",
    )
    ax.text(
        pos_x2,
        y_position - wall_height * 0.3,
        "$m_2$",
        color=PALETTE["MASS2"],
        ha="center",
        fontweight="bold",
    )

    ax.text(
        (left_wall_x + pos_x1) / 2,
        y_position + wall_height * 0.2,
        "$k_1$",
        color=PALETTE["MASS2"],
        ha="center",
    )
    ax.text(
        (pos_x1 + pos_x2) / 2,
        y_position + wall_height * 0.2,
        "$k_2$",
        color=PALETTE["MASS2"],
        ha="center",
    )
    ax.text(
        (pos_x2 + right_wall_x) / 2,
        y_position + wall_height * 0.2,
        "$k_3$",
        color=PALETTE["MASS2"],
        ha="center",
    )

    # Plot limits
    x_min = left_wall_x - padding
    x_max = right_wall_x + padding
    y_min = y_position - wall_height / 2 - padding
    y_max = y_position + wall_height / 2 + padding

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis("off")
    plt.tight_layout()

    return fig, ax


# Embed limit has been increased to 1000 MB to prevent truncation on long animations
plt.rcParams["animation.embed_limit"] = 1000.0


st.set_page_config(
    page_title="Coupled Oscillators Studio",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)


SYSTEMS: dict[str, dict[str, Any]] = {
    "double": {
        "title": "🔗 Double Pendulum",
        "short": "Double Pendulum",
        "accent": PALETTE["MASS1"],
        "function": double_pendulum.double_pendulum_animation_with_plots,
        "frequencies": ("Mode 1 · in phase", "Mode 2 · out of phase"),
    },
    "triple": {
        "title": "⚓ Triple Pendulum",
        "short": "Triple Pendulum",
        "accent": PALETTE["MASS3"],
        "function": triple_pendulum.three_coupled_pendulum_animation_with_plots,
        "frequencies": (
            "Mode 1 · sloshing",
            "Mode 2 · anti-symmetric",
            "Mode 3 · breathing",
        ),
    },
    "spring": {
        "title": "🔧 Mass–Spring Chain",
        "short": "Mass–Spring Chain",
        "accent": PALETTE["PIVOT"],
        "function": mass_spring.coupled_mass_spring_system_animation_with_plots,
        "frequencies": ("Mode 1 · low", "Mode 2 · high"),
    },
}


def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');
        :root {{
            --bg: {PALETTE["BG_MAIN"]};
            --panel: #101a20;
            --panel-2: #14232b;
            --pink: {PALETTE["MASS1"]};
            --cyan: {PALETTE["MASS2"]};
            --lime: {PALETTE["MASS3"]};
            --amber: {PALETTE["PIVOT"]};
            --muted: #8ea0aa;
        }}
        html, body, [data-testid="stAppViewContainer"] {{
            background: radial-gradient(circle at 8% 0%, #17313a 0%, var(--bg) 34%, #060a0d 100%);
            color: #eef7f8;
            font-family: Inter, ui-sans-serif, system-ui, sans-serif;
        }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0c151a 0%, #071014 100%);
            border-right: 1px solid #21343c;
        }}
        [data-testid="stSidebar"] > div:first-child {{ padding-top: 1.25rem; }}
        .hero {{ padding: 0.8rem 0 0.3rem; }}
        .eyebrow {{ color: var(--cyan); font: 700 0.72rem 'Space Mono', monospace; letter-spacing: .18em; text-transform: uppercase; }}
        .hero h1 {{ margin: .2rem 0 .25rem; font-size: clamp(2rem, 5vw, 4.2rem); letter-spacing: -.055em; line-height: 1; }}
        .hero p {{ color: #b8c8cd; max-width: 760px; font-size: 1.03rem; margin: 0; }}
        .spectrum {{ height: 3px; margin: 1.2rem 0 1.4rem; border-radius: 99px; background: linear-gradient(90deg, var(--pink), var(--cyan), var(--lime), var(--amber)); box-shadow: 0 0 24px #00eaff44; }}
        .section-kicker {{ color: var(--cyan); font: 700 .75rem 'Space Mono', monospace; letter-spacing: .12em; text-transform: uppercase; margin: .35rem 0 .8rem; }}
        .theory-card, .empty-state {{ background: linear-gradient(135deg, #112029cc, #0b1419dd); border: 1px solid #28424c; border-radius: 18px; padding: 1rem 1.15rem; margin-bottom: 1rem; }}
        .empty-state {{ min-height: 155px; display: grid; place-items: center; text-align: center; color: var(--muted); border-style: dashed; }}
        .empty-orbit {{ color: var(--cyan); font: 700 2.2rem 'Space Mono', monospace; letter-spacing: .15em; }}
        .caption {{ color: var(--muted); font-size: .86rem; }}
        .mono {{ font-family: 'Space Mono', monospace; }}
        div[data-testid="stMetric"] {{ background: #10202a; border: 1px solid #2a4a56; border-radius: 14px; padding: .8rem .9rem; }}
        div[data-testid="stMetricLabel"] {{ color: #9ab3bc; }}
        div[data-testid="stMetricValue"] {{ font-family: 'Space Mono', monospace; color: #f3fbfb; }}
        .stButton > button, .stDownloadButton > button {{ border-radius: 11px; border: 1px solid #2c6978; background: linear-gradient(135deg, #12313a, #10242c); color: #ecffff; font-weight: 700; transition: transform .15s, border-color .15s; }}
        .stButton > button:hover, .stDownloadButton > button:hover {{ transform: translateY(-1px); border-color: var(--cyan); color: white; }}
        div[data-baseweb="tab-list"] {{ gap: .45rem; background: transparent; }}
        button[data-baseweb="tab"] {{ color: #9eb3ba; border-radius: 12px 12px 0 0; padding: .75rem 1rem; }}
        button[data-baseweb="tab"][aria-selected="true"] {{ color: white; background: #12242c; box-shadow: inset 0 -3px 0 var(--cyan); }}
        [data-testid="stExpander"] {{ border-color: #29434d; background: #0c171d99; border-radius: 12px; }}
        [data-testid="stSidebar"] label {{ color: #d5e3e6; }}
        footer {{ visibility: hidden; }}
        .footer {{ border-top: 1px solid #263a42; margin-top: 2rem; padding: 1rem 0 0; color: #748890; font-size: .78rem; display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }}
        @media (max-width: 768px) {{
            .hero h1 {{ font-size: 2.35rem; }}
            .hero p {{ font-size: .92rem; }}
            [data-testid="stSidebar"] {{ min-width: 300px; }}
            .footer {{ display: block; line-height: 1.7; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _set_default(key: str, value: Any) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def common_controls() -> dict[str, Any]:
    _set_default("common_g", 9.81)
    _set_default("common_time", 20.0)
    _set_default("common_fps", 30)
    _set_default("common_trace", 50)
    with st.sidebar.expander("⚙️ Common Settings", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            g = st.number_input(
                "Gravity g",
                min_value=0.01,
                max_value=30.0,
                step=0.01,
                key="common_g",
                help="Gravitational acceleration in m/s².",
            )
            fps = st.slider(
                "FPS",
                min_value=10,
                max_value=60,
                key="common_fps",
                help="Animation frames per second; higher values increase HTML size.",
            )
        with c2:
            sim_time = st.slider(
                "Simulation time",
                min_value=5.0,
                max_value=90.0,
                step=2.0,
                key="common_time",
                help="Total simulated time in seconds.",
            )
            trace = st.slider(
                "Trace length",
                min_value=5,
                max_value=300,
                key="common_trace",
                help="Number of recent positions retained in the motion trail.",
            )
    return {"g": g, "simulation_time": sim_time, "fps": fps, "trace_length": trace}


def sidebar_controls() -> dict[str, dict[str, Any]]:
    common = common_controls()

    def set_preset(**kwargs):
        for k, v in kwargs.items():
            st.session_state[k] = v

    def styled_preset_button(
        label: str, key: str, kwargs: dict, is_active: bool, active_color: str
    ):
        marker_class = f"marker_{key}"
        st.markdown(
            f'<div class="{marker_class}" style="display:none;"></div>',
            unsafe_allow_html=True,
        )
        if is_active:
            st.markdown(
                f"""
            <style>
            div[data-testid="stElementContainer"]:has(.{marker_class}) + div[data-testid="stElementContainer"] button,
            div[data-testid="stElementContainer"]:has(.{marker_class}) + div[data-testid="stElementContainer"] button[kind="primary"],
            div[data-testid="stElementContainer"]:has(.{marker_class}) + div[data-testid="stElementContainer"] button[data-testid="stBaseButton-primary"],
            div.element-container:has(.{marker_class}) + div.element-container button {{
                background-color: {active_color} !important;
                background: {active_color} !important;
                border-color: {active_color} !important;
                color: white !important;
            }}
            div[data-testid="stElementContainer"]:has(.{marker_class}) + div[data-testid="stElementContainer"] button:hover,
            div.element-container:has(.{marker_class}) + div.element-container button:hover {{
                background-color: {active_color} !important;
                background: {active_color} !important;
                border-color: {active_color} !important;
                filter: brightness(1.15);
                color: white !important;
            }}
            </style>
            """,
                unsafe_allow_html=True,
            )
        st.button(
            label,
            key=key,
            type="primary" if is_active else "secondary",
            use_container_width=True,
            on_click=set_preset,
            kwargs=kwargs,
        )

    with st.sidebar.expander("🔗 Double Pendulum Parameters", expanded=True):
        _set_default("double_theta1", 0.0)
        _set_default("double_theta2", 10.0)
        p1, p2, p3 = st.columns(3)
        cur_t1 = st.session_state.get("double_theta1")
        cur_t2 = st.session_state.get("double_theta2")
        with p1:
            styled_preset_button(
                "In-phase",
                "preset_in",
                {"double_theta1": 10.0, "double_theta2": 10.0},
                cur_t1 == 10.0 and cur_t2 == 10.0,
                "#E74C3C",
            )
        with p2:
            styled_preset_button(
                "Out-of-phase",
                "preset_out",
                {"double_theta1": 10.0, "double_theta2": -10.0},
                cur_t1 == 10.0 and cur_t2 == -10.0,
                "#3498DB",
            )
        with p3:
            styled_preset_button(
                "Beats",
                "preset_beats",
                {"double_theta1": 0.0, "double_theta2": 10.0},
                cur_t1 == 0.0 and cur_t2 == 10.0,
                "#2ECC71",
            )
        c1, c2 = st.columns(2)
        with c1:
            theta1 = st.slider(
                "θ₁ initial (°)",
                -180.0,
                180.0,
                key="double_theta1",
                help="Initial angle of pendulum 1, measured from vertical.",
            )
            m1 = st.number_input(
                "m₁ (kg)",
                min_value=0.001,
                value=1.0,
                step=0.1,
                key="double_m1",
                help="Mass of pendulum bob 1 in kilograms.",
            )
            k = st.number_input(
                "k (N/m)",
                min_value=0.001,
                value=0.5,
                step=0.25,
                key="double_k",
                help="Coupling spring constant in newtons per metre.",
            )
        with c2:
            theta2 = st.slider(
                "θ₂ initial (°)",
                -180.0,
                180.0,
                key="double_theta2",
                help="Initial angle of pendulum 2, measured from vertical.",
            )
            m2 = st.number_input(
                "m₂ (kg)",
                min_value=0.001,
                value=1.0,
                step=0.1,
                key="double_m2",
                help="Mass of pendulum bob 2 in kilograms.",
            )
            L = st.number_input(
                "L (m)",
                min_value=0.01,
                value=2.0,
                step=0.1,
                key="double_L",
                help="Length of each pendulum rod in metres.",
            )
    with st.sidebar.expander("⚓ Triple Pendulum Parameters"):
        _set_default("triple_theta1", 0.0)
        _set_default("triple_theta2", 10.0)
        _set_default("triple_theta3", 0.0)
        p1, p2, p3 = st.columns(3)
        cur_t1 = st.session_state.get("triple_theta1")
        cur_t2 = st.session_state.get("triple_theta2")
        cur_t3 = st.session_state.get("triple_theta3")
        with p1:
            styled_preset_button(
                "Sloshing",
                "tp_slosh",
                {"triple_theta1": 10.0, "triple_theta2": 10.0, "triple_theta3": 10.0},
                cur_t1 == 10.0 and cur_t2 == 10.0 and cur_t3 == 10.0,
                "#E74C3C",
            )
        with p2:
            styled_preset_button(
                "Anti-Sym",
                "tp_anti",
                {"triple_theta1": 10.0, "triple_theta2": 0.0, "triple_theta3": -10.0},
                cur_t1 == 10.0 and cur_t2 == 0.0 and cur_t3 == -10.0,
                "#9B59B6",
            )
        with p3:
            styled_preset_button(
                "Breathing",
                "tp_breath",
                {"triple_theta1": 10.0, "triple_theta2": -20.0, "triple_theta3": 10.0},
                cur_t1 == 10.0 and cur_t2 == -20.0 and cur_t3 == 10.0,
                "#E67E22",
            )
        c1, c2 = st.columns(2)
        with c1:
            t1 = st.slider(
                "θ₁ initial (°)",
                -180.0,
                180.0,
                0.0,
                key="triple_theta1",
                help="Initial angle of pendulum 1.",
            )
            t2 = st.slider(
                "θ₂ initial (°)",
                -180.0,
                180.0,
                10.0,
                key="triple_theta2",
                help="Initial angle of pendulum 2.",
            )
            m1t = st.number_input(
                "m₁ (kg)",
                min_value=0.001,
                value=1.0,
                step=0.1,
                key="triple_m1",
                help="Mass of bob 1.",
            )
            k1 = st.number_input(
                "k₁ (N/m)",
                min_value=0.001,
                value=0.5,
                step=0.25,
                key="triple_k1",
                help="Spring constant between masses 1 and 2.",
            )
        with c2:
            t3 = st.slider(
                "θ₃ initial (°)",
                -180.0,
                180.0,
                0.0,
                key="triple_theta3",
                help="Initial angle of pendulum 3.",
            )
            m2t = st.number_input(
                "m₂ (kg)",
                min_value=0.001,
                value=1.0,
                step=0.1,
                key="triple_m2",
                help="Mass of bob 2.",
            )
            m3t = st.number_input(
                "m₃ (kg)",
                min_value=0.001,
                value=1.0,
                step=0.1,
                key="triple_m3",
                help="Mass of bob 3.",
            )
            k2 = st.number_input(
                "k₂ (N/m)",
                min_value=0.001,
                value=0.5,
                step=0.25,
                key="triple_k2",
                help="Spring constant between masses 2 and 3.",
            )
            Lt = st.number_input(
                "L (m)",
                min_value=0.01,
                value=2.0,
                step=0.1,
                key="triple_L",
                help="Length of each pendulum rod.",
            )
    with st.sidebar.expander("🔧 Mass–Spring Chain Parameters"):
        _set_default("spring_x1", 0.5)
        _set_default("spring_x2", -0.5)
        p1, p2, p3 = st.columns(3)
        cur_x1 = st.session_state.get("spring_x1")
        cur_x2 = st.session_state.get("spring_x2")
        with p1:
            styled_preset_button(
                "In-phase",
                "sp_in",
                {
                    "spring_x1": 1.0,
                    "spring_x2": 1.0,
                    "spring_v1": 0.0,
                    "spring_v2": 0.0,
                },
                cur_x1 == 1.0 and cur_x2 == 1.0,
                "#E74C3C",
            )
        with p2:
            styled_preset_button(
                "Out-of-phase",
                "sp_out",
                {
                    "spring_x1": 1.0,
                    "spring_x2": -1.0,
                    "spring_v1": 0.0,
                    "spring_v2": 0.0,
                },
                cur_x1 == 1.0 and cur_x2 == -1.0,
                "#3498DB",
            )
        with p3:
            styled_preset_button(
                "Beats",
                "sp_beats",
                {
                    "spring_x1": 1.0,
                    "spring_x2": 0.0,
                    "spring_v1": 0.0,
                    "spring_v2": 0.0,
                },
                cur_x1 == 1.0 and cur_x2 == 0.0,
                "#2ECC71",
            )
        c1, c2 = st.columns(2)
        with c1:
            x1 = st.number_input(
                "x₁ initial (m)",
                min_value=-10.0,
                max_value=10.0,
                value=0.5,
                step=0.05,
                key="spring_x1",
                help="Initial displacement of mass 1 from equilibrium.",
            )
            v1 = st.number_input(
                "v₁ initial (m/s)",
                min_value=-20.0,
                max_value=20.0,
                value=0.0,
                step=0.1,
                key="spring_v1",
                help="Initial velocity of mass 1.",
            )
            m1s = st.number_input(
                "m₁ (kg)",
                min_value=0.001,
                value=1.0,
                step=0.1,
                key="spring_m1",
                help="Mass of the first block.",
            )
            k1s = st.number_input(
                "k₁ (N/m)",
                min_value=0.001,
                value=10.0,
                step=0.5,
                key="spring_k1",
                help="Left wall spring constant.",
            )
            k2s = st.number_input(
                "k₂ (N/m)",
                min_value=0.001,
                value=10.0,
                step=0.5,
                key="spring_k2",
                help="Coupling spring constant.",
            )
        with c2:
            x2 = st.number_input(
                "x₂ initial (m)",
                min_value=-10.0,
                max_value=10.0,
                value=-0.5,
                step=0.05,
                key="spring_x2",
                help="Initial displacement of mass 2 from equilibrium.",
            )
            v2 = st.number_input(
                "v₂ initial (m/s)",
                min_value=-20.0,
                max_value=20.0,
                value=0.0,
                step=0.1,
                key="spring_v2",
                help="Initial velocity of mass 2.",
            )
            m2s = st.number_input(
                "m₂ (kg)",
                min_value=0.001,
                value=1.0,
                step=0.1,
                key="spring_m2",
                help="Mass of the second block.",
            )
            k3s = st.number_input(
                "k₃ (N/m)",
                min_value=0.001,
                value=10.0,
                step=0.5,
                key="spring_k3",
                help="Right wall spring constant.",
            )
            length = st.number_input(
                "System length (m)",
                min_value=0.1,
                value=12.0,
                step=0.5,
                key="spring_length",
                help="Distance between the two fixed walls.",
            )
    return {
        "double": {
            **common,
            "theta_1_init": theta1,
            "theta_2_init": theta2,
            "m1": m1,
            "m2": m2,
            "k": k,
            "L": L,
        },
        "triple": {
            **common,
            "theta_1_init": t1,
            "theta_2_init": t2,
            "theta_3_init": t3,
            "m1": m1t,
            "m2": m2t,
            "m3": m3t,
            "k1": k1,
            "k2": k2,
            "L": Lt,
        },
        "spring": {k: v for k, v in common.items() if k != "g"}
        | {
            "x1_init": x1,
            "x2_init": x2,
            "v1_init": v1,
            "v2_init": v2,
            "m1": m1s,
            "m2": m2s,
            "k1": k1s,
            "k2": k2s,
            "k3": k3s,
            "system_length": length,
        },
    }


@st.cache_data(show_spinner=False)
def cached_data(
    system: str, parameter_items: tuple[tuple[str, Any], ...]
) -> SimulationData:
    params = dict(parameter_items)
    return SYSTEMS[system]["function"](**params, return_data=True, show=False)


def create_animation(
    system: str,
    params: dict[str, Any],
    data: SimulationData,
    *,
    save_anim: bool = False,
    save_format: str = "gif",
    filename: str | None = None,
):
    return SYSTEMS[system]["function"](
        **params,
        precomputed_data=data,
        save_anim=save_anim,
        save_format=save_format,
        filename=filename,
        show=False,
    )


def default_filename(system: str, params: dict[str, Any]) -> str:
    """Mirror each module's established CLI naming convention."""
    if system == "double":
        return (
            f"time_series_m1_{params['m1']}_m2_{params['m2']}_k_{params['k']}_L_{params['L']}_"
            f"theta1_{params['theta_1_init']}_theta2_{params['theta_2_init']}"
        )
    if system == "triple":
        return (
            f"triple_pendulum_m1_{params['m1']}_m2_{params['m2']}_m3_{params['m3']}_"
            f"k1_{params['k1']}_k2_{params['k2']}_L_{params['L']}_"
            f"theta1_{params['theta_1_init']}_theta2_{params['theta_2_init']}_theta3_{params['theta_3_init']}"
        )
    return (
        f"mass_spring_m1={params['m1']}_m2={params['m2']}_k1={params['k1']}_"
        f"k2={params['k2']}_k3={params['k3']}"
    )


def schematic(system: str) -> None:
    if system == "double":
        fig, ax = draw_coupled_pendulum(figsize=(4, 2.8))
    elif system == "triple":
        fig, ax = draw_three_coupled_pendulum(figsize=(4, 3))
    else:
        fig, ax = draw_coupled_mass_spring_system()
        fig.set_size_inches(6, 3)

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
    plt.close(fig)


def theory(system: str) -> None:
    with st.expander("THEORY · equations, modes, and system layout", expanded=True):
        if system == "double":
            st.markdown(
                "Two pendulum bobs are coupled by a spring. The exact nonlinear dynamics become the familiar in-phase and out-of-phase normal modes in the small-angle limit.\n\n"
                "**Where:**\n"
                "- $\\theta_1, \\theta_2$ are the angular displacements\n"
                "- $m_1, m_2$ are the masses of the pendulum bobs\n"
                "- $k$ is the spring constant of the coupling spring\n"
                "- $L$ is the length of the pendulum string\n"
                "- $g$ is the acceleration due to gravity ($9.81\\text{ m/s}^2$)"
            )
            st.latex(
                r"\ddot{\theta}_1=-\frac{g}{L}\sin\theta_1+\frac{k}{m_1}(\sin\theta_2-\sin\theta_1)\cos\theta_1"
            )
            st.latex(
                r"\ddot{\theta}_2=-\frac{g}{L}\sin\theta_2-\frac{k}{m_2}(\sin\theta_2-\sin\theta_1)\cos\theta_2"
            )
            st.markdown(
                "**Small-Angle Approximation with Identical Parameters ($m_1=m_2=m$):**"
            )
            st.latex(
                r"\omega_1^2=\frac{g}{L} \quad \text{(In-Phase)},\qquad \omega_2^2=\frac{g}{L}+\frac{2k}{m} \quad \text{(Out-of-Phase)}"
            )
        elif system == "triple":
            st.markdown(
                "Three pendulums exchange energy through two springs. The linearized system has three collective modes: sloshing, anti-symmetric motion, and breathing.\n\n"
                "**Where:**\n"
                "- $\\theta_1, \\theta_2, \\theta_3$ are the angular displacements\n"
                "- $m_1, m_2, m_3$ are the masses of the pendulum bobs\n"
                "- $k_1, k_2$ are the spring constants of the coupling springs\n"
                "- $L$ is the length of the pendulum string\n"
                "- $g$ is the acceleration due to gravity ($9.81\\text{ m/s}^2$)"
            )
            st.latex(
                r"\ddot{\theta}_1=-\frac{g}{L}\sin\theta_1+\frac{k_1}{m_1}(\sin\theta_2-\sin\theta_1)\cos\theta_1"
            )
            st.latex(
                r"\ddot{\theta}_2=-\frac{g}{L}\sin\theta_2-\frac{k_1}{m_2}(\sin\theta_2-\sin\theta_1)\cos\theta_2+\frac{k_2}{m_2}(\sin\theta_3-\sin\theta_2)\cos\theta_2"
            )
            st.latex(
                r"\ddot{\theta}_3=-\frac{g}{L}\sin\theta_3-\frac{k_2}{m_3}(\sin\theta_3-\sin\theta_2)\cos\theta_3"
            )
            st.markdown(
                "**Small-Angle Approximation with Identical Parameters ($m_i=m, k_i=k$):**"
            )
            st.latex(
                r"\omega_1^2 = \frac{g}{L} \;\text{(Sloshing)}, \quad \omega_2^2 = \frac{g}{L} + \frac{k}{m} \;\text{(Anti-Sym)}, \quad \omega_3^2 = \frac{g}{L} + \frac{3k}{m} \;\text{(Breathing)}"
            )
        else:
            st.markdown(
                "Two masses sit between fixed walls. Three springs provide restoring forces and the middle spring couples the masses.\n\n"
                "**Where:**\n"
                "- $x_1, x_2$ are the displacements of the masses from equilibrium\n"
                "- $m_1, m_2$ are the masses\n"
                "- $k_1, k_2, k_3$ are the spring constants\n"
                "- $\\ddot{x}$ denotes the second derivative with respect to time (acceleration)"
            )
            st.latex(r"m_1\ddot{x}_1+(k_1+k_2)x_1-k_2x_2=0")
            st.latex(r"m_2\ddot{x}_2+(k_2+k_3)x_2-k_2x_1=0")
            st.markdown("**Identical Parameters ($m_1=m_2=m, k_1=k_2=k_3=k$):**")
            st.latex(
                r"\omega_1^2 = \frac{k}{m} \quad \text{(In-Phase)}, \qquad \omega_2^2 = \frac{3k}{m} \quad \text{(Out-of-Phase)}"
            )
        schematic(system)


def diagnostics(system: str, data: SimulationData) -> None:
    cols = st.columns(len(data.theoretical_frequencies))
    for idx, (col, theory_value, estimate, label) in enumerate(
        zip(
            cols,
            data.theoretical_frequencies,
            data.estimated_frequencies,
            SYSTEMS[system]["frequencies"],
        )
    ):
        delta = estimate - theory_value
        with col:
            st.metric(
                label,
                f"{theory_value:.4f} Hz",
                delta=f"FFT {estimate:.4f} Hz · {delta:+.4f}",
            )
    st.markdown(
        "<div class='section-kicker'>Energy conservation</div>", unsafe_allow_html=True
    )
    import pandas as pd

    energy = np.asarray(data.energy)
    ke = np.asarray(data.metadata.get("ke", []))
    pe = np.asarray(data.metadata.get("pe", []))
    time = np.asarray(data.time)

    if len(ke) > 0 and len(pe) > 0:
        df_energy = pd.DataFrame(
            {
                "Total (J)": energy,
                "Kinetic (J)": ke,
                "Potential (J)": pe,
            },
            index=time,
        )
        st.line_chart(
            df_energy,
            width="stretch",
            height=320,
            x_label="Time (s)",  # type: ignore
            y_label="Energy (J)",  # type: ignore
        )
    else:
        st.line_chart(
            {"Total mechanical energy (J)": energy},
            width="stretch",
            height=190,
            x_label="Time (s)",  # type: ignore
            y_label="Energy (J)",  # type: ignore
        )

    drift = float(np.ptp(energy) / max(abs(energy[0]), 1e-12) * 100)
    if drift < 0.5:
        st.caption(f"Energy variation: {drift:.3f}% · numerically stable")
    else:
        st.warning(
            f"Energy variation: {drift:.3f}% · consider shorter runs or tighter solver tolerances."
        )

    st.markdown(
        "<div class='section-kicker'>Velocity Evolution</div>", unsafe_allow_html=True
    )

    v1 = np.asarray(data.metadata.get("v1", []))
    v2 = np.asarray(data.metadata.get("v2", []))
    v3 = np.asarray(data.metadata.get("v3", []))

    if system == "triple" and len(v3) > 0:
        df_vel = pd.DataFrame(
            {
                "v1 (rad/s)": v1,
                "v2 (rad/s)": v2,
                "v3 (rad/s)": v3,
            },
            index=time,
        )
        st.line_chart(
            df_vel,
            width="stretch",
            height=360,
            x_label="Time (s)",  # type: ignore
            y_label="Angular Velocity (rad/s)",  # type: ignore
        )
    elif len(v1) > 0 and len(v2) > 0:
        if system == "spring":
            df_vel = pd.DataFrame(
                {
                    "v1 (m/s)": v1,
                    "v2 (m/s)": v2,
                },
                index=time,
            )
            y_label = "Velocity (m/s)"
        else:
            df_vel = pd.DataFrame(
                {
                    "v1 (rad/s)": v1,
                    "v2 (rad/s)": v2,
                },
                index=time,
            )
            y_label = "Angular Velocity (rad/s)"
        st.line_chart(
            df_vel,
            width="stretch",
            height=360,
            x_label="Time (s)",  # type: ignore
            y_label=y_label,  # type: ignore
        )
    else:
        st.caption("Velocity data not available.")


def export_panel(system: str, params: dict[str, Any], data: SimulationData) -> None:
    st.markdown("<div class='section-kicker'>Export</div>", unsafe_allow_html=True)
    enabled = st.checkbox("Save this animation", key=f"export_enable_{system}")
    if not enabled:
        return
    ffmpeg_available = shutil.which("ffmpeg") is not None
    formats = ["GIF"] + (["MP4"] if ffmpeg_available else [])
    if not ffmpeg_available:
        st.info(
            "MP4 is unavailable because ffmpeg was not found on PATH. GIF export remains available."
        )
    fmt_label = st.radio(
        "Format", formats, horizontal=True, key=f"export_format_{system}"
    )
    fmt = fmt_label.lower()
    default_name = default_filename(system, params)
    base = st.text_input(
        "Base filename",
        value=default_name,
        key=f"export_name_{system}",
        help="A safe base filename; the selected extension is added automatically.",
    )
    if st.button("💾 Save Animation", key=f"save_{system}", use_container_width=True):
        stem = Path(base.strip() or default_name).stem
        filename = f"{stem}.{fmt}"
        try:
            with st.spinner(f"Rendering {fmt.upper()} export..."):
                create_animation(
                    system,
                    params,
                    data,
                    save_anim=True,
                    save_format=fmt,
                    filename=filename,
                )
            path = SAVE_DIR / filename
            if not path.exists():
                raise RuntimeError(
                    "The animation writer did not create the expected file."
                )
            payload = path.read_bytes()
            st.session_state[f"download_{system}"] = (filename, payload, fmt)
            st.success(f"Saved {filename}")
        except Exception as exc:
            st.error(f"Could not save the animation: {exc}")
    download = st.session_state.get(f"download_{system}")
    if download:
        filename, payload, fmt = download
        st.download_button(
            "Download file",
            payload,
            file_name=filename,
            mime="video/mp4" if fmt == "mp4" else "image/gif",
            key=f"download_button_{system}",
            use_container_width=True,
        )


def simulation_section(system: str, params: dict[str, Any]) -> None:
    st.markdown(
        "<div class='section-kicker'>SIMULATION · run, inspect, export</div>",
        unsafe_allow_html=True,
    )
    run_key = f"run_{system}"
    if st.button(
        "▶ Run Simulation", key=run_key, type="primary", use_container_width=False
    ):
        try:
            start_time = time.time()
            with st.spinner("Solving equations of motion...", show_time=True):  # type: ignore
                items = tuple(
                    sorted(
                        (key, value)
                        for key, value in params.items()
                        if key != "trace_length"
                    )
                )
                data = cached_data(system, items)
                anim = create_animation(system, params, data)

            solve_time = time.time() - start_time
            st.success(f"Equations solved in {solve_time:.2f} seconds!")

            progress_bar = st.progress(0.0, text="Generating animation frames...")

            def progress_cb(current_frame, total_frames):
                if total_frames:
                    progress = max(0.0, min(1.0, current_frame / total_frames))
                    progress_bar.progress(
                        progress,
                        text=f"Generating animation frames... ({current_frame}/{total_frames} frames)",
                    )
                else:
                    progress_bar.progress(
                        0.0,
                        text=f"Generating animation frames... (frame {current_frame})",
                    )

            fps = params.get("fps", 30)
            default_mode = "loop" if getattr(anim, "_repeat", False) else "once"

            anim_start_time = time.time()
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir, "temp.html")
                writer = HTMLWriter(
                    fps=fps, embed_frames=True, default_mode=default_mode
                )
                anim.save(str(path), writer=writer, progress_callback=progress_cb)
                html = path.read_text(encoding="utf-8")

            anim_time = time.time() - anim_start_time
            progress_bar.empty()
            st.success(f"Animation generated in {anim_time:.2f} seconds!")

            fig = getattr(anim, "_fig", None)
            if fig is not None:
                plt.close(fig)
            st.session_state[f"run_result_{system}"] = {
                "data": data,
                "html": html,
                "params": params,
            }
        except Exception as exc:
            st.error(f"Simulation could not run: {exc}")
    result = st.session_state.get(f"run_result_{system}")
    if result is None:
        st.markdown(
            "<div class='empty-state'><div><div class='empty-orbit'>◌  ◌  ◌</div><div>Adjust the controls, then run the model to reveal its motion.</div></div></div>",
            unsafe_allow_html=True,
        )
        return
    components.html(result["html"], width="100%", height=900, scrolling=False)
    diagnostics(system, result["data"])
    export_panel(system, result["params"], result["data"])


def render_system(system: str, params: dict[str, Any]) -> None:
    st.markdown(
        f"<h2 style='color:{SYSTEMS[system]['accent']}; margin-bottom:.15rem'>{SYSTEMS[system]['short']}</h2>",
        unsafe_allow_html=True,
    )
    st.caption(
        {
            "double": "Two nonlinear pendulums coupled by a spring.",
            "triple": "Three pendulums coupled by two springs.",
            "spring": "Two masses exchanging energy through three springs fixed between walls.",
        }[system]
    )
    theory(system)
    simulation_section(system, params)


def main() -> None:
    inject_theme()
    params_by_system = sidebar_controls()
    st.markdown(
        "<div class='hero'><div class='eyebrow'>NONLINEAR DYNAMICS · COUPLED OSCILLATIONS</div><h1>Coupled Oscillators Studio</h1><p>Explore normal modes, energy transfer, and frequency fingerprints across three coupled mechanical systems.</p><div class='spectrum'></div></div>",
        unsafe_allow_html=True,
    )
    tabs = st.tabs(
        [
            SYSTEMS["double"]["title"],
            SYSTEMS["triple"]["title"],
            SYSTEMS["spring"]["title"],
        ]
    )
    for tab, system in zip(tabs, ("double", "triple", "spring")):
        with tab:
            render_system(system, params_by_system[system])

    st.markdown(
        "<div class='footer'><span>Built with Streamlit · Matplotlib · SciPy</span><span>Source Code available at: "
        "<a href='https://github.com/PuspenduPH/coupled-oscillations-streamlit-app' target='_blank'>https://github.com/PuspenduPH/coupled-oscillations-streamlit-app</a></span></div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
