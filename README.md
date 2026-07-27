# 🌌 Coupled Oscillators Studio

A sleek, interactive physics simulation environment built with Python and Streamlit. This application numerically solves the equations of motion for complex coupled oscillator systems, visualizes their dynamics in real-time, and analyzes their energy transfer and normal modes.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)



> 🌐 **Live Demo:** Explore the web app in your browser at [coupled-oscillations-streamlit-app](https://coupled-oscillations-app-app-7wpyffmaklhjjozxnkjszf.streamlit.app/)


## ✨ Features

- **Three Physical Systems:**
  - **Double Pendulum:** Two pendulums coupled by a spring, showcasing in-phase and out-of-phase normal modes.
  - **Triple Pendulum:** Three pendulums coupled by two springs, demonstrating complex sloshing, anti-symmetric, and breathing modes.
  - **Mass-Spring System:** Two masses interacting through three springs fixed between walls.
- **Deep Physics Analysis:**
  - Accurate numerical integration using `scipy.integrate.solve_ivp` (RK45).
  - Fast Fourier Transform (FFT) analysis to extract exact normal mode frequencies from the simulated time-series data.
  - Real-time comparison between theoretical (small-angle) eigenvalues and empirical frequencies.
- **Beautiful Visualizations:**
  - Dark, neon-themed Matplotlib animations.
  - Comprehensive diagnostic dashboards: Time-domain plots, Phase Space ($\dot{x}$ vs $x$), and Energy evolution (Kinetic vs Potential).
- **Interactive UI:**
  - Control physical parameters ($m$, $k$, $L$, $\theta$) and initial conditions via an intuitive Streamlit sidebar.
  - Special-case presets to instantly trigger specific physical phenomena like "Beats".
  - Export generated animations directly to GIF or MP4.

## 🚀 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/PuspenduPH/coupled-oscillations-streamlit-app.git
   cd coupled-oscillations-streamlit-app
   ```

2. **Install dependencies**
   Ensure you have Python 3.9+ installed. It is recommended to use a virtual environment.
   ```bash
   pip install numpy scipy matplotlib streamlit
   ```
   Or you can install all the dependencies from the `requirements.txt` file.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🎮 Usage Guide

1. **Select a System**: Use the tabs at the top of the screen to choose between the Double Pendulum, Triple Pendulum, or Mass-Spring system.
2. **Read the Theory**: Expand the Theory section to view the equations of motion, mathematical approximations, and a clean schematic of the selected system.
3. **Configure Parameters**: Adjust masses, spring constants, initial displacements, and animation settings in the left sidebar.
4. **Run Simulation**: Click the primary "Run Simulation" button. The app will:
   - Solve the differential equations.
   - Analyze the frequency spectrum.
   - Render the animation frames.
5. **Inspect the Output**: Watch the animation, review the phase space trajectories, and study the energy conservation graphs.

## 🛠️ Project Structure

- `app.py`: The main Streamlit orchestration layer. Handles UI, state, caching, and layout routing.
- `double_pendulum.py`: Physics engine and Matplotlib rendering logic for the coupled double pendulum.
- `triple_pendulum.py`: Physics engine and Matplotlib rendering logic for the coupled triple pendulum.
- `mass_spring.py`: Physics engine and Matplotlib rendering logic for the horizontal mass-spring system.
- `utils.py`: Shared utilities including the dark theme color palette, geometry drawing helpers, data schemas, and frequency formatters.

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
