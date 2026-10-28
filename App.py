"""Variational method for the hydrogen atom: Streamlit app.

Trial radial function  R(r) = N r^l exp(-(alpha r)^p)   (atomic units)
compared with the exact hydrogen orbital |n, l>.
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize_scalar
from scipy.special import eval_genlaguerre

st.set_page_config(page_title="Hydrogen Variational Method", page_icon="⚛️", layout="wide")

# ---------- numerics ----------
RMAX, N = 400.0, 16000            # Simpson grid (N even)
H = RMAX / N
R = np.arange(1, N + 1) * H
W = np.where(np.arange(1, N + 1) % 2 == 1, 4.0, 2.0)
W[-1] = 1.0
P_LIST = [0.5, 1.0, 1.5, 2.0, 3.0]
L_NAMES = "spdfghikl"   # l = 0..8


def trial_energy(alpha, p, l):
    """E[phi] = <phi|H|phi>/<phi|phi> using u(r) = r R(r) and the radial Hamiltonian."""
    u2 = np.exp(-2 * (alpha * R) ** p + 2 * (l + 1) * np.log(R))
    du = (l + 1) / R - p * alpha**p * R ** (p - 1)          # u'/u
    num = np.sum(W * u2 * (0.5 * du**2 + l * (l + 1) / (2 * R**2) - 1 / R))
    if l == 0:
        num += 0.5          # r = 0 endpoint: kinetic integrand is 1/2 for s states
    return num / np.sum(W * u2)


@st.cache_data(show_spinner=False)
def optimise(p, l):
    scan = np.log(np.geomspace(0.03, 8, 40))
    x0 = scan[np.argmin([trial_energy(np.exp(x), p, l) for x in scan])]
    step = scan[1] - scan[0]
    res = minimize_scalar(lambda x: trial_energy(np.exp(x), p, l),
                          bounds=(x0 - step, x0 + step), method="bounded",
                          options={"xatol": 1e-9})
    return float(np.exp(res.x)), float(res.fun)


@st.cache_data(show_spinner=False)
def energy_curve(p, l):
    al = np.geomspace(0.02, 8.0, 150)
    return al, np.array([trial_energy(x, p, l) for x in al])


def exact_radial(r, n, l):
    x = 2 * r / n
    return x**l * np.exp(-x / 2) * eval_genlaguerre(n - l - 1, 2 * l + 1, x)


def normalise(f):
    """Return c such that  integral (c f)^2 r^2 dr = 1."""
    return 1 / np.sqrt(np.sum(W * (f(R) * R) ** 2) * H / 3)


# ---------- sidebar ----------
st.sidebar.header("Orbital")
n = st.sidebar.selectbox("n", list(range(1, 10)))
l = st.sidebar.selectbox("l", list(range(n)))
st.sidebar.header("Trial function")
p = st.sidebar.selectbox("Exponent p in exp(-(αr)^p)", P_LIST, index=1)
alpha_opt, E_opt = optimise(p, l)
A_MIN, A_MAX = 0.02, 8.0
alpha_start = float(np.clip(round(alpha_opt, 3), A_MIN, A_MAX))
if st.session_state.get("orbital_key") != (p, l):      # new p or l -> start at the optimum
    st.session_state["orbital_key"] = (p, l)
    st.session_state["alpha"] = alpha_start


def reset_alpha():
    st.session_state["alpha"] = alpha_start


alpha = st.sidebar.slider("Variational parameter α", A_MIN, A_MAX, step=0.001, key="alpha", format="%.3f")
st.sidebar.button("Reset to optimal α", on_click=reset_alpha)
st.sidebar.caption(f"Optimal α = {alpha_opt:.4f}  (E = {E_opt:.5f} Ha)")
quantity = st.sidebar.radio("Plot", ["Radial function R(r)", "Radial probability r²R²"])

# ---------- calculation ----------
E_trial = trial_energy(alpha, p, l)
E_exact = -1 / (2 * n**2)
diff = E_trial - E_exact

st.title("⚛️ Variational method for the hydrogen atom")
st.caption(f"Trial: R(r) = N·r^{l}·exp(−(αr)^{p:g}) vs exact {n}{L_NAMES[l]} orbital. Atomic units (Hartree).")

c = st.columns(4)
c[0].metric("Trial energy E[φ]", f"{E_trial:.5f} Ha", f"{E_trial*27.2114:.3f} eV", delta_color="off")
c[1].metric("Exact energy −1/(2n²)", f"{E_exact:.5f} Ha", f"{E_exact*27.2114:.3f} eV", delta_color="off")
c[2].metric("E[φ] − E_exact", f"{diff:.2e} Ha", f"{diff/abs(E_exact)*100:.2f} % of |E|", delta_color="off")
c[3].metric("α (slider)", f"{alpha:.4f}", f"optimal {alpha_opt:.4f}", delta_color="off")

# ---------- plot ----------
rmax = min(RMAX, max(8, 2.5 * n * n + 8 + (4 if l else 0)))
r = np.linspace(1e-6, rmax, 800)
ct = normalise(lambda x: x**l * np.exp(-(alpha * x) ** p))
ce = normalise(lambda x: exact_radial(x, n, l))
if exact_radial(1e-3, n, l) < 0:
    ce = -ce
yt = ct * r**l * np.exp(-(alpha * r) ** p)
ye = ce * exact_radial(r, n, l)
# radius of maximum radial probability P(r) = r^2 R^2 (global maximum, fine grid)
r_max_trial = R[np.argmax((ct * R**l * np.exp(-(alpha * R) ** p) * R) ** 2)]
r_max_exact = R[np.argmax((ce * exact_radial(R, n, l) * R) ** 2)]
ylab = "R(r)"
if quantity.startswith("Radial prob"):
    yt, ye, ylab = (yt * r) ** 2, (ye * r) ** 2, "r² R(r)²"

COL_E, COL_T = "#1f77b4", "#ff7f0e"
fig = go.Figure()
fig.add_scatter(x=r, y=ye, name=f"Exact {n}{L_NAMES[l]}", line=dict(width=3, color=COL_E))
fig.add_scatter(x=r, y=yt, name=f"Trial (p={p:g}, α={alpha:.3f})", line=dict(width=3, dash="dash", color=COL_T))
ytop = 1.05 * max(np.max(ye), np.max(yt))
ybot = min(0.0, 1.05 * min(np.min(ye), np.min(yt)))
for rm, col, lab in [(r_max_exact, COL_E, "exact"), (r_max_trial, COL_T, "trial")]:
    fig.add_scatter(x=[rm, rm], y=[ybot, ytop], mode="lines",
                    line=dict(color=col, width=1.5, dash="dot"),
                    name=f"r_max ({lab}) = {rm:.3f}")
fig.update_layout(xaxis_title="r (Bohr radii)", yaxis_title=ylab, height=470,
                  legend=dict(orientation="h", y=1.08), margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

# ---------- E(alpha) curve ----------
al, Ec = energy_curve(p, l)
Emn = min(Ec.min(), E_exact)
fig2 = go.Figure()
fig2.add_scatter(x=al, y=Ec, name="E[φ](α)", line=dict(width=3, color=COL_T))
fig2.add_hline(y=E_exact, line=dict(color=COL_E, dash="dash"), annotation_text=f"exact E_n = {E_exact:.5f}")
fig2.add_scatter(x=[alpha_opt], y=[E_opt], mode="markers", name="optimal α",
                 marker=dict(size=11, symbol="diamond", color="green"))
fig2.add_scatter(x=[alpha], y=[E_trial], mode="markers", name="current α",
                 marker=dict(size=12, color="red"))
fig2.update_layout(xaxis_title="α", yaxis_title="E[φ] (Ha)", xaxis_type="log", height=340,
                   yaxis=dict(range=[Emn - 0.15 * abs(Emn), Emn + 2.5 * abs(Emn)]),
                   legend=dict(orientation="h", y=1.15), margin=dict(l=10, r=10, t=30, b=10))
st.subheader("Energy as a function of α")
st.plotly_chart(fig2, use_container_width=True)

c2 = st.columns(3)
c2[0].metric("r_max exact (peak of r²R²)", f"{r_max_exact:.3f} a₀")
c2[1].metric("r_max trial", f"{r_max_trial:.3f} a₀")
c2[2].metric("Difference", f"{r_max_trial - r_max_exact:+.3f} a₀")

# ---------- table over p ----------
st.subheader(f"Best α and energy for each exponent (l = {l})")
rows = []
for q in P_LIST:
    a_q, e_q = optimise(q, l)
    rows.append({"p": q, "α (optimal)": round(a_q, 4), "E[φ] (Ha)": round(e_q, 5),
                 "E[φ] − E_exact (Ha)": f"{e_q - E_exact:.2e}"})
st.dataframe(rows, hide_index=True, use_container_width=True)

with st.expander("Physics notes"):
    st.markdown(
        "- The energy does not depend on the magnetic quantum number m, so the app only asks for n and l.\n"
        "- The variational principle guarantees E[φ] ≥ the lowest energy of that *l* channel, −1/(2(l+1)²).\n"
        "- **r_max** is the position of the global maximum of the radial probability r²R²; the dotted vertical lines mark it for both functions on either plot. For 1s it is exactly 1 a₀ (and for n = l+1 it is n² a₀).\n"
        "- For **n = l+1** (1s, 2p, 3d, …) p = 1 with α = 1/n reproduces the exact orbital.\n"
        "- For **n > l+1** the exact orbital has radial nodes; a node-free trial function cannot reproduce them, "
        "so a gap above E_n remains.")