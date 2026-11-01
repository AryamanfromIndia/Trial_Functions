# Variational Method for the Hydrogen Atom

An interactive [Streamlit](https://streamlit.io) app that tests trial wavefunctions of the form `exp(−(αr)^p)` against the exact hydrogen orbitals using the **variational principle**. Choose an orbital, choose the exponent `p`, drag the variational parameter `α`, and see how the trial energy and wavefunction compare with the exact solution.

<!-- Add a screenshot: ![App screenshot](screenshot.png) -->
<!-- Live demo: https://your-app-name.streamlit.app -->

## Features

- **Orbitals up to n = 9:** pick `n` and `l` (`0 ≤ l ≤ n−1`) from dropdowns.
- **Choice of trial function:** radial part `R(r) = N · r^l · exp(−(αr)^p)` with `p` ∈ {0.5, 1, 1.5, 2, 3}.
- **Energy comparison:** the trial energy `E[φ] = ⟨φ|H|φ⟩ / ⟨φ|φ⟩` against the exact `E_n = −1/(2n²)`, in Hartree and eV.
- **Variational parameter slider:** `α` starts at its optimal value for the chosen orbital and `p`; move it to see the energy change, or press *Reset to optimal α*.
- **Wavefunction plot:** trial and exact curves on the same axes, switchable between `R(r)` and the radial probability `r²R²`.
- **Most probable radius:** `r_max` (the peak of `r²R²`) is marked on the plot for both functions and shown as numbers.
- **Energy vs α curve:** shows the minimum, the current `α`, and the exact energy as a lower bound.
- **Summary table:** best `α` and energy for every `p` at the chosen `l`.

## Installation and usage

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
streamlit run app.py
```

Requires Python 3.9+ and the packages in `requirements.txt` (`streamlit`, `numpy`, `scipy`, `plotly`).

### Deploying on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt` and `README.md` to a GitHub repository.
2. On <https://share.streamlit.io>, choose **New app**, select the repository, and set the main file to `app.py`.

## Method

Atomic units are used throughout. With `u(r) = r R(r)` the radial energy functional is

```
E[u] = ∫ [ ½ u'² + ( l(l+1)/(2r²) − 1/r ) u² ] dr  /  ∫ u² dr
```

- The integrals are evaluated with Simpson's rule on a 16 000-point grid out to r = 400 a₀, including the r = 0 endpoint (which contributes ½ for s states).
- The optimal `α` is found by a log-spaced scan followed by bounded one-dimensional minimisation.
- Exact orbitals use associated Laguerre polynomials, and both trial and exact radial functions are normalised numerically.

## Example results (1s orbital)

| p | Optimal α | E[φ] (Ha) | Exact E (Ha) |
|---|-----------|-----------|--------------|
| 0.5 | 3.76 | −0.3855 | −0.5 |
| 1 | 1.000 | −0.5000 | −0.5 |
| 1.5 | 0.665 | −0.4711 | −0.5 |
| 2 (Gaussian) | 0.532 | −0.4244 | −0.5 |
| 3 | 0.401 | −0.3422 | −0.5 |

## Physics notes

- **Variational bound.** `E[φ]` never falls below the lowest energy of its `l` channel, `−1/(2(l+1)²)`.
- **Exact cases.** For `n = l+1` (1s, 2p, 3d, …), `p = 1` with `α = 1/n` reproduces the exact orbital and energy.
- **Nodal states.** For `n > l+1` (2s, 3s, 3p, …) the exact orbital has radial nodes that a node-free trial function cannot reproduce, so a gap above `E_n` remains. Polynomial prefactors or a sum of several terms would reduce it.
- **No m input.** The energy and radial functions do not depend on `m`, because the angular part `Y_lm` is the same for trial and exact functions.

## Possible extensions

- Add polynomial prefactors or linear combinations of trial functions to describe radial nodes.
- Extend to a helium atom with electron–electron repulsion.
- Plot the full 3D orbital including the angular part.

## Project structure

| File | Purpose |
|------|---------|
| `app.py` | Streamlit application (numerics and interface) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## Author

*Your name*, with the project link or contact details if you like.
