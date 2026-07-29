# Pricing Library — Rough Volatility, Markovian Lifting & Path Signatures

*Quant Researcher · Advanced · Flask + Plotly + LaTeX*

This project is a **teaching dashboard** for three ideas at the frontier of
derivatives research. It does **not** try to be a production quant library.
Instead it picks the three most important modern extensions of classical option
pricing and *illustrates each one honestly* with a small, correct, live
numerical experiment and a rendered-LaTeX explanation:

1. **Rough volatility** — volatility is rougher than Brownian motion, and this
   single fact reproduces the steep short-maturity skew of index options.
2. **Markovian lifting** — how to make a non-Markovian rough model tractable by
   approximating its fractional kernel with a sum of exponentials (OU factors).
3. **Path signatures & model-free pricing** — the signature as a universal
   feature map, and pricing as a linear functional of the *expected signature*.

Everything runs offline with only `flask`, `numpy`, `scipy` and `plotly`.

---

## Quick start

```powershell
cd "2026 Video Lectures/130. Projects to Help you Become a Quant (by Role and Level)/02 - Quant Researcher/03 - Advanced - Pricing Library"
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:5004
```

You can also run each module's self-check headlessly:

```powershell
python rough_vol.py          # variance identity + a rough Bergomi smile
python markovian_lifting.py  # kernel error falls as the factor count grows
python signatures.py         # signature dimension + Asian-option regression
```

---

## 1 · Rough volatility (`rough_vol.py`)

Gatheral, Jaisson & Rosenbaum (*Volatility is rough*, 2018) measured realised
volatility across thousands of assets and found log-vol behaves like a
fractional Brownian motion with Hurst exponent **H ≈ 0.1** — far below the
`H = 1/2` of Brownian motion. Bayer, Friz & Gatheral (2016) turned this into the
**rough Bergomi** model. With a Volterra process

$$Y_t=\int_0^t K(t-s)\,dW_s,\qquad K(u)=\sqrt{2H}\,u^{H-1/2},$$

the (log-normal Bergomi) variance and spot are

$$v_t=\xi_0\exp\!\Big(\eta Y_t-\tfrac12\eta^2 t^{2H}\Big),\qquad
\frac{dS_t}{S_t}=\sqrt{v_t}\,\big(\rho\,dW_t+\sqrt{1-\rho^2}\,dB_t\big).$$

The normalisation `sqrt(2H)` gives `Var(Y_t) = t^{2H}` exactly (the code proves
this by comparing the closed-form covariance to `t^{2H}`). We simulate the pair
`(W, Y)` **jointly and exactly on the grid** by taking a Cholesky factor of
their closed-form covariance matrix — the `W–W`, `Y–Y` (via a Gauss
hypergeometric `2F1`) and `Y–W` blocks are all analytic. This avoids the
discretisation bias of naive Euler schemes near the singular kernel. Option
prices come from Monte-Carlo and are inverted to Black–Scholes implied vols.

**Why markets are rough — what to look for in the dashboard.** The ATM
volatility skew of rough Bergomi explodes as a power law `|skew| ~ T^{H-1/2}` as
maturity `T → 0`. Classical diffusion models have a *bounded* short-maturity
skew and cannot fit short-dated options; roughness fixes this with one
parameter. Lower the `H` slider and watch both the vol paths get rougher and the
log-log skew plot steepen toward the dashed `T^{H-1/2}` reference line.

## 2 · Markovian lifting (`markovian_lifting.py`)

The rough kernel has infinite memory, so `Y` is **not Markovian**: expensive to
simulate, no PDE. The lift exploits the completely monotone representation

$$t^{H-1/2}=\int_0^\infty e^{-xt}\,\mu(dx),\qquad
\mu(dx)=\frac{x^{-H-1/2}}{\Gamma(1/2-H)}\,dx,$$

which follows from `Γ(s) t^{-s} = ∫₀^∞ x^{s-1} e^{-xt} dx` with `s = 1/2 - H`.
Discretising the *measure* `μ ≈ Σ wⱼ δ_{xⱼ}` gives a **sum of exponentials**
`K_N(t) = Σ wⱼ e^{-xⱼ t}`, and each mode

$$U^j_t=\int_0^t e^{-x_j(t-s)}\,dW_s,\qquad dU^j=-x_j\,U^j\,dt+dW$$

is a mean-reverting **Ornstein–Uhlenbeck factor**. The lifted process
`Y^N = Σ wⱼ Uʲ` lives on the finite Markov state `(U¹,…,Uᴺ)` and converges to
the rough process as `N → ∞` (Abi Jaber & El Euch 2019; Cuchiero & Teichmann
2019). We choose `(wⱼ, xⱼ)` by the transparent **mass / centroid quadrature** of
`μ` on a geometric grid (Abi Jaber, *Lifting the Heston model*, 2019). The
dashboard shows the true kernel vs. its `N`-factor approximation (with a few
individual OU modes), the relative `L²` error falling as `N` grows, and a lifted
sample path tracking the true Volterra path on the *same* Brownian motion.

## 3 · Path signatures & model-free pricing (`signatures.py`)

The **signature** of a path `X:[0,T]→ℝ^d` is the graded set of iterated
integrals

$$S(X)=\Big(1,\ \int dX^{i},\ \iint dX^{i}dX^{j},\ \iiint dX^{i}dX^{j}dX^{k},\dots\Big),$$

truncated at level `M` to a vector of dimension `(d^{M+1} − 1)/(d − 1)`. Two
facts (Lyons; Kidger & Lyons) make it a powerful feature map:

* **Universality** — any continuous path functional is approximately *linear*
  in a high-enough signature: `F(X) ≈ ⟨ℓ, S(X)⟩`.
* **Chen's identity** — the signature of a concatenation is the tensor product
  of signatures, which is exactly how we compute it: tensor-exponential of each
  straight increment, folded together with a truncated tensor product.

Because the payoff is linear in the signature, pricing **factorises**:

$$\text{price}=\mathbb E[F(X)]\approx\big\langle\ell,\ \mathbb E[S(X)]\big\rangle,$$

into a payoff object `ℓ` (learned once) and the **expected signature**
`E[S(X)]` — a property of the market/measure, not of a specific SDE. This is the
**model-free pricing** philosophy: given the expected signature, the same `ℓ`
prices the payoff under any dynamics. The demo *honestly* illustrates this: we
simulate GBM paths, learn `ℓ` for an **arithmetic Asian option** by linear
regression of the discounted payoff on signature features of the time-augmented
path, report in/out-of-sample `R²`, then price the option as the linear
functional `⟨ℓ, E[S(X)]⟩` and compare with plain Monte-Carlo (they agree within
MC error). The signature is computed with a **pure-NumPy** tensor-algebra
routine; the optional `iisignature` C library is auto-detected but never
required.

> This is an illustration, not a proof of model-independence: the regression
> *learns* the payoff functional from data, and the level-3 truncation is a
> deliberately modest approximation. The point is to make the factorisation
> `price = ⟨ℓ, E[S(X)]⟩` concrete and checkable.

---

## Files

| File | What it does |
|------|--------------|
| `app.py` | Flask app (port **5004**), three tabbed sections, MathJax + Plotly. |
| `rough_vol.py` | Rough Bergomi via exact joint-Cholesky simulation; implied smile & skew term structure. |
| `markovian_lifting.py` | Sum-of-exponentials approximation of the fractional kernel; lifted OU simulation; error-vs-`N`. |
| `signatures.py` | NumPy truncated signatures (Chen's identity) + signature-regression Asian pricing. |
| `requirements.txt` | `flask`, `numpy`, `scipy`, `plotly` (`iisignature` optional). |

## References

- J. Gatheral, T. Jaisson, M. Rosenbaum. *Volatility is rough.* Quantitative Finance, 2018.
- C. Bayer, P. Friz, J. Gatheral. *Pricing under rough volatility.* Quantitative Finance, 2016.
- M. Bennedsen, A. Lunde, M. Pakkanen. *Hybrid scheme for Brownian semistationary processes.* Finance & Stochastics, 2017.
- E. Abi Jaber, O. El Euch. *Multifactor approximation of rough volatility models.* SIAM J. Financial Math., 2019.
- E. Abi Jaber. *Lifting the Heston model.* Quantitative Finance, 2019.
- C. Cuchiero, J. Teichmann. *Markovian lifts of positive semidefinite affine Volterra-type processes.* Decisions in Economics and Finance, 2019.
- T. Lyons. *Rough paths, signatures and the modelling of functions on streams.* ICM, 2014.
- P. Kidger, T. Lyons. *Signatory / signature methods in machine learning.*
- L. Perez Arribas, C. Salvi, T. Lyons et al. *Signature payoffs* and model-free pricing literature.

Built for the [Quant Guild](https://quantguild.com) by Roman Paolucci.
