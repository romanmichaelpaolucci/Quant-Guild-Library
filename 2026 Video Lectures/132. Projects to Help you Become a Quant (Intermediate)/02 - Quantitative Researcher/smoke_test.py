"""
smoke_test.py
=============

Headless correctness checks for the PDE Solver. No web server is started.

The tests verify the central claim of the project -- that finite differences,
Monte-Carlo and closed-form pricing all agree:

1. Vanilla European call: FD ~= Black-Scholes AND MC ~= Black-Scholes (within CI).
2. Vanilla European put:  FD ~= Black-Scholes.
3. Digital cash-or-nothing call: FD ~= closed form.
4. Barrier up-and-out call: solver runs, FD ~= analytic, and price < vanilla.
5. American put: FD (PSOR) >= European put (early-exercise premium) and FD ~= MC (LSM).
6. Flask app boots and GET '/' returns HTTP 200.

Run:  python smoke_test.py
Exit code 0 => all checks passed.
"""

from __future__ import annotations

import sys

import analytics
import fd_solver
import mc_pricer

PASS, FAIL = "PASS", "FAIL"
_results = []


def check(name: str, condition: bool, detail: str = ""):
    tag = PASS if condition else FAIL
    _results.append(condition)
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    return condition


def main() -> int:
    print("=" * 72)
    print("PDE Solver -- headless smoke test")
    print("=" * 72)

    base = {"S0": 100.0, "K": 100.0, "r": 0.05, "sigma": 0.2, "T": 1.0}

    # --- 1. Vanilla call: FD ~= BS and MC ~= BS -----------------------------
    print("\n[1] Vanilla European CALL -- FD vs MC vs Black-Scholes")
    p = dict(base, option_type="call")
    bs = analytics.bs_price(base["S0"], base["K"], base["r"], base["sigma"],
                            base["T"], "call")
    fd = fd_solver.solve("vanilla", p, M=300, N=300, scheme="crank-nicolson")
    mc = mc_pricer.price("vanilla", p, n_paths=200_000, seed=7)
    print(f"      BS={bs:.4f}  FD={fd.price:.4f}  "
          f"MC={mc.price:.4f} +/- {mc.std_error:.4f} "
          f"[{mc.ci_low:.4f}, {mc.ci_high:.4f}]")
    check("FD ~= Black-Scholes (|err| < 5e-3)", abs(fd.price - bs) < 5e-3,
          f"err={abs(fd.price - bs):.2e}")
    check("MC ~= Black-Scholes (BS within MC 95% CI)",
          mc.ci_low <= bs <= mc.ci_high)
    check("FD price within MC 95% CI", mc.ci_low <= fd.price <= mc.ci_high)

    # --- 2. Vanilla put ---------------------------------------------------
    print("\n[2] Vanilla European PUT -- FD vs Black-Scholes")
    pput = dict(base, option_type="put")
    bs_put = analytics.bs_price(base["S0"], base["K"], base["r"], base["sigma"],
                                base["T"], "put")
    fd_put = fd_solver.solve("vanilla", pput, M=300, N=300)
    print(f"      BS_put={bs_put:.4f}  FD_put={fd_put.price:.4f}")
    check("FD put ~= Black-Scholes (|err| < 5e-3)",
          abs(fd_put.price - bs_put) < 5e-3, f"err={abs(fd_put.price - bs_put):.2e}")

    # --- 3. Digital cash-or-nothing --------------------------------------
    print("\n[3] Digital cash-or-nothing CALL -- FD vs closed form")
    pdig = dict(base, option_type="call", cash=1.0)
    an_dig = analytics.digital_cash_or_nothing(base["S0"], base["K"], base["r"],
                                               base["sigma"], base["T"], "call", 1.0)
    fd_dig = fd_solver.solve("digital", pdig, M=400, N=400)
    mc_dig = mc_pricer.price("digital", pdig, n_paths=200_000, seed=11)
    print(f"      analytic={an_dig:.4f}  FD={fd_dig.price:.4f}  MC={mc_dig.price:.4f}")
    # Digital terminal condition is discontinuous => allow a looser FD tolerance.
    check("FD digital ~= analytic (|err| < 5e-3)", abs(fd_dig.price - an_dig) < 5e-3,
          f"err={abs(fd_dig.price - an_dig):.2e}")
    check("MC digital ~= analytic (within CI)",
          mc_dig.ci_low <= an_dig <= mc_dig.ci_high)

    # --- 4. Barrier up-and-out call --------------------------------------
    print("\n[4] Barrier UP-AND-OUT CALL -- sane price, < vanilla, FD ~= analytic")
    pbar = dict(base, option_type="call", barrier=130.0, barrier_type="up-and-out")
    an_bar = analytics.barrier_price(base["S0"], base["K"], 130.0, base["r"],
                                     base["sigma"], base["T"], "call", "up-and-out")
    fd_bar = fd_solver.solve("barrier", pbar, M=400, N=400)
    mc_bar = mc_pricer.price("barrier", pbar, n_paths=200_000, n_steps=400, seed=13)
    print(f"      analytic={an_bar:.4f}  FD={fd_bar.price:.4f}  "
          f"MC={mc_bar.price:.4f} [{mc_bar.ci_low:.4f}, {mc_bar.ci_high:.4f}]  "
          f"vanilla={bs:.4f}")
    check("Barrier solver runs and returns finite price",
          fd_bar.price == fd_bar.price and 0 <= fd_bar.price < 1e6)
    check("Barrier FD price < vanilla FD price", fd_bar.price < bs)
    check("Barrier FD ~= analytic (|err| < 0.05)", abs(fd_bar.price - an_bar) < 0.05,
          f"err={abs(fd_bar.price - an_bar):.2e}")

    # --- 5. American put --------------------------------------------------
    print("\n[5] American PUT -- FD (PSOR) >= European put, FD ~= MC (LSM)")
    pam = dict(base, option_type="put")
    fd_am = fd_solver.solve("american", pam, M=300, N=300, scheme="implicit")
    mc_am = mc_pricer.price("american", pam, n_paths=80_000, n_steps=50, seed=17)
    print(f"      European put={bs_put:.4f}  American FD={fd_am.price:.4f}  "
          f"American MC(LSM)={mc_am.price:.4f} [{mc_am.ci_low:.4f}, {mc_am.ci_high:.4f}]")
    check("American put FD >= European put (early-exercise premium)",
          fd_am.price >= bs_put - 1e-6)
    check("American put FD ~= MC LSM (|err| < 0.10)",
          abs(fd_am.price - mc_am.price) < 0.10, f"err={abs(fd_am.price - mc_am.price):.2e}")

    # --- 6. Flask boots ---------------------------------------------------
    print("\n[6] Flask app -- test client GET '/'")
    import app as flask_app
    client = flask_app.app.test_client()
    resp = client.get("/")
    check("GET / returns 200", resp.status_code == 200,
          f"status={resp.status_code}")
    # Also exercise the pricing endpoint end-to-end.
    api = client.post("/api/price", json=dict(base, framework="vanilla",
                                              option_type="call", M=120, N=120,
                                              n_paths=20000))
    api_ok = api.status_code == 200 and api.get_json().get("ok") is True
    check("POST /api/price returns ok", api_ok, f"status={api.status_code}")

    print("\n" + "=" * 72)
    total = len(_results); passed = sum(_results)
    print(f"RESULT: {passed}/{total} checks passed")
    print("=" * 72)
    return 0 if all(_results) else 1


if __name__ == "__main__":
    sys.exit(main())

