"""
app.py
======

Tkinter desktop **Algorithmic Trading System** cockpit.

This is the entry point / view layer. It owns no trading logic itself — it wires
together:

* :mod:`strategies` — the alpha framework (signals, allocation blending, sizing).
* :mod:`ib_client`  — the IBKR connection with transparent DEMO fallback.

UI layout (a real trading "cockpit")
------------------------------------
+--------------------------------------------------------------------------+
| Top bar: title | connection controls | status pill (Connected / DEMO)    |
+----------------------------------------+---------------------------------+
| STRATEGY BOOK (left, wide)             | PORTFOLIO (right)               |
|  - Treeview of strategy rows           |  - Account readouts (Net Liq,   |
|    (ON/OFF, type, symbol, params,      |    Cash, Buying Power, Gross,    |
|     alloc%, live signal, target%)      |    Unrealized/Realized P/L)     |
|  - Add-strategy form (type dropdown,   |  - Positions Treeview           |
|    dynamic params, symbol, alloc)      |                                 |
|  - Toggle / Edit / Remove buttons      |                                 |
+----------------------------------------+---------------------------------+
| Rebalance preview + Send Orders (guarded) | activity log                 |
+--------------------------------------------------------------------------+

Threading
---------
The IBKR message loop / demo price feed run on background threads inside
:class:`~ib_client.TradingBackend`. This view NEVER touches Tk from a worker
thread. Instead it polls, from the Tk main thread via ``root.after``:
  * ``events`` queue (status/log/fill/error messages), and
  * ``backend.snapshot()`` (numeric state) to repaint the tables.

Importing this module does NOT open a window; the Tk root and ``mainloop()`` are
created only when run as ``__main__`` (or via :func:`main`).
"""

from __future__ import annotations

import sys
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from typing import Dict, List

from ib_client import TradingBackend
from strategies import (
    STRATEGY_REGISTRY,
    StrategyAllocation,
    aggregate_target_weights,
    build_order_plans,
    create_strategy,
)

# ---------------------------------------------------------------------------
# Cockpit color palette (dark "trading terminal" aesthetic).
# ---------------------------------------------------------------------------
COL_BG = "#0e1117"        # window background
COL_PANEL = "#161b22"     # panel background
COL_PANEL_2 = "#1c2430"   # alt panel
COL_FG = "#e6edf3"        # primary text
COL_MUTED = "#8b949e"     # secondary text
COL_ACCENT = "#2f81f7"    # accent / buttons
COL_GREEN = "#3fb950"     # positive / ON / connected
COL_RED = "#f85149"       # negative / errors
COL_AMBER = "#d29922"     # demo / warning
COL_GRID = "#30363d"      # borders / gridlines
COL_ROW_OFF = "#485460"   # disabled row text
FONT = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_MONO = ("Consolas", 10)
FONT_TITLE = ("Segoe UI Semibold", 15)
FONT_BIG = ("Consolas", 15, "bold")


class TradingCockpit:
    """The main application window."""

    def __init__(self, root: tk.Tk, symbols: List[str] | None = None,
                 host: str = "127.0.0.1", port: int = 7497, client_id: int = 17):
        self.root = root
        self._alloc_seq = 0
        # tree item id -> StrategyAllocation
        self.rows: Dict[str, StrategyAllocation] = {}

        default_symbols = symbols or ["AAPL", "MSFT", "SPY", "NVDA"]
        self.backend = TradingBackend(default_symbols, host=host, port=port, client_id=client_id)

        self._build_styles()
        self._build_topbar()
        self._build_body()
        self._build_bottom()

        # Seed a couple of example strategies so the cockpit is populated.
        self._seed_examples()

        # Kick off connection (async: tries IBKR, falls back to DEMO).
        self.root.after(200, self._start_backend)
        # Start the two polling loops.
        self.root.after(250, self._drain_events)
        self.root.after(500, self._refresh)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ================= styling ============================================
    def _build_styles(self):
        self.root.title("Algorithmic Trading System — Quant Guild Cockpit")
        self.root.geometry("1360x820")
        self.root.minsize(1180, 720)
        self.root.configure(bg=COL_BG)

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")  # most themeable cross-platform base
        except tk.TclError:
            pass

        style.configure(".", background=COL_BG, foreground=COL_FG, font=FONT)
        style.configure("TFrame", background=COL_BG)
        style.configure("Panel.TFrame", background=COL_PANEL)
        style.configure("TLabel", background=COL_BG, foreground=COL_FG)
        style.configure("Panel.TLabel", background=COL_PANEL, foreground=COL_FG)
        style.configure("Muted.TLabel", background=COL_PANEL, foreground=COL_MUTED)
        style.configure("Title.TLabel", background=COL_BG, foreground=COL_FG, font=FONT_TITLE)
        style.configure("Heading.TLabel", background=COL_PANEL, foreground=COL_ACCENT, font=FONT_BOLD)
        style.configure("Value.TLabel", background=COL_PANEL, foreground=COL_FG, font=FONT_BIG)

        style.configure("TButton", background=COL_PANEL_2, foreground=COL_FG,
                        borderwidth=0, focusthickness=0, padding=(10, 6), font=FONT_BOLD)
        style.map("TButton",
                  background=[("active", COL_ACCENT), ("pressed", COL_ACCENT)],
                  foreground=[("active", "#ffffff")])
        style.configure("Accent.TButton", background=COL_ACCENT, foreground="#ffffff")
        style.map("Accent.TButton", background=[("active", "#4c9bff")])
        style.configure("Danger.TButton", background=COL_RED, foreground="#ffffff")
        style.map("Danger.TButton", background=[("active", "#ff6b62")])

        style.configure("TEntry", fieldbackground=COL_PANEL_2, foreground=COL_FG,
                        insertcolor=COL_FG, borderwidth=1)
        style.configure("TCombobox", fieldbackground=COL_PANEL_2, background=COL_PANEL_2,
                        foreground=COL_FG, arrowcolor=COL_FG)
        style.map("TCombobox", fieldbackground=[("readonly", COL_PANEL_2)],
                  foreground=[("readonly", COL_FG)])
        style.configure("TCheckbutton", background=COL_PANEL, foreground=COL_FG)
        style.map("TCheckbutton", background=[("active", COL_PANEL)])

        # Treeview ("grids").
        style.configure("Treeview", background=COL_PANEL_2, fieldbackground=COL_PANEL_2,
                        foreground=COL_FG, rowheight=26, borderwidth=0, font=FONT)
        style.configure("Treeview.Heading", background=COL_PANEL, foreground=COL_ACCENT,
                        font=FONT_BOLD, borderwidth=0, relief="flat")
        style.map("Treeview.Heading", background=[("active", COL_PANEL)])
        style.map("Treeview", background=[("selected", COL_ACCENT)],
                  foreground=[("selected", "#ffffff")])

    # ================= top bar ============================================
    def _build_topbar(self):
        bar = ttk.Frame(self.root, style="TFrame")
        bar.pack(side=tk.TOP, fill=tk.X, padx=14, pady=(12, 6))

        ttk.Label(bar, text="◈  ALGORITHMIC TRADING SYSTEM", style="Title.TLabel").pack(side=tk.LEFT)

        # Connection status pill (right side).
        self.status_pill = tk.Label(bar, text="  ● STARTING…  ", bg=COL_AMBER, fg="#0e1117",
                                    font=FONT_BOLD, padx=8, pady=4)
        self.status_pill.pack(side=tk.RIGHT)

        # Connection controls.
        conn = ttk.Frame(bar, style="TFrame")
        conn.pack(side=tk.RIGHT, padx=12)
        ttk.Label(conn, text="Host").grid(row=0, column=0, padx=(0, 4))
        self.host_var = tk.StringVar(value=self.backend.host)
        ttk.Entry(conn, textvariable=self.host_var, width=11).grid(row=0, column=1, padx=(0, 8))
        ttk.Label(conn, text="Port").grid(row=0, column=2, padx=(0, 4))
        self.port_var = tk.StringVar(value=str(self.backend.port))
        ttk.Entry(conn, textvariable=self.port_var, width=6).grid(row=0, column=3, padx=(0, 8))
        ttk.Label(conn, text="Client Id").grid(row=0, column=4, padx=(0, 4))
        self.cid_var = tk.StringVar(value=str(self.backend.client_id))
        ttk.Entry(conn, textvariable=self.cid_var, width=5).grid(row=0, column=5, padx=(0, 8))
        ttk.Button(conn, text="Reconnect", command=self._on_reconnect).grid(row=0, column=6)

    # ================= body (strategy book + portfolio) ===================
    def _build_body(self):
        body = ttk.Frame(self.root, style="TFrame")
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=14, pady=6)

        # ---- LEFT: strategy book -----------------------------------------
        left = ttk.Frame(body, style="Panel.TFrame")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        ttk.Label(left, text="STRATEGY BOOK", style="Heading.TLabel").pack(
            anchor="w", padx=12, pady=(10, 4))

        cols = ("state", "type", "symbol", "params", "alloc", "signal", "target")
        headings = {
            "state": "ON/OFF", "type": "Strategy", "symbol": "Symbol",
            "params": "Parameters", "alloc": "Alloc %", "signal": "Signal", "target": "Target %",
        }
        widths = {"state": 70, "type": 130, "symbol": 70, "params": 170,
                  "alloc": 70, "signal": 80, "target": 80}
        tree_wrap = ttk.Frame(left, style="Panel.TFrame")
        tree_wrap.pack(fill=tk.BOTH, expand=True, padx=12)
        self.strategy_tree = ttk.Treeview(tree_wrap, columns=cols, show="headings", height=10)
        for c in cols:
            self.strategy_tree.heading(c, text=headings[c])
            anchor = "w" if c in ("type", "params") else "center"
            self.strategy_tree.column(c, width=widths[c], anchor=anchor,
                                      stretch=(c in ("type", "params")))
        self.strategy_tree.tag_configure("on", foreground=COL_FG)
        self.strategy_tree.tag_configure("off", foreground=COL_ROW_OFF)
        self.strategy_tree.tag_configure("long", foreground=COL_GREEN)
        self.strategy_tree.tag_configure("short", foreground=COL_RED)
        vsb = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.strategy_tree.yview)
        self.strategy_tree.configure(yscrollcommand=vsb.set)
        self.strategy_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.strategy_tree.bind("<Double-1>", self._on_row_double_click)

        # Row action buttons.
        btns = ttk.Frame(left, style="Panel.TFrame")
        btns.pack(fill=tk.X, padx=12, pady=8)
        ttk.Button(btns, text="Toggle ON/OFF", command=self._toggle_selected).pack(side=tk.LEFT)
        ttk.Button(btns, text="Edit Allocation…", command=self._edit_alloc_selected).pack(
            side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Remove", style="Danger.TButton",
                  command=self._remove_selected).pack(side=tk.LEFT)
        self.gross_lbl = ttk.Label(btns, text="Gross allocation: 0%", style="Muted.TLabel")
        self.gross_lbl.pack(side=tk.RIGHT)

        self._build_add_form(left)

        # ---- RIGHT: portfolio --------------------------------------------
        right = ttk.Frame(body, style="Panel.TFrame")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(8, 0))
        right.configure(width=420)
        right.pack_propagate(False)

        ttk.Label(right, text="PORTFOLIO", style="Heading.TLabel").pack(
            anchor="w", padx=12, pady=(10, 4))

        # Account readout grid.
        grid = ttk.Frame(right, style="Panel.TFrame")
        grid.pack(fill=tk.X, padx=12, pady=(0, 8))
        self.account_labels: Dict[str, ttk.Label] = {}
        readouts = [
            ("NetLiquidation", "Net Liquidation"),
            ("TotalCashValue", "Total Cash"),
            ("BuyingPower", "Buying Power"),
            ("GrossPositionValue", "Gross Position Value"),
            ("UnrealizedPnL", "Unrealized P/L"),
            ("RealizedPnL", "Realized P/L"),
        ]
        for i, (key, label) in enumerate(readouts):
            r, c = divmod(i, 2)
            cell = ttk.Frame(grid, style="Panel.TFrame")
            cell.grid(row=r, column=c, sticky="ew", padx=6, pady=6)
            grid.columnconfigure(c, weight=1)
            ttk.Label(cell, text=label.upper(), style="Muted.TLabel",
                      font=("Segoe UI", 8, "bold")).pack(anchor="w")
            val = ttk.Label(cell, text="—", style="Value.TLabel")
            val.pack(anchor="w")
            self.account_labels[key] = val

        ttk.Label(right, text="POSITIONS", style="Heading.TLabel").pack(
            anchor="w", padx=12, pady=(6, 4))
        pcols = ("symbol", "qty", "avg", "mkt", "upnl")
        pheads = {"symbol": "Symbol", "qty": "Qty", "avg": "Avg Cost",
                  "mkt": "Mkt Value", "upnl": "Unrl P/L"}
        pos_wrap = ttk.Frame(right, style="Panel.TFrame")
        pos_wrap.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))
        self.positions_tree = ttk.Treeview(pos_wrap, columns=pcols, show="headings", height=8)
        for c in pcols:
            self.positions_tree.heading(c, text=pheads[c])
            self.positions_tree.column(c, width=80, anchor="center" if c != "symbol" else "w")
        self.positions_tree.tag_configure("pos", foreground=COL_GREEN)
        self.positions_tree.tag_configure("neg", foreground=COL_RED)
        pvsb = ttk.Scrollbar(pos_wrap, orient="vertical", command=self.positions_tree.yview)
        self.positions_tree.configure(yscrollcommand=pvsb.set)
        self.positions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pvsb.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_add_form(self, parent):
        form = ttk.Frame(parent, style="Panel.TFrame")
        form.pack(fill=tk.X, padx=12, pady=(0, 12))
        ttk.Label(form, text="ADD STRATEGY", style="Heading.TLabel").grid(
            row=0, column=0, columnspan=8, sticky="w", pady=(6, 4))

        ttk.Label(form, text="Type", style="Muted.TLabel").grid(row=1, column=0, sticky="w")
        self.type_var = tk.StringVar(value=list(STRATEGY_REGISTRY.keys())[0])
        type_cb = ttk.Combobox(form, textvariable=self.type_var, state="readonly",
                               values=list(STRATEGY_REGISTRY.keys()), width=16)
        type_cb.grid(row=2, column=0, padx=(0, 10), sticky="w")
        type_cb.bind("<<ComboboxSelected>>", lambda e: self._rebuild_param_fields())

        ttk.Label(form, text="Symbol", style="Muted.TLabel").grid(row=1, column=1, sticky="w")
        self.symbol_var = tk.StringVar(value="AAPL")
        ttk.Entry(form, textvariable=self.symbol_var, width=8).grid(
            row=2, column=1, padx=(0, 10), sticky="w")

        ttk.Label(form, text="Alloc %", style="Muted.TLabel").grid(row=1, column=2, sticky="w")
        self.alloc_var = tk.StringVar(value="20")
        ttk.Entry(form, textvariable=self.alloc_var, width=6).grid(
            row=2, column=2, padx=(0, 10), sticky="w")

        # Dynamic parameter fields container.
        self.param_frame = ttk.Frame(form, style="Panel.TFrame")
        self.param_frame.grid(row=2, column=3, columnspan=4, sticky="w")
        self.param_vars: Dict[str, tk.StringVar] = {}
        self._rebuild_param_fields()

        ttk.Button(form, text="＋ Add", style="Accent.TButton",
                  command=self._on_add_strategy).grid(row=2, column=7, padx=(10, 0))

    def _rebuild_param_fields(self):
        for child in self.param_frame.winfo_children():
            child.destroy()
        self.param_vars.clear()
        cls = STRATEGY_REGISTRY[self.type_var.get()]
        for i, (key, label, default) in enumerate(cls.param_spec):
            ttk.Label(self.param_frame, text=label, style="Muted.TLabel").grid(
                row=0, column=i, sticky="w", padx=(0, 4))
            var = tk.StringVar(value=str(default))
            ttk.Entry(self.param_frame, textvariable=var, width=7).grid(
                row=1, column=i, sticky="w", padx=(0, 8))
            self.param_vars[key] = var

    # ================= bottom (rebalance + log) ===========================
    def _build_bottom(self):
        bottom = ttk.Frame(self.root, style="Panel.TFrame")
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=14, pady=(6, 12))

        bar = ttk.Frame(bottom, style="Panel.TFrame")
        bar.pack(fill=tk.X, padx=12, pady=8)
        ttk.Button(bar, text="↻ Compute Rebalance", command=self._preview_rebalance).pack(side=tk.LEFT)
        ttk.Button(bar, text="▶ Send Orders", style="Accent.TButton",
                  command=self._send_orders).pack(side=tk.LEFT, padx=8)
        self.arm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Arm LIVE orders (IBKR)", variable=self.arm_var).pack(side=tk.LEFT)
        ttk.Label(bar, text="  (DEMO always simulates fills; live orders require Arm + IBKR)",
                  style="Muted.TLabel").pack(side=tk.LEFT)

        panes = ttk.Frame(bottom, style="Panel.TFrame")
        panes.pack(fill=tk.X, padx=12, pady=(0, 8))

        # Rebalance preview.
        prev_wrap = ttk.Frame(panes, style="Panel.TFrame")
        prev_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        ttk.Label(prev_wrap, text="REBALANCE PREVIEW", style="Heading.TLabel").pack(anchor="w")
        self.preview = tk.Text(prev_wrap, height=6, bg=COL_PANEL_2, fg=COL_FG,
                               insertbackground=COL_FG, font=FONT_MONO, relief="flat",
                               wrap="none", highlightthickness=1, highlightbackground=COL_GRID)
        self.preview.pack(fill=tk.BOTH, expand=True, pady=4)
        self.preview.configure(state="disabled")

        # Activity log.
        log_wrap = ttk.Frame(panes, style="Panel.TFrame")
        log_wrap.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))
        ttk.Label(log_wrap, text="ACTIVITY LOG", style="Heading.TLabel").pack(anchor="w")
        self.log = tk.Text(log_wrap, height=6, bg=COL_PANEL_2, fg=COL_MUTED,
                           insertbackground=COL_FG, font=FONT_MONO, relief="flat",
                           wrap="word", highlightthickness=1, highlightbackground=COL_GRID)
        self.log.pack(fill=tk.BOTH, expand=True, pady=4)
        self.log.configure(state="disabled")

    # ================= backend lifecycle ==================================
    def _start_backend(self):
        self.backend.start()

    def _on_reconnect(self):
        try:
            host = self.host_var.get().strip()
            port = int(self.port_var.get())
            cid = int(self.cid_var.get())
        except ValueError:
            messagebox.showerror("Invalid connection", "Port and Client Id must be integers.")
            return
        self._append_log(f"Reconnecting to {host}:{port} (clientId={cid})…")
        self.backend.stop()
        symbols = list(self.backend.symbols)
        self.backend = TradingBackend(symbols, host=host, port=port, client_id=cid)
        # Re-register any symbols referenced by existing strategy rows.
        for alloc in self.rows.values():
            self.backend.add_symbol(alloc.symbol)
        self.root.after(100, self._start_backend)

    # ================= strategy CRUD ======================================
    def _next_id(self) -> int:
        self._alloc_seq += 1
        return self._alloc_seq

    def _seed_examples(self):
        try:
            self._add_allocation(create_strategy("Trend Following", "AAPL", fast=20, slow=50), 25.0, True)
            self._add_allocation(create_strategy("Mean Reversion", "MSFT", lookback=20, entry_z=2.0), 20.0, True)
            self._add_allocation(create_strategy("Momentum", "NVDA", lookback=60), 15.0, False)
        except Exception as exc:  # pragma: no cover - defensive
            self._append_log(f"Seed error: {exc!r}")

    def _add_allocation(self, strategy, alloc_pct: float, enabled: bool):
        alloc = StrategyAllocation(strategy=strategy, allocation_pct=alloc_pct,
                                   enabled=enabled, id=self._next_id())
        self.backend.add_symbol(strategy.symbol)
        item = self.strategy_tree.insert("", "end", values=self._row_values(alloc, 0.0, 0.0))
        self.rows[item] = alloc
        self._apply_row_style(item, alloc, 0.0)
        self._update_gross_label()

    def _row_values(self, alloc: StrategyAllocation, signal: float, target: float):
        state = "● ON" if alloc.enabled else "○ OFF"
        return (
            state,
            alloc.strategy.type_name,
            alloc.symbol,
            alloc.strategy.describe_params(),
            f"{alloc.allocation_pct:.0f}%",
            f"{signal:+.2f}" if alloc.enabled else "—",
            f"{target*100:+.1f}%" if alloc.enabled else "—",
        )

    def _apply_row_style(self, item, alloc: StrategyAllocation, signal: float):
        if not alloc.enabled:
            self.strategy_tree.item(item, tags=("off",))
        elif signal > 0.05:
            self.strategy_tree.item(item, tags=("long",))
        elif signal < -0.05:
            self.strategy_tree.item(item, tags=("short",))
        else:
            self.strategy_tree.item(item, tags=("on",))

    def _on_add_strategy(self):
        type_name = self.type_var.get()
        symbol = self.symbol_var.get().strip().upper()
        if not symbol:
            messagebox.showerror("Missing symbol", "Please enter a ticker symbol.")
            return
        try:
            alloc_pct = float(self.alloc_var.get())
            if not (0 < alloc_pct <= 100):
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid allocation", "Allocation must be a number in (0, 100].")
            return
        params = {}
        for key, var in self.param_vars.items():
            raw = var.get().strip()
            try:
                params[key] = float(raw)
            except ValueError:
                messagebox.showerror("Invalid parameter", f"Parameter '{key}' must be numeric.")
                return
        try:
            strat = create_strategy(type_name, symbol, **params)
        except Exception as exc:
            messagebox.showerror("Invalid strategy", str(exc))
            return
        self._add_allocation(strat, alloc_pct, True)
        self._append_log(f"Added {type_name} on {symbol} @ {alloc_pct:.0f}% "
                         f"({strat.describe_params()})")

    def _selected_items(self):
        return list(self.strategy_tree.selection())

    def _toggle_selected(self):
        for item in self._selected_items():
            alloc = self.rows.get(item)
            if alloc:
                alloc.enabled = not alloc.enabled
                self._append_log(f"{alloc.strategy.type_name} on {alloc.symbol} -> "
                                 f"{'ON' if alloc.enabled else 'OFF'}")
        self._update_gross_label()

    def _on_row_double_click(self, event):
        item = self.strategy_tree.identify_row(event.y)
        if item and item in self.rows:
            self.rows[item].enabled = not self.rows[item].enabled
            self._update_gross_label()

    def _edit_alloc_selected(self):
        items = self._selected_items()
        if not items:
            messagebox.showinfo("No selection", "Select a strategy row first.")
            return
        item = items[0]
        alloc = self.rows[item]
        new_val = simpledialog.askfloat(
            "Edit allocation", f"Allocation % for {alloc.strategy.type_name} ({alloc.symbol}):",
            initialvalue=alloc.allocation_pct, minvalue=0.0, maxvalue=100.0, parent=self.root)
        if new_val is not None:
            alloc.allocation_pct = new_val
            self._append_log(f"{alloc.strategy.type_name} on {alloc.symbol} allocation -> {new_val:.0f}%")
            self._update_gross_label()

    def _remove_selected(self):
        for item in self._selected_items():
            alloc = self.rows.pop(item, None)
            self.strategy_tree.delete(item)
            if alloc:
                self._append_log(f"Removed {alloc.strategy.type_name} on {alloc.symbol}")
        self._update_gross_label()

    def _update_gross_label(self):
        gross = sum(a.allocation_pct for a in self.rows.values() if a.enabled)
        color = COL_RED if gross > 100 else COL_MUTED
        self.gross_lbl.configure(text=f"Gross allocation (enabled): {gross:.0f}%",
                                 foreground=color)

    # ================= polling loops ======================================
    def _drain_events(self):
        """Poll the backend event queue (runs on the Tk main thread)."""
        try:
            while True:
                evt = self.backend.events.get_nowait()
                if evt.kind == "status":
                    self._update_status_pill(evt.payload.get("connected", False),
                                             evt.payload.get("mode", "DEMO"))
                    self._append_log(evt.message)
                elif evt.kind in ("log", "fill"):
                    self._append_log(evt.message)
                elif evt.kind == "error":
                    self._append_log("⚠ " + evt.message)
        except Exception:
            pass  # queue empty
        finally:
            self.root.after(200, self._drain_events)

    def _refresh(self):
        """Repaint numeric state from a backend snapshot (Tk main thread)."""
        try:
            snap = self.backend.snapshot()
            self._refresh_strategies(snap)
            self._refresh_account(snap["account"])
            self._refresh_positions(snap["positions"])
        except Exception as exc:  # pragma: no cover - defensive
            self._append_log(f"refresh error: {exc!r}")
        finally:
            self.root.after(1000, self._refresh)

    def _refresh_strategies(self, snap):
        history = snap["history"]
        allocs = list(self.rows.values())
        weights = aggregate_target_weights(allocs, history)
        for item, alloc in self.rows.items():
            prices = history.get(alloc.symbol, [])
            signal = alloc.strategy.generate_signal(prices) if alloc.enabled else 0.0
            target = weights.get(alloc.symbol, 0.0) if alloc.enabled else 0.0
            self.strategy_tree.item(item, values=self._row_values(alloc, signal, target))
            self._apply_row_style(item, alloc, signal)

    def _refresh_account(self, account):
        for key, lbl in self.account_labels.items():
            val = account.get(key, 0.0)
            text = f"${val:,.0f}"
            color = COL_FG
            if key in ("UnrealizedPnL", "RealizedPnL"):
                color = COL_GREEN if val > 0 else (COL_RED if val < 0 else COL_FG)
                text = f"${val:,.0f}"
            lbl.configure(text=text, foreground=color)

    def _refresh_positions(self, positions):
        existing = {self.positions_tree.set(i, "symbol"): i for i in self.positions_tree.get_children()}
        seen = set()
        for p in positions:
            seen.add(p["symbol"])
            vals = (
                p["symbol"], f"{p['quantity']:,.0f}", f"{p['avg_cost']:,.2f}",
                f"${p['market_value']:,.0f}", f"${p['unrealized_pnl']:,.0f}",
            )
            tag = "pos" if p["unrealized_pnl"] >= 0 else "neg"
            if p["symbol"] in existing:
                self.positions_tree.item(existing[p["symbol"]], values=vals, tags=(tag,))
            else:
                self.positions_tree.insert("", "end", values=vals, tags=(tag,))
        # Drop rows for closed positions.
        for sym, item in existing.items():
            if sym not in seen:
                self.positions_tree.delete(item)

    # ================= rebalance / orders =================================
    def _compute_plans(self):
        snap = self.backend.snapshot()
        return build_order_plans(
            list(self.rows.values()),
            snap["history"],
            snap["prices"],
            {p["symbol"]: int(p["quantity"]) for p in snap["positions"]},
            snap["account"].get("NetLiquidation", 0.0),
        )

    def _preview_rebalance(self):
        plans = self._compute_plans()
        self.preview.configure(state="normal")
        self.preview.delete("1.0", tk.END)
        header = f"{'SYM':<6}{'TGT%':>7}{'TGT SH':>9}{'CUR SH':>9}{'DELTA':>9}  ACTION\n"
        self.preview.insert(tk.END, header)
        self.preview.insert(tk.END, "-" * 52 + "\n")
        actionable = [p for p in plans if p.delta_shares != 0]
        if not actionable:
            self.preview.insert(tk.END, "No trades needed — book is at target.\n")
        for p in sorted(plans, key=lambda x: -abs(x.delta_shares)):
            self.preview.insert(
                tk.END,
                f"{p.symbol:<6}{p.target_weight*100:>6.1f}%{p.target_shares:>9,}"
                f"{p.current_shares:>9,}{p.delta_shares:>+9,}  {p.action}\n",
            )
        self.preview.configure(state="disabled")
        return plans

    def _send_orders(self):
        plans = self._preview_rebalance()
        actionable = [p for p in plans if p.delta_shares != 0]
        if not actionable:
            self._append_log("Rebalance: nothing to do.")
            return
        live = self.arm_var.get() and self.backend.mode == "IBKR" and self.backend.connected
        guarded = not live
        if live:
            if not messagebox.askyesno(
                "Confirm LIVE orders",
                f"Send {len(actionable)} LIVE market order(s) to IBKR?\n"
                "This will transmit real orders to your account."):
                self._append_log("Live order submission cancelled by user.")
                return
        messages = self.backend.submit_orders(actionable, guarded=guarded)
        for m in messages:
            self._append_log(m)

    # ================= status pill + log ==================================
    def _update_status_pill(self, connected: bool, mode: str):
        if connected and mode == "IBKR":
            self.status_pill.configure(text="  ● CONNECTED TO IBKR  ", bg=COL_GREEN, fg="#0e1117")
        else:
            self.status_pill.configure(text="  ● DEMO MODE (SIMULATED)  ", bg=COL_AMBER, fg="#0e1117")

    def _append_log(self, message: str):
        self.log.configure(state="normal")
        self.log.insert(tk.END, message.rstrip() + "\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    # ================= shutdown ===========================================
    def _on_close(self):
        try:
            self.backend.stop()
        finally:
            self.root.destroy()


def main():
    """Create the Tk root and run the cockpit. Only called explicitly."""
    root = tk.Tk()
    TradingCockpit(root)
    root.mainloop()


if __name__ == "__main__":
    # A GUI window is only created here, never on import.
    if "--check" in sys.argv:
        # Headless import/wiring check (no window, no mainloop).
        print("app.py import OK; registry:", list(STRATEGY_REGISTRY.keys()))
    else:
        main()
