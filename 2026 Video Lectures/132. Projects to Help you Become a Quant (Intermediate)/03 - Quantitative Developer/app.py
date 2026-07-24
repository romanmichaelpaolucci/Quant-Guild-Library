"""
app.py
======

A **tkinter** front-end for the *Order Book Simulator*.

It wires together the pure matching engine (:mod:`order_book`) and the
stochastic flow generator (:mod:`market_sim`) into a live, exchange-style
"depth of market" screen:

* a **price ladder** (bids in green, asks in red) with the best bid/ask, spread
  and mid highlighted,
* a **cumulative-depth staircase** drawn on a Canvas,
* a **time & sales tape** of recent executions,
* a **manual order-entry** panel (submit limit/market buy/sell, cancel your
  resting orders and watch them fill), and
* run **controls** (start/pause the simulated flow, speed, aggressiveness,
  reset).

The real-time loop is driven by ``root.after`` -- there are no blocking sleeps
on the Tk main thread. This module is import-safe: constructing the window and
entering the main loop only happens inside ``main()`` under the
``if __name__ == "__main__"`` guard, so ``import app`` never opens a window.
"""

from __future__ import annotations

import time
import tkinter as tk
from collections import deque
from tkinter import font as tkfont
from tkinter import messagebox, ttk

from market_sim import MarketSimulator, SimConfig
from order_book import OrderBook, OrderType, Side, Trade

# --------------------------------------------------------------------------
# Palette -- a dark, terminal-like theme reminiscent of a real trading screen.
# --------------------------------------------------------------------------
BG = "#0e1116"          # window background
PANEL = "#161b22"       # panel / card background
GRID = "#232a34"        # subtle separators
TEXT = "#d1d4dc"        # primary text
MUTED = "#7d8590"       # secondary text
GREEN = "#26a69a"       # bids / buy aggressor
GREEN_HI = "#0f3d38"    # bid row highlight
RED = "#ef5350"         # asks / sell aggressor
RED_HI = "#43201f"      # ask row highlight
ACCENT = "#f0b90b"      # spread / mid / accents
USER = "#4aa3ff"        # the user's own orders

LADDER_LEVELS = 12      # price levels shown per side in the ladder
DEPTH_LEVELS = 20       # price levels fed into the depth chart
TAPE_MAX = 200          # trades retained in the tape

# Exchange-style price band ("fat-finger" collar). A manual LIMIT order priced
# more than this fraction *through* the market -- a buy well above the market or
# a sell well below it -- would sweep the book and execute at once, then leave
# its residual resting at that absurd price. Real venues reject or collar such
# orders. We warn, and if confirmed, cap ("collar") the price to the band edge
# so it fills up to there and any remainder rests near the market instead of
# parking at the typed price.
FAT_FINGER_BAND = 0.10  # 10% collar around the reference (mid) price


class OrderBookApp:
    """The whole GUI, owning one :class:`OrderBook` and one simulator."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.book = OrderBook(tick_size=0.01)
        self.sim = MarketSimulator(self.book, SimConfig(seed=None))

        self.running = False
        self.interval_ms = 250          # sim tick period (driven by the slider)
        self.tape: deque[tuple[str, Trade]] = deque(maxlen=TAPE_MAX)
        self.user_orders: set[int] = set()
        self._last_prices: dict[int, float] = {}  # ladder-side -> last mid (flash)

        self.mono = tkfont.Font(family="Consolas", size=11)
        self.mono_big = tkfont.Font(family="Consolas", size=22, weight="bold")
        self.mono_small = tkfont.Font(family="Consolas", size=10)

        self._build_style()
        self._build_layout()

        # Prime the book so the first frame isn't empty, then start the loop.
        self.sim.step()
        self._refresh()
        self.root.after(self.interval_ms, self._tick)

    # ------------------------------------------------------------------ style
    def _build_style(self) -> None:
        self.root.title("Order Book Simulator  —  Quant Guild")
        self.root.configure(bg=BG)
        self.root.geometry("1180x760")
        self.root.minsize(1000, 640)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=BG, foreground=TEXT, fieldbackground=PANEL)
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Panel.TLabel", background=PANEL, foreground=TEXT)
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
        style.configure(
            "Header.TLabel", background=BG, foreground=TEXT,
            font=("Segoe UI", 15, "bold"),
        )
        style.configure(
            "TButton", background=GRID, foreground=TEXT, borderwidth=0,
            focuscolor=GRID, padding=6,
        )
        style.map(
            "TButton",
            background=[("active", "#2d3746"), ("pressed", "#39485a")],
        )
        style.configure(
            "Accent.TButton", background=ACCENT, foreground="#1a1a1a",
            font=("Segoe UI", 10, "bold"),
        )
        style.map("Accent.TButton", background=[("active", "#ffcf3f")])
        style.configure("TRadiobutton", background=PANEL, foreground=TEXT)
        style.map("TRadiobutton", background=[("active", PANEL)])
        style.configure("TLabelframe", background=PANEL, foreground=MUTED,
                        bordercolor=GRID)
        style.configure("TLabelframe.Label", background=PANEL, foreground=MUTED)
        style.configure("TEntry", fieldbackground="#0b0e13", foreground=TEXT,
                        insertcolor=TEXT, bordercolor=GRID)

        # Treeviews (ladder + tape + user orders).
        style.configure(
            "Book.Treeview", background=PANEL, fieldbackground=PANEL,
            foreground=TEXT, borderwidth=0, rowheight=22, font=self.mono,
        )
        style.configure(
            "Book.Treeview.Heading", background=GRID, foreground=MUTED,
            font=("Segoe UI", 9, "bold"), relief="flat",
        )
        style.map("Book.Treeview", background=[("selected", "#243044")])
        style.configure("Horizontal.TScale", background=PANEL)

    # ----------------------------------------------------------------- layout
    def _build_layout(self) -> None:
        # Top statistics bar.
        top = ttk.Frame(self.root, padding=(14, 10))
        top.pack(side="top", fill="x")
        ttk.Label(top, text="ORDER BOOK", style="Header.TLabel").pack(side="left")
        ttk.Label(top, text="   SIM-X perpetual", style="TLabel",
                  foreground=MUTED).pack(side="left")

        self.stat_vars = {k: tk.StringVar(value="—")
                          for k in ("bid", "ask", "spread", "mid", "last")}
        self._build_stat(top, "BID", "bid", GREEN)
        self._build_stat(top, "ASK", "ask", RED)
        self._build_stat(top, "SPREAD", "spread", ACCENT)
        self._build_stat(top, "MID", "mid", TEXT)
        self._build_stat(top, "LAST", "last", MUTED)

        # Main three-column body.
        body = ttk.Frame(self.root, padding=(12, 4, 12, 12))
        body.pack(side="top", fill="both", expand=True)
        body.columnconfigure(0, weight=0, minsize=300)  # ladder
        body.columnconfigure(1, weight=1)               # depth chart
        body.columnconfigure(2, weight=0, minsize=300)  # tape + controls
        body.rowconfigure(0, weight=1)

        self._build_ladder(body)
        self._build_depth_chart(body)
        self._build_right_column(body)

    def _build_stat(self, parent, label: str, key: str, color: str) -> None:
        box = ttk.Frame(parent)
        box.pack(side="right", padx=12)
        ttk.Label(box, text=label, foreground=MUTED,
                  font=("Segoe UI", 8, "bold")).pack(anchor="e")
        lbl = tk.Label(box, textvariable=self.stat_vars[key], bg=BG, fg=color,
                       font=self.mono_big)
        lbl.pack(anchor="e")

    # -- ladder -------------------------------------------------------------
    def _build_ladder(self, parent) -> None:
        frame = ttk.Frame(parent, style="Panel.TFrame", padding=8)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        ttk.Label(frame, text="DEPTH OF MARKET", style="Muted.TLabel",
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 6))

        cols = ("bid", "price", "ask")
        tv = ttk.Treeview(frame, columns=cols, show="headings",
                          style="Book.Treeview", height=LADDER_LEVELS * 2 + 1)
        tv.heading("bid", text="Bid Size")
        tv.heading("price", text="Price")
        tv.heading("ask", text="Ask Size")
        tv.column("bid", width=90, anchor="e")
        tv.column("price", width=100, anchor="center")
        tv.column("ask", width=90, anchor="w")

        tv.tag_configure("ask", foreground=RED)
        tv.tag_configure("bid", foreground=GREEN)
        tv.tag_configure("best_ask", foreground="#ffffff", background=RED_HI)
        tv.tag_configure("best_bid", foreground="#ffffff", background=GREEN_HI)
        tv.tag_configure("spread", foreground=ACCENT, background="#1c2029")
        tv.tag_configure("user", foreground=USER)
        tv.pack(fill="both", expand=True)
        self.ladder = tv

        legend = ttk.Frame(frame, style="Panel.TFrame")
        legend.pack(fill="x", pady=(6, 0))
        ttk.Label(legend, text="■ bids", style="Panel.TLabel",
                  foreground=GREEN, font=self.mono_small).pack(side="left")
        ttk.Label(legend, text="  ■ asks", style="Panel.TLabel",
                  foreground=RED, font=self.mono_small).pack(side="left")
        ttk.Label(legend, text="  ■ your orders", style="Panel.TLabel",
                  foreground=USER, font=self.mono_small).pack(side="left")

    # -- depth chart --------------------------------------------------------
    def _build_depth_chart(self, parent) -> None:
        frame = ttk.Frame(parent, style="Panel.TFrame", padding=8)
        frame.grid(row=0, column=1, sticky="nsew")
        ttk.Label(frame, text="CUMULATIVE DEPTH", style="Muted.TLabel",
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 6))
        self.canvas = tk.Canvas(frame, bg=PANEL, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._draw_depth())

    # -- right column: tape, order entry, controls --------------------------
    def _build_right_column(self, parent) -> None:
        col = ttk.Frame(parent)
        col.grid(row=0, column=2, sticky="nsew", padx=(10, 0))
        col.rowconfigure(0, weight=1)
        col.columnconfigure(0, weight=1)

        # Trades tape.
        tape_frame = ttk.Frame(col, style="Panel.TFrame", padding=8)
        tape_frame.grid(row=0, column=0, sticky="nsew")
        ttk.Label(tape_frame, text="TIME & SALES", style="Muted.TLabel",
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 6))
        cols = ("time", "price", "size", "side")
        tape = ttk.Treeview(tape_frame, columns=cols, show="headings",
                            style="Book.Treeview", height=10)
        for c, w, a in (("time", 78, "center"), ("price", 78, "e"),
                        ("size", 56, "e"), ("side", 52, "center")):
            tape.heading(c, text=c.capitalize())
            tape.column(c, width=w, anchor=a)
        tape.tag_configure("buy", foreground=GREEN)
        tape.tag_configure("sell", foreground=RED)
        tape.pack(fill="both", expand=True)
        self.tape_view = tape

        self._build_order_entry(col)
        self._build_controls(col)

    def _build_order_entry(self, parent) -> None:
        f = ttk.Labelframe(parent, text=" MANUAL ORDER ENTRY ", padding=10)
        f.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        for c in range(4):
            f.columnconfigure(c, weight=1)

        self.side_var = tk.StringVar(value="buy")
        self.type_var = tk.StringVar(value="limit")
        ttk.Radiobutton(f, text="Buy", value="buy", variable=self.side_var
                        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(f, text="Sell", value="sell", variable=self.side_var
                        ).grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(f, text="Limit", value="limit", variable=self.type_var,
                        command=self._sync_price_entry
                        ).grid(row=0, column=2, sticky="w")
        ttk.Radiobutton(f, text="Market", value="market", variable=self.type_var,
                        command=self._sync_price_entry
                        ).grid(row=0, column=3, sticky="w")

        ttk.Label(f, text="Price", style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Label(f, text="Size", style="Muted.TLabel").grid(
            row=1, column=2, sticky="w", pady=(8, 0))
        self.price_entry = ttk.Entry(f, font=self.mono)
        self.price_entry.grid(row=2, column=0, columnspan=2, sticky="ew", padx=(0, 6))
        self.size_entry = ttk.Entry(f, font=self.mono)
        self.size_entry.grid(row=2, column=2, columnspan=2, sticky="ew")
        self.size_entry.insert(0, "10")

        ttk.Button(f, text="Submit Order", style="Accent.TButton",
                   command=self._submit_manual
                   ).grid(row=3, column=0, columnspan=4, sticky="ew", pady=(10, 4))

        # The user's resting orders, with a cancel button.
        ttk.Label(f, text="Your resting orders", style="Muted.TLabel").grid(
            row=4, column=0, columnspan=4, sticky="w", pady=(6, 2))
        cols = ("id", "side", "price", "qty")
        uv = ttk.Treeview(f, columns=cols, show="headings",
                          style="Book.Treeview", height=4)
        for c, w, a in (("id", 46, "center"), ("side", 52, "center"),
                        ("price", 78, "e"), ("qty", 56, "e")):
            uv.heading(c, text=c.capitalize())
            uv.column(c, width=w, anchor=a)
        uv.tag_configure("buy", foreground=GREEN)
        uv.tag_configure("sell", foreground=RED)
        uv.grid(row=5, column=0, columnspan=4, sticky="ew")
        self.user_view = uv
        ttk.Button(f, text="Cancel Selected", command=self._cancel_selected
                   ).grid(row=6, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        self._sync_price_entry()

    def _build_controls(self, parent) -> None:
        f = ttk.Labelframe(parent, text=" SIMULATION ", padding=10)
        f.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)

        self.start_btn = ttk.Button(f, text="▶  Start Flow", style="Accent.TButton",
                                    command=self._toggle_running)
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(f, text="⟲  Reset", command=self._reset
                   ).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        ttk.Label(f, text="Speed", style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", pady=(10, 0))
        self.speed = tk.DoubleVar(value=4.0)  # ticks/sec
        ttk.Scale(f, from_=1.0, to=20.0, variable=self.speed,
                  orient="horizontal", command=self._on_speed
                  ).grid(row=2, column=0, columnspan=2, sticky="ew")

        ttk.Label(f, text="Aggressiveness", style="Muted.TLabel").grid(
            row=3, column=0, sticky="w", pady=(6, 0))
        self.aggr = tk.DoubleVar(value=1.0)
        ttk.Scale(f, from_=0.2, to=3.0, variable=self.aggr,
                  orient="horizontal", command=self._on_aggr
                  ).grid(row=4, column=0, columnspan=2, sticky="ew")

        ttk.Button(f, text="How matching works  (rules & big-O)",
                   command=self._show_help
                   ).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    # --------------------------------------------------------------- controls
    def _toggle_running(self) -> None:
        self.running = not self.running
        self.start_btn.configure(
            text="⏸  Pause Flow" if self.running else "▶  Start Flow")

    def _on_speed(self, _=None) -> None:
        self.interval_ms = max(20, int(1000 / self.speed.get()))

    def _on_aggr(self, _=None) -> None:
        self.sim.cfg.aggressiveness = float(self.aggr.get())

    def _reset(self) -> None:
        self.running = False
        self.start_btn.configure(text="▶  Start Flow")
        self.book.clear()
        self.sim = MarketSimulator(self.book, SimConfig(seed=None))
        self.tape.clear()
        self.user_orders.clear()
        self.sim.step()
        self._refresh()

    def _sync_price_entry(self) -> None:
        is_market = self.type_var.get() == "market"
        if is_market:
            self.price_entry.delete(0, "end")
            self.price_entry.insert(0, "MKT")
            self.price_entry.configure(state="disabled")
        else:
            self.price_entry.configure(state="normal")
            if self.price_entry.get() in ("", "MKT"):
                self.price_entry.delete(0, "end")
                ref = self.book.mid() or self.sim.fair_value
                self.price_entry.insert(0, f"{ref:.2f}")

    # ----------------------------------------------------------- manual entry
    def _reference_price(self) -> float:
        """A sensible 'where the market is' price for the fat-finger band.

        Prefers the current mid; falls back to the last trade, then the
        simulator's latent fair value so the check works even on a one-sided or
        freshly-reset book.
        """
        mid = self.book.mid()
        if mid is not None:
            return mid
        if self.tape:
            return self.tape[0][1].price
        return self.sim.fair_value

    def _guard_fat_finger(self, side: Side, price: float):
        """Vet a manual limit price against the exchange price band.

        Returns the price to actually submit (possibly *collared* to the band
        edge), or ``None`` if the user declined the confirmation. Only the
        aggressive/marketable direction is dangerous -- a buy far *above* the
        market or a sell far *below* it -- so a genuinely passive far-away quote
        (buy well below, sell well above) is left untouched and simply rests.
        """
        ref = self._reference_price()
        if ref is None or ref <= 0:
            return price

        hi, lo = ref * (1 + FAT_FINGER_BAND), ref * (1 - FAT_FINGER_BAND)
        band_pct = int(round(FAT_FINGER_BAND * 100))

        if side is Side.BUY and price > hi:
            collar = round(hi, 2)
            thru = (price / ref - 1.0) * 100.0
            direction, edge = "above", collar
        elif side is Side.SELL and price < lo:
            collar = round(lo, 2)
            thru = (1.0 - price / ref) * 100.0
            direction, edge = "below", collar
        else:
            return price  # within band, or passive -- send as-is (it will rest)

        ok = messagebox.askyesno(
            "Fat-finger price check",
            f"Your {side.value.upper()} limit at {price:,.2f} is "
            f"{thru:,.0f}% {direction} the reference price ({ref:.2f}).\n\n"
            f"An order priced this far through the market sweeps the book and "
            f"executes immediately, leaving its remainder resting at "
            f"{price:,.2f} — a phantom price far from where the market is.\n\n"
            f"To stay realistic it will be collared to {edge:.2f} (a "
            f"{band_pct}% band): it fills up to there and any remainder rests "
            f"near the market.\n\n"
            f"Send the collared order?   (No = cancel and re-enter a price)",
            icon="warning",
        )
        return edge if ok else None

    def _submit_manual(self) -> None:
        side = Side.BUY if self.side_var.get() == "buy" else Side.SELL
        try:
            size = int(self.size_entry.get())
            if size <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid size", "Size must be a positive integer.")
            return

        try:
            if self.type_var.get() == "market":
                order, trades = self.book.submit_market(side, size, owner="user")
            else:
                price = float(self.price_entry.get())
                guarded = self._guard_fat_finger(side, price)
                if guarded is None:
                    return  # fat-finger check declined -- nothing sent
                order, trades = self.book.submit_limit(side, guarded, size,
                                                       owner="user")
        except ValueError:
            messagebox.showerror("Invalid price", "Enter a numeric limit price.")
            return

        for tr in trades:
            self.tape.appendleft((time.strftime("%H:%M:%S"), tr))
        if order.type is OrderType.LIMIT and not order.is_filled:
            self.user_orders.add(order.id)
        self._refresh()

    def _cancel_selected(self) -> None:
        sel = self.user_view.selection()
        if not sel:
            return
        for item in sel:
            oid = int(self.user_view.item(item, "values")[0])
            self.book.cancel(oid)
            self.user_orders.discard(oid)
        self._refresh()

    # -------------------------------------------------------------- main loop
    def _tick(self) -> None:
        if self.running:
            trades = self.sim.step()
            now = time.strftime("%H:%M:%S")
            for tr in trades:
                self.tape.appendleft((now, tr))
            self._refresh()
        self.root.after(self.interval_ms, self._tick)

    # ---------------------------------------------------------------- refresh
    def _refresh(self) -> None:
        self._update_stats()
        self._update_ladder()
        self._update_tape()
        self._update_user_orders()
        self._draw_depth()

    def _update_stats(self) -> None:
        bid, ask = self.book.best_bid(), self.book.best_ask()
        spread, mid = self.book.spread(), self.book.mid()
        self.stat_vars["bid"].set(f"{bid:.2f}" if bid is not None else "—")
        self.stat_vars["ask"].set(f"{ask:.2f}" if ask is not None else "—")
        self.stat_vars["spread"].set(f"{spread:.2f}" if spread is not None else "—")
        self.stat_vars["mid"].set(f"{mid:.2f}" if mid is not None else "—")
        if self.tape:
            self.stat_vars["last"].set(f"{self.tape[0][1].price:.2f}")

    def _user_prices(self) -> set[float]:
        return {
            o.price
            for oid in self.user_orders
            if (o := self.book.get_order(oid)) is not None and o.price is not None
        }

    def _update_ladder(self) -> None:
        tv = self.ladder
        tv.delete(*tv.get_children())
        asks = self.book.depth(Side.SELL, LADDER_LEVELS)   # best-first (low->high)
        bids = self.book.depth(Side.BUY, LADDER_LEVELS)    # best-first (high->low)
        user_px = self._user_prices()

        # Asks printed with the highest price on top so best ask sits just above
        # the spread line (classic DOM layout).
        for i, (px, qty) in enumerate(reversed(asks)):
            is_best = i == len(asks) - 1
            tags = ("best_ask",) if is_best else ("ask",)
            if px in user_px:
                tags = tags + ("user",)
            tv.insert("", "end", values=("", f"{px:.2f}", qty), tags=tags)

        spread = self.book.spread()
        tv.insert("", "end",
                  values=("", f"— {spread:.2f} —" if spread is not None else "—", ""),
                  tags=("spread",))

        for i, (px, qty) in enumerate(bids):
            tags = ("best_bid",) if i == 0 else ("bid",)
            if px in user_px:
                tags = tags + ("user",)
            tv.insert("", "end", values=(qty, f"{px:.2f}", ""), tags=tags)

    def _update_tape(self) -> None:
        tv = self.tape_view
        tv.delete(*tv.get_children())
        for ts, tr in self.tape:
            tag = "buy" if tr.aggressor is Side.BUY else "sell"
            arrow = "▲" if tr.aggressor is Side.BUY else "▼"
            tv.insert("", "end",
                      values=(ts, f"{tr.price:.2f}", tr.size, arrow), tags=(tag,))

    def _update_user_orders(self) -> None:
        # Drop ids the book no longer holds (filled/cancelled), then list them.
        self.user_orders = {o for o in self.user_orders if o in self.book}
        tv = self.user_view
        tv.delete(*tv.get_children())
        for oid in sorted(self.user_orders):
            o = self.book.get_order(oid)
            if o is None:
                continue
            tag = "buy" if o.side is Side.BUY else "sell"
            tv.insert("", "end",
                      values=(o.id, o.side.value, f"{o.price:.2f}", o.remaining),
                      tags=(tag,))

    # ------------------------------------------------------------ depth chart
    def _draw_depth(self) -> None:
        c = self.canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 40 or h < 40:
            return

        pad = 34
        bids = self.book.depth(Side.BUY, DEPTH_LEVELS)     # high -> low
        asks = self.book.depth(Side.SELL, DEPTH_LEVELS)    # low -> high
        if not bids or not asks:
            c.create_text(w / 2, h / 2, text="waiting for liquidity…",
                          fill=MUTED, font=self.mono)
            return

        mid = self.book.mid() or ((bids[0][0] + asks[0][0]) / 2)
        # Price window: symmetric span around mid covering both sides shown.
        lo = min(bids[-1][0], asks[0][0])
        hi = max(asks[-1][0], bids[0][0])
        span = max(hi - mid, mid - lo, self.book.tick_size)
        pmin, pmax = mid - span, mid + span

        # Cumulative depth curves.
        bid_cum, cum = [], 0
        for px, q in bids:
            cum += q
            bid_cum.append((px, cum))
        ask_cum, cum = [], 0
        for px, q in asks:
            cum += q
            ask_cum.append((px, cum))
        max_cum = max(bid_cum[-1][1], ask_cum[-1][1], 1)

        def px_x(p: float) -> float:
            return pad + (p - pmin) / (pmax - pmin) * (w - 2 * pad)

        def q_y(q: float) -> float:
            return (h - pad) - q / max_cum * (h - 2 * pad)

        baseline = h - pad
        # Grid + axis labels.
        for frac in (0.25, 0.5, 0.75, 1.0):
            y = baseline - frac * (h - 2 * pad)
            c.create_line(pad, y, w - pad, y, fill=GRID)
            c.create_text(pad - 6, y, text=f"{int(max_cum * frac)}", anchor="e",
                          fill=MUTED, font=self.mono_small)

        x_mid = px_x(mid)
        c.create_line(x_mid, pad, x_mid, baseline, fill=ACCENT, dash=(3, 3))
        c.create_text(x_mid, pad - 12, text=f"mid {mid:.2f}", fill=ACCENT,
                      font=self.mono_small)

        # Bid staircase (green, to the left of mid).
        poly = [(px_x(bids[0][0]), baseline)]
        prev = 0
        for px, cq in bid_cum:
            x = px_x(px)
            poly.append((x, q_y(prev)))
            poly.append((x, q_y(cq)))
            prev = cq
        poly.append((px_x(bid_cum[-1][0]), baseline))
        c.create_polygon(poly, fill=GREEN_HI, outline=GREEN, width=2)

        # Ask staircase (red, to the right of mid).
        poly = [(px_x(asks[0][0]), baseline)]
        prev = 0
        for px, cq in ask_cum:
            x = px_x(px)
            poly.append((x, q_y(prev)))
            poly.append((x, q_y(cq)))
            prev = cq
        poly.append((px_x(ask_cum[-1][0]), baseline))
        c.create_polygon(poly, fill=RED_HI, outline=RED, width=2)

        # Price axis ticks.
        for p in (pmin, mid, pmax):
            x = px_x(p)
            c.create_text(x, baseline + 14, text=f"{p:.2f}", fill=MUTED,
                          font=self.mono_small)

    # ------------------------------------------------------------------- help
    def _show_help(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("How the matching engine works")
        win.configure(bg=PANEL)
        win.geometry("620x560")
        txt = tk.Text(win, bg=PANEL, fg=TEXT, wrap="word", relief="flat",
                      font=("Segoe UI", 10), padx=16, pady=14)
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", HELP_TEXT)
        txt.configure(state="disabled")


HELP_TEXT = """PRICE-TIME PRIORITY MATCHING

The book keeps two sorted sides:
  • BIDS  — buy orders, best (highest) price first
  • ASKS  — sell orders, best (lowest) price first

An incoming order that crosses the spread is a TAKER. It is matched against
resting MAKER orders in a strict order:

  1. PRICE priority — the best price is filled first. A buyer sweeps the
     cheapest asks; a seller hits the highest bids.
  2. TIME priority — within one price level, the order that arrived first is
     filled first (FIFO). Resting early is rewarded.

Trades print at the resting maker's price, so the taker can get price
improvement versus its own limit.

ORDER TYPES
  • LIMIT  — has a price. Marketable portion trades immediately; the rest RESTS
             in the book as passive liquidity. A limit priced away from the
             market on the passive side (buy below / sell above) just sits and
             rests — it never trades until the market comes to it.
  • MARKET — no price; sweeps the far side for as much as it can. Any unfilled
             remainder is dropped (market orders never rest).
  • PARTIAL FILLS — a large order eats several resting orders/levels and the
             residual either rests (limit) or is dropped (market).

PRICE BAND (fat-finger protection)
  A manual LIMIT priced far THROUGH the market (a buy well above it or a sell
  well below it) would sweep the whole book and leave its remainder resting at
  that absurd price. Real venues reject or collar such orders, so the simulator
  warns you and, if you confirm, COLLARS the order to a 10% band: it fills up to
  the band edge and any remainder rests near the market — never at the typed
  price. Passive far-away quotes are unaffected; they just sit and rest.

DATA STRUCTURES & COMPLEXITY  (L = number of price levels)
  • Price levels: a SortedDict per side, keyed by integer ticks.
        best bid/ask ............ O(log L)
        insert a new level ...... O(log L)
  • Each level: an OrderedDict of orders in arrival order (FIFO time priority).
        cancel by id ............ O(1) within the level  → O(log L) overall
  • Matching a taker touching k orders across m emptied levels:
        O(k + m·log L)

Conservation: every fill moves the same quantity from a buyer to a seller, so
total shares bought == total shares sold, always.

TRY IT
  • Start Flow to let the simulator quote and trade around a drifting fair
    value. Use Speed/Aggressiveness to change the regime.
  • Submit your own LIMIT order inside the book and watch it rest, then get
    filled as the market comes to it — or send a MARKET order and watch it
    walk the book on the tape."""


def main() -> None:
    """Construct the window and enter the Tk main loop (guarded entry point)."""
    root = tk.Tk()
    OrderBookApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
