(() => {
  const holdingsEl = document.getElementById("thesis-body");
  let holdings = JSON.parse(holdingsEl.dataset.holdings || "[]");

  const thesisSymbol = document.getElementById("thesis-symbol");
  const thesisNarrative = document.getElementById("thesis-narrative");
  const thesisTarget = document.getElementById("thesis-target");
  const thesisEntry = document.getElementById("thesis-entry");
  const thesisConviction = document.getElementById("thesis-conviction");
  const thesisSourceTag = document.getElementById("thesis-source-tag");
  const thesisReevalBtn = document.getElementById("thesis-reeval-btn");

  const chatLog = document.getElementById("chat-log");
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const chatSend = document.getElementById("chat-send");
  const promptSelect = document.getElementById("prompt-select");
  const promptAskBtn = document.getElementById("prompt-ask-btn");
  const clockEl = document.getElementById("clock");
  const brainMeta = document.getElementById("brain-meta");

  let selectedSymbol = holdings[0]?.symbol || null;

  // Build client history from restored DOM messages (skip greeting / compressed banners for agent —
  // server already injects compressed memory; we still keep turns for UX continuity).
  const chatHistory = [];
  chatLog.querySelectorAll(".chat-msg").forEach((el) => {
    const role = el.classList.contains("user") ? "user" : "assistant";
    const content = el.querySelector(".chat-text")?.textContent || "";
    if (content && !content.startsWith("Terminal online")) {
      chatHistory.push({ role, content });
    }
  });

  function money(n) {
    if (n === null || n === undefined || n === "") return "—";
    return `$${Number(n).toFixed(2)}`;
  }

  function applyThesis(row) {
    const t = row?.thesis || { missing: true };
    thesisSymbol.textContent = `(${row?.symbol || "—"})`;
    if (t.missing || !t.narrative) {
      thesisNarrative.textContent =
        "No thesis saved yet. Ask the agent to research this name and call save_thesis.";
      thesisTarget.textContent = "—";
      thesisEntry.textContent = "—";
      thesisConviction.textContent = "—";
      if (thesisSourceTag) thesisSourceTag.textContent = "AWAITING LLM";
      return;
    }
    thesisNarrative.textContent = t.narrative;
    thesisTarget.textContent = money(t.target);
    thesisEntry.textContent = money(t.cost_basis ?? t.entry);
    const conv = t.conviction || "";
    const saved = t.saved_at ? ` · ${t.saved_at}` : "";
    thesisConviction.textContent = `${conv}${saved}`;
    if (thesisSourceTag) {
      thesisSourceTag.textContent = t.saved_at ? `JSON · ${t.saved_at}` : "JSON";
    }
  }

  function selectHolding(symbol) {
    const row = holdings.find((h) => h.symbol === symbol);
    if (!row) return;

    selectedSymbol = symbol;

    document.querySelectorAll(".holding-row").forEach((el) => {
      el.classList.toggle("is-selected", el.dataset.symbol === symbol);
    });

    applyThesis(row);

    if (thesisReevalBtn) {
      thesisReevalBtn.disabled = chatBusy || !symbol || symbol === "—";
    }
  }

  function refreshHoldings(next) {
    if (!Array.isArray(next)) return;
    holdings = next;
    holdingsEl.dataset.holdings = JSON.stringify(holdings);
    renderHoldingsTable(holdings);
    if (selectedSymbol && holdings.some((h) => h.symbol === selectedSymbol)) {
      selectHolding(selectedSymbol);
    } else if (holdings[0]?.symbol && holdings[0].symbol !== "—") {
      selectHolding(holdings[0].symbol);
    }
  }

  function renderHoldingsTable(rows) {
    const tbody = document.getElementById("holdings-tbody");
    if (!tbody) return;
    const usable = (rows || []).filter((h) => h.symbol && h.symbol !== "—");
    if (!usable.length) {
      tbody.innerHTML =
        '<tr class="holding-empty"><td colspan="4">No stock positions</td></tr>';
      return;
    }
    tbody.innerHTML = usable
      .map((h, i) => {
        const selected =
          h.symbol === selectedSymbol || (!selectedSymbol && i === 0);
        const pnlClass = Number(h.pnl_pct) >= 0 ? "up" : "down";
        const pnl = Number(h.pnl_pct);
        const sign = pnl >= 0 ? "+" : "";
        return (
          `<tr class="holding-row${selected ? " is-selected" : ""}" ` +
          `data-symbol="${h.symbol}" tabindex="0">` +
          `<td class="sym">${h.symbol}</td>` +
          `<td class="num">${Number(h.last).toFixed(2)}</td>` +
          `<td class="num ${pnlClass}">${sign}${pnl.toFixed(2)}</td>` +
          `<td class="num">${Number(h.weight).toFixed(2)}</td>` +
          `</tr>`
        );
      })
      .join("");
    tbody.querySelectorAll(".holding-row").forEach((row) => {
      row.addEventListener("click", () => selectHolding(row.dataset.symbol));
      row.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          selectHolding(row.dataset.symbol);
        }
      });
    });
  }

  function renderMetrics(metrics) {
    const grid = document.getElementById("metrics-grid");
    if (!grid || !Array.isArray(metrics)) return;
    grid.innerHTML = metrics
      .map(
        (m) =>
          `<article class="metric tone-${m.tone || "neutral"}" data-metric="${m.id}">` +
          `<div class="metric-label">${m.label}</div>` +
          `<div class="metric-value">${m.value}</div>` +
          `<div class="metric-delta">${m.delta || ""}</div>` +
          `</article>`
      )
      .join("");
  }

  function updateConnectionChrome(connection, source) {
    const pulse = document.querySelector("#connection-meta .pulse");
    const status = document.getElementById("conn-status");
    const port = document.getElementById("conn-port");
    const src = document.getElementById("conn-source");
    const holdingsTag = document.getElementById("holdings-source-tag");
    const metricsTag = document.getElementById("metrics-source-tag");
    if (pulse) {
      pulse.classList.toggle("is-offline", !(connection && connection.connected));
    }
    if (status && connection) {
      status.textContent = connection.status_label || status.textContent;
    }
    if (port && connection) {
      port.textContent = connection.label || port.textContent;
    }
    if (src) src.textContent = source === "demo" ? "DEMO FALLBACK" : "LIVE TWS";
    if (holdingsTag) holdingsTag.textContent = source === "tws" ? "TWS" : "DEMO";
    if (metricsTag) {
      metricsTag.textContent = source === "tws" ? "TWS ACCOUNT" : "DEMO";
    }
  }

  async function refreshLivePortfolio() {
    // Avoid racing the shared IB client while the agent is mid-tool-call.
    if (typeof chatBusy !== "undefined" && chatBusy) return;
    try {
      const [holdingsRes, metricsRes] = await Promise.all([
        fetch("/api/holdings"),
        fetch("/api/metrics"),
      ]);
      const holdingsPayload = await holdingsRes.json();
      const metricsPayload = await metricsRes.json();
      if (!holdingsRes.ok) return;

      const next = holdingsPayload.holdings || holdingsPayload;
      refreshHoldings(Array.isArray(next) ? next : []);
      if (metricsRes.ok) renderMetrics(metricsPayload.metrics || []);
      updateConnectionChrome(
        holdingsPayload.connection || metricsPayload.connection,
        holdingsPayload.source || metricsPayload.source
      );
    } catch (_) {
      /* keep last good snapshot */
    }
  }

  function scrollChatToBottom() {
    const pin = () => {
      chatLog.scrollTop = chatLog.scrollHeight;
    };
    pin();
    requestAnimationFrame(() => {
      pin();
      requestAnimationFrame(pin);
    });
  }

  // Keep the desk pinned to the newest message as content streams / hydrates.
  const chatScrollObserver = new MutationObserver(() => scrollChatToBottom());
  chatScrollObserver.observe(chatLog, {
    childList: true,
    subtree: true,
    characterData: true,
  });


  function renderMarkdown(text) {
    const raw = text == null ? "" : String(text);
    if (typeof marked === "undefined" || typeof DOMPurify === "undefined") {
      return null;
    }
    try {
      if (marked.setOptions) {
        marked.setOptions({ gfm: true, breaks: true });
      }
      const html = marked.parse(raw);
      return DOMPurify.sanitize(html, {
        USE_PROFILES: { html: true },
      });
    } catch (_) {
      return null;
    }
  }

  function typesetMath(el) {
    if (!el || typeof renderMathInElement !== "function") return;
    try {
      renderMathInElement(el, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\[", right: "\\]", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false },
        ],
        throwOnError: false,
      });
    } catch (_) {
      /* ignore katex errors on partial streams */
    }
  }

  function setMessageContent(textEl, content, { markdown = true } = {}) {
    textEl.dataset.raw = content == null ? "" : String(content);
    if (!markdown) {
      textEl.classList.add("raw");
      textEl.classList.remove("md");
      textEl.textContent = textEl.dataset.raw;
      return;
    }
    const html = renderMarkdown(textEl.dataset.raw);
    if (html == null) {
      textEl.classList.add("raw");
      textEl.classList.remove("md");
      textEl.textContent = textEl.dataset.raw;
      return;
    }
    textEl.classList.add("md");
    textEl.classList.remove("raw");
    textEl.innerHTML = html;
    typesetMath(textEl);
  }

  function appendMessage(role, content, pending = false) {
    const wrap = document.createElement("div");
    wrap.className = `chat-msg ${role}${pending ? " pending" : ""}`;

    const roleEl = document.createElement("div");
    roleEl.className = "chat-role";
    roleEl.textContent = role === "assistant" ? "ASTB" : "YOU";

    const body = document.createElement("div");
    body.className = "chat-body";

    if (role === "assistant") {
      const statusEl = document.createElement("div");
      statusEl.className = "chat-status";
      statusEl.hidden = true;
      body.appendChild(statusEl);
    }

    const textEl = document.createElement("div");
    textEl.className = "chat-text";
    setMessageContent(textEl, content, { markdown: role === "assistant" || !pending });

    body.appendChild(textEl);
    wrap.append(roleEl, body);
    chatLog.appendChild(wrap);
    scrollChatToBottom();
    return wrap;
  }

  function getAssistantParts(wrap) {
    return {
      statusEl: wrap.querySelector(".chat-status"),
      textEl: wrap.querySelector(".chat-text"),
    };
  }

  function updateBrainMeta(brain) {
    if (!brainMeta || !brain) return;
    const counts = brain.counts || {};
    brainMeta.textContent =
      `BRAIN T${counts.active_theses ?? 0} · TR${counts.trades ?? 0} · G${counts.goal_history ?? 0}`;
    if (brain.summary) brainMeta.title = brain.summary;
  }

  // Hydrate any server-rendered chat messages into markdown
  chatLog.querySelectorAll(".chat-msg").forEach((el) => {
    const textEl = el.querySelector(".chat-text");
    if (!textEl) return;
    const raw = textEl.textContent || "";
    const isUser = el.classList.contains("user");
    // Wrap assistant body if needed
    if (!el.querySelector(".chat-body")) {
      const roleEl = el.querySelector(".chat-role");
      const body = document.createElement("div");
      body.className = "chat-body";
      if (!isUser) {
        const statusEl = document.createElement("div");
        statusEl.className = "chat-status";
        statusEl.hidden = true;
        body.appendChild(statusEl);
      }
      body.appendChild(textEl);
      el.appendChild(body);
      if (roleEl && roleEl.parentElement === el) {
        el.insertBefore(roleEl, body);
      }
    }
    setMessageContent(textEl, raw, { markdown: true });
  });

  document.querySelectorAll(".holding-row").forEach((row) => {
    row.addEventListener("click", () => selectHolding(row.dataset.symbol));
    row.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        selectHolding(row.dataset.symbol);
      }
    });
  });

  let chatBusy = false;

  function setChatBusy(busy) {
    chatBusy = busy;
    chatSend.disabled = busy;
    if (promptAskBtn) {
      promptAskBtn.disabled = busy || !promptSelect?.value;
    }
    if (thesisReevalBtn) {
      thesisReevalBtn.disabled = busy || !selectedSymbol || selectedSymbol === "—";
    }
  }

  async function sendChatMessage(message, opts = {}) {
    const mode = opts.mode || null;
    const automated = mode === "auto_review";
    const text = automated
      ? "[AUTO] 5-minute thesis reevaluation"
      : String(message || "").trim();
    if ((!text && !automated) || chatBusy) return;

    appendMessage("user", text);
    chatHistory.push({ role: "user", content: text });
    if (!automated) chatInput.value = "";
    setChatBusy(true);
    if (autoToggle) autoToggle.closest(".auto-toggle")?.classList.toggle("is-running", automated);

    const pending = appendMessage("assistant", "", true);
    const { statusEl, textEl } = getAssistantParts(pending);
    pending.classList.add("streaming");
    if (statusEl) {
      statusEl.hidden = false;
      statusEl.textContent = automated ? "auto-review…" : "connecting…";
    }
    setMessageContent(textEl, "_…_", { markdown: true });
    scrollChatToBottom();

    let assembled = "";
    let toolNames = [];
    let finalized = false;

    try {
      const body = {
        message: automated ? "" : text,
        symbol: selectedSymbol,
        history: automated ? [] : chatHistory.slice(0, -1).slice(-12),
      };
      if (mode) body.mode = mode;

      const res = await fetch("/api/chat?stream=1", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        let errMsg = `chat failed (${res.status})`;
        try {
          const errBody = await res.json();
          errMsg = errBody.error || errMsg;
        } catch (_) {}
        throw new Error(errMsg);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const part of parts) {
          const line = part
            .split("\n")
            .map((l) => l.trim())
            .find((l) => l.startsWith("data:"));
          if (!line) continue;
          let event;
          try {
            event = JSON.parse(line.slice(5).trim());
          } catch (_) {
            continue;
          }

          const et = event.type;
          if (et === "status") {
            if (statusEl) {
              statusEl.hidden = false;
              statusEl.textContent = event.text || "";
            }
            scrollChatToBottom();
          } else if (et === "tool") {
            if (event.name) toolNames.push(event.name);
            if (statusEl) {
              statusEl.hidden = false;
              statusEl.textContent = `tool: ${event.name || "?"} ✓`;
            }
            scrollChatToBottom();
          } else if (et === "delta") {
            if (!assembled) {
              pending.classList.remove("pending");
              if (statusEl) statusEl.hidden = true;
            }
            assembled += event.text || "";
            let display = assembled;
            if (toolNames.length) {
              display = `\`[${toolNames.join(" → ")}]\`\n\n${assembled}`;
            }
            setMessageContent(textEl, display, { markdown: true });
            scrollChatToBottom();
          } else if (et === "done") {
            finalized = true;
            assembled = event.content || assembled;
            pending.classList.remove("pending", "streaming");
            if (statusEl) statusEl.hidden = true;
            const names = (event.tool_calls || []).map((t) => t.name).filter(Boolean);
            if (names.length) toolNames = names;
            let display = assembled;
            if (toolNames.length) {
              display = `\`[${toolNames.join(" → ")}]\`\n\n${assembled}`;
            }
            setMessageContent(textEl, display, { markdown: true });
            chatHistory.push({ role: "assistant", content: assembled });
            scrollChatToBottom();
          } else if (et === "meta") {
            updateBrainMeta(event.brain);
            refreshHoldings(event.holdings);
            if (Array.isArray(event.metrics)) renderMetrics(event.metrics);
            if (event.connection || event.source) {
              updateConnectionChrome(event.connection, event.source);
            }
            if (event.ux) applyUxState(event.ux);
            scrollChatToBottom();
          } else if (et === "error") {
            throw new Error(event.error || "stream error");
          }
        }
      }

      if (!finalized) {
        pending.classList.remove("pending", "streaming");
        if (!assembled) {
          setMessageContent(textEl, "_(empty response)_", { markdown: true });
        } else {
          chatHistory.push({ role: "assistant", content: assembled });
        }
      }
    } catch (err) {
      pending.classList.remove("pending", "streaming");
      if (statusEl) statusEl.hidden = true;
      setMessageContent(
        textEl,
        `**Error:** ${err.message || "request failed"}`,
        { markdown: true }
      );
    } finally {
      pending.classList.remove("streaming");
      setChatBusy(false);
      if (autoToggle) autoToggle.closest(".auto-toggle")?.classList.remove("is-running");
      chatInput.focus();
      scrollChatToBottom();
    }
  }


  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    await sendChatMessage(chatInput.value);
  });

  if (promptSelect && promptAskBtn) {
    promptSelect.addEventListener("change", () => {
      promptAskBtn.disabled = chatBusy || !promptSelect.value;
    });

    promptAskBtn.addEventListener("click", async () => {
      const prompt = promptSelect.value;
      if (!prompt) return;
      await sendChatMessage(prompt);
    });
  }

  if (thesisReevalBtn) {
    thesisReevalBtn.addEventListener("click", async () => {
      const sym = selectedSymbol;
      if (!sym || sym === "—") return;
      const row = holdings.find((h) => h.symbol === sym);
      const missing = !row?.thesis || row.thesis.missing;
      const last = row?.last != null ? Number(row.last) : null;
      const lastHint =
        last != null && Number.isFinite(last) && last > 0
          ? ` UI last/marketPrice≈$${last.toFixed(2)}.`
          : "";
      const prompt = missing
        ? `Research ${sym} and create a full thesis — narrative, OPTIMISTIC target strictly above current last, cost basis, and conviction — then call save_thesis.${lastHint} Prefer get_portfolio / get_historical_bars if get_market_snapshot fails or times out. Do not ask the UX for approval. Never set target equal to last price.`
        : `Revaluate ${sym} and ACT (do not ask the UX whether to proceed).${lastHint} ` +
          `Priority: (R) overweight vs max_single_name / cash_floor / avoid-list → EXECUTE risk trim or exit first ` +
          `(propose_position_size SELL target_weight_pct=max_single_name for trim, or liquidate=true if no upside / avoid); ` +
          `(B) at/through target or thesis played out → FULL liquidation liquidate=true → place_equity_order(confirm=true) → record_trade; ` +
          `(C) thesis invalidated → full exit + save_thesis; ` +
          `(A) valid + upside → HOLD residual; save_thesis OPTIMISTIC target STRICTLY ABOVE last (never target=last); optional BUY if underweight + cash allows; ` +
          `(D) no material change → one short affirmation. ` +
          `Skip qualify_stock for portfolio names. Prefer get_portfolio marketPrice if snapshot fails. Summarize what you DID.`;
      await sendChatMessage(prompt);
    });
  }

  function tickClock() {
    const now = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    const stamp =
      `${now.getUTCFullYear()}-${pad(now.getUTCMonth() + 1)}-${pad(now.getUTCDate())} ` +
      `${pad(now.getUTCHours())}:${pad(now.getUTCMinutes())}:${pad(now.getUTCSeconds())} UTC`;
    clockEl.textContent = stamp;
  }

  tickClock();
  setInterval(tickClock, 1000);
  setInterval(refreshLivePortfolio, 30000);
  chatInput.focus();
  if (selectedSymbol) selectHolding(selectedSymbol);
  scrollChatToBottom();

  // --- Automated thesis reevaluation (server-side async scheduler) ------
  const autoToggle = document.getElementById("auto-toggle");
  const autoEvery = document.getElementById("auto-every");
  const autoUnit = document.getElementById("auto-unit");
  const autoStatusTag = document.getElementById("auto-status-tag");
  let autoPollTimer = null;
  let lastSeenAutoRunAt = null;
  let uxState = {
    automated: !!autoToggle?.checked,
    auto_every: Number(autoEvery?.value || 5),
    auto_unit: autoUnit?.value || "minutes",
    auto_interval_sec: 300,
    interval_label: "5M",
    last_auto_run_at: null,
  };

  function intervalLabel(every, unit) {
    const short = { minutes: "M", hours: "H", days: "D" }[unit] || "M";
    return `${every}${short}`;
  }

  function applyUxState(state, { reschedule = true } = {}) {
    if (!state) return;
    const wasAutomated = !!uxState.automated;
    uxState = { ...uxState, ...state };
    if (autoToggle) autoToggle.checked = !!uxState.automated;
    if (autoEvery && uxState.auto_every != null) {
      autoEvery.value = String(uxState.auto_every);
    }
    if (autoUnit && uxState.auto_unit) {
      autoUnit.value = uxState.auto_unit;
    }
    const disabled = !uxState.automated;
    if (autoEvery) autoEvery.disabled = disabled;
    if (autoUnit) autoUnit.disabled = disabled;

    const label =
      uxState.interval_label ||
      intervalLabel(uxState.auto_every || 5, uxState.auto_unit || "minutes");
    if (autoStatusTag) {
      let tag = uxState.automated ? `ON · ${label}` : "AI ADVISOR";
      if (uxState.automated && state.running) tag = `RUN · ${label}`;
      autoStatusTag.textContent = tag;
      autoStatusTag.classList.toggle("is-on", !!uxState.automated);
    }
    // Only (re)arm the lightweight poller when automation toggles — not every status tick
    if (reschedule && wasAutomated !== !!uxState.automated) {
      scheduleAutoPoll();
    } else if (reschedule && uxState.automated && !autoPollTimer) {
      scheduleAutoPoll();
    } else if (!uxState.automated && autoPollTimer) {
      clearInterval(autoPollTimer);
      autoPollTimer = null;
    }
  }

  async function persistUxState(patch) {
    try {
      const res = await fetch("/api/ux", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      });
      const data = await res.json();
      if (res.ok) applyUxState(data);
    } catch (_) {
      /* keep local controls */
    }
  }

  function collectSchedulePatch(extra = {}) {
    return {
      automated: !!autoToggle?.checked,
      auto_every: Number(autoEvery?.value || 5),
      auto_unit: autoUnit?.value || "minutes",
      ...extra,
    };
  }

  function knownChatFingerprints() {
    const known = new Set();
    for (const m of chatHistory) {
      known.add(`${m.role}\0${m.content}`);
    }
    chatLog.querySelectorAll(".chat-msg").forEach((el) => {
      const role = el.classList.contains("user") ? "user" : "assistant";
      const raw =
        el.querySelector(".chat-text")?.dataset?.raw ||
        el.querySelector(".chat-text")?.textContent ||
        "";
      known.add(`${role}\0${raw}`);
    });
    return known;
  }

  async function syncAutoChatMessages() {
    try {
      const res = await fetch("/api/chat/history");
      if (!res.ok) return;
      const data = await res.json();
      const messages = data.messages || [];
      const known = knownChatFingerprints();
      let added = false;
      for (let i = 0; i < messages.length; i++) {
        const msg = messages[i];
        if (msg?.role !== "user" || !String(msg.content || "").startsWith("[AUTO]")) {
          continue;
        }
        const userKey = `user\0${msg.content}`;
        if (!known.has(userKey)) {
          appendMessage("user", msg.content);
          chatHistory.push({ role: "user", content: msg.content });
          known.add(userKey);
          added = true;
        }
        const next = messages[i + 1];
        if (next?.role === "assistant" && next.content) {
          const aKey = `assistant\0${next.content}`;
          if (!known.has(aKey)) {
            appendMessage("assistant", next.content);
            chatHistory.push({ role: "assistant", content: next.content });
            known.add(aKey);
            added = true;
          }
        }
      }
      if (added) scrollChatToBottom();
      if (data.brain) updateBrainMeta(data.brain);
    } catch (_) {
      /* ignore poll errors */
    }
  }

  async function pollAutoStatus() {
    if (!uxState.automated) return;
    try {
      const res = await fetch("/api/auto/status");
      if (!res.ok) return;
      const status = await res.json();
      applyUxState(status, { reschedule: false });
      const wrap = autoToggle?.closest(".auto-toggle");
      wrap?.classList.toggle("is-running", !!status.running);

      if (
        status.last_auto_run_at &&
        status.last_auto_run_at !== lastSeenAutoRunAt
      ) {
        const prev = lastSeenAutoRunAt;
        lastSeenAutoRunAt = status.last_auto_run_at;
        if (prev != null) {
          await syncAutoChatMessages();
          // Refresh holdings/thesis after a completed sweep (non-blocking)
          refreshLivePortfolio();
        }
      }
    } catch (_) {
      /* ignore */
    }
  }

  function scheduleAutoPoll() {
    if (autoPollTimer) {
      clearInterval(autoPollTimer);
      autoPollTimer = null;
    }
    if (!uxState.automated) return;
    // Lightweight status poll only — LLM/IB work stays on the server worker thread
    autoPollTimer = setInterval(pollAutoStatus, 8000);
    pollAutoStatus();
  }

  function onScheduleChanged() {
    const patch = collectSchedulePatch();
    applyUxState({
      ...patch,
      interval_label: intervalLabel(patch.auto_every, patch.auto_unit),
    });
    persistUxState(patch);
  }

  if (autoToggle) {
    autoToggle.addEventListener("change", onScheduleChanged);
  }
  if (autoEvery) {
    autoEvery.addEventListener("change", onScheduleChanged);
  }
  if (autoUnit) {
    autoUnit.addEventListener("change", onScheduleChanged);
  }

  // Load persisted UX on launch
  fetch("/api/ux")
    .then((r) => r.json())
    .then((state) => {
      lastSeenAutoRunAt = state.last_auto_run_at || null;
      applyUxState(state);
    })
    .catch(() => scheduleAutoPoll());

  // Persist on close / hide
  window.addEventListener("pagehide", () => {
    const payload = JSON.stringify(collectSchedulePatch());
    if (navigator.sendBeacon) {
      navigator.sendBeacon(
        "/api/ux",
        new Blob([payload], { type: "application/json" })
      );
    }
  });

  // --- Portfolio objectives edit / save ---------------------------------
  const objectivesPanel = document.getElementById("objectives-panel");
  const objectivesGrid = document.getElementById("objectives-grid");
  const objectivesStatus = document.getElementById("objectives-status");
  const objViewActions = document.getElementById("obj-view-actions");
  const objSaveActions = document.getElementById("obj-save-actions");
  const objEditBtn = document.getElementById("obj-edit-btn");
  const objCancelBtn = document.getElementById("obj-cancel-btn");
  const objSaveBtn = document.getElementById("obj-save-btn");

  function defaultObjStatus(savedAt) {
    return savedAt ? `SAVED ${savedAt}` : "RISK · GOALS · PREFERENCES";
  }

  let lastSavedAt = objectivesStatus?.textContent?.replace(/^SAVED\s+/, "") || null;
  if (lastSavedAt && lastSavedAt === "RISK · GOALS · PREFERENCES") lastSavedAt = null;

  function setObjectivesMode(mode) {
    objectivesGrid.dataset.mode = mode;
    objectivesPanel.classList.toggle("is-editing", mode === "edit");
    objViewActions.classList.toggle("is-hidden", mode === "edit");
    objSaveActions.classList.toggle("is-hidden", mode !== "edit");
  }

  function setObjectivesStatus(text, tone = "") {
    objectivesStatus.textContent = text;
    objectivesStatus.classList.remove("is-saved", "is-error");
    if (tone) objectivesStatus.classList.add(tone);
  }

  function parseTags(value) {
    return String(value || "")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
  }

  function renderTags(el, items) {
    const muted = (el.dataset.type || "").includes("muted");
    el.innerHTML = "";
    items.forEach((item) => {
      const tag = document.createElement("span");
      tag.className = muted ? "tag muted" : "tag";
      tag.textContent = item;
      el.appendChild(tag);
    });
  }

  function applyObjectivesToView(data) {
    objectivesGrid.querySelectorAll(".obj-value").forEach((el) => {
      const field = el.dataset.field;
      const value = data[field];
      if ((el.dataset.type || "").includes("tags")) {
        renderTags(el, Array.isArray(value) ? value : parseTags(value));
      } else {
        el.textContent = value ?? "";
      }
    });

    objectivesGrid.querySelectorAll(".obj-input").forEach((el) => {
      const field = el.dataset.field;
      const value = data[field];
      if (el.dataset.type === "tags") {
        el.value = Array.isArray(value) ? value.join(", ") : String(value || "");
      } else {
        el.value = value ?? "";
      }
    });

    if (data.saved_at) lastSavedAt = data.saved_at;
  }

  function collectObjectivesFromInputs() {
    const payload = {};
    objectivesGrid.querySelectorAll(".obj-input").forEach((el) => {
      const field = el.dataset.field;
      if (el.dataset.type === "tags") {
        payload[field] = parseTags(el.value);
      } else {
        payload[field] = el.value.trim();
      }
    });
    return payload;
  }

  function snapshotInputs() {
    const snap = {};
    objectivesGrid.querySelectorAll(".obj-input").forEach((el) => {
      snap[el.dataset.field] = el.value;
    });
    return snap;
  }

  let inputSnapshot = null;

  objEditBtn.addEventListener("click", () => {
    inputSnapshot = snapshotInputs();
    setObjectivesStatus("EDITING", "");
    setObjectivesMode("edit");
    const first = objectivesGrid.querySelector(".obj-input");
    if (first) first.focus();
  });

  objCancelBtn.addEventListener("click", () => {
    if (inputSnapshot) {
      objectivesGrid.querySelectorAll(".obj-input").forEach((el) => {
        el.value = inputSnapshot[el.dataset.field] ?? "";
      });
    }
    setObjectivesMode("view");
    setObjectivesStatus(defaultObjStatus(lastSavedAt), "");
  });

  objSaveBtn.addEventListener("click", async () => {
    const payload = collectObjectivesFromInputs();
    objSaveBtn.disabled = true;
    objCancelBtn.disabled = true;
    setObjectivesStatus("SAVING…", "");

    try {
      const res = await fetch("/api/objectives", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "save failed");

      applyObjectivesToView(data);
      setObjectivesMode("view");
      const hist = data.history_count != null ? ` · HIST ${data.history_count}/10` : "";
      setObjectivesStatus(`SAVED ${data.saved_at || ""}${hist}`, "is-saved");
      setTimeout(() => {
        if (objectivesStatus.classList.contains("is-saved")) {
          setObjectivesStatus(defaultObjStatus(data.saved_at || lastSavedAt), "");
        }
      }, 2200);

      fetch("/api/brain")
        .then((r) => r.json())
        .then(updateBrainMeta)
        .catch(() => {});
    } catch (err) {
      setObjectivesStatus(`SAVE FAILED: ${err.message || "error"}`, "is-error");
    } finally {
      objSaveBtn.disabled = false;
      objCancelBtn.disabled = false;
    }
  });
})();
