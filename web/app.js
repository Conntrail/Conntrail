/*
 * Conntrail static demo.
 *
 * Loads precomputed use cases from data/demo.json and renders each incident's
 * simulated surface alongside Conntrail's detection. No LLM calls, no server,
 * no signup: every entropy / attribution / failure figure was measured when the
 * fixtures were frozen (offline mode derives them with the SDK's own formulas).
 */
(function () {
  "use strict";

  var CIRCUMFERENCE = 2 * Math.PI * 52;
  var VARIANTS = ["similar", "neutral", "opposite"];
  var VARIANT_LABELS = { similar: "Similar", neutral: "Neutral", opposite: "Opposite" };

  var state = { cases: [], index: 0, phase: "before" };

  function $(id) { return document.getElementById(id); }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function fmtUsd(usd) {
    if (usd === null || usd === undefined) return null;
    return "$" + Number(usd).toFixed(5);
  }

  function fmtTokens(n) {
    if (n === null || n === undefined) return null;
    return Number(n).toLocaleString() + " tokens";
  }

  // ---- icons --------------------------------------------------------------

  var ICONS = {
    user: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 3.6-6.5 8-6.5s8 2.5 8 6.5"/></svg>',
    radar: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="1.2" fill="currentColor" stroke="none"/><path d="M12 12l6.4-6.4"/></svg>',
    check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12.5l5 5L20 6.5"/></svg>',
    flip: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 4v13M7 4L3.5 7.5M7 4l3.5 3.5"/><path d="M17 20V7m0 13l3.5-3.5M17 20l-3.5-3.5"/></svg>',
    arrow: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>',
    alert: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3.5l9.5 16.5H2.5z"/><path d="M12 10v4.5"/><circle cx="12" cy="17.6" r="0.4" fill="currentColor"/></svg>',
    bolt: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2L4.5 13.5H11L10 22l8.5-11.5H13z"/></svg>'
  };

  function icon(name) {
    var span = document.createElement("span");
    span.className = "icon";
    span.setAttribute("aria-hidden", "true");
    span.innerHTML = ICONS[name];
    return span;
  }

  // ---- case picker (tabs) -------------------------------------------------

  function renderPicker() {
    var picker = $("case-picker");
    picker.innerHTML = "";
    state.cases.forEach(function (c, i) {
      var btn = el("button", i === state.index ? "active" : "");
      btn.type = "button";
      btn.setAttribute("role", "tab");
      btn.id = "case-tab-" + i;
      btn.setAttribute("aria-selected", i === state.index ? "true" : "false");
      btn.setAttribute("tabindex", i === state.index ? "0" : "-1");
      btn.appendChild(el("span", "idx", "0" + (i + 1)));
      btn.appendChild(el("span", null, c.title));
      btn.addEventListener("click", function () { selectCase(i); });
      btn.addEventListener("keydown", function (e) { onPickerKey(e, i); });
      picker.appendChild(btn);
    });
  }

  function selectCase(i) {
    state.index = i;
    state.phase = "before";
    renderPicker();
    renderCase();
  }

  function onPickerKey(e, i) {
    var next = null;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") next = (i + 1) % state.cases.length;
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = (i - 1 + state.cases.length) % state.cases.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = state.cases.length - 1;
    if (next === null) return;
    e.preventDefault();
    selectCase(next);
    var tab = $("case-tab-" + next);
    if (tab) tab.focus();
  }

  // ---- surfaces -----------------------------------------------------------

  function bubble(role, text, opts) {
    opts = opts || {};
    var b = el("div", "bubble " + role + (opts.flag ? " flag" : "") + (opts.error ? " error" : ""));
    b.appendChild(el("span", "bubble-text", text));
    if (opts.meta) b.appendChild(el("span", "meta", opts.meta));
    return b;
  }

  function chatTime(i) {
    return "14:" + String(2 + i).padStart(2, "0");
  }

  function msgRow(role, text, opts) {
    opts = opts || {};
    var row = el("div", "msg-row " + (role === "user" ? "user" : "other"));
    if (role === "bot" || role === "agent") {
      var av = el("span", "avatar " + role, role === "bot" ? "B" : "M");
      av.setAttribute("aria-hidden", "true");
      row.appendChild(av);
    }
    row.appendChild(bubble(role, text, opts));
    return row;
  }

  function renderChat(surface) {
    var wrap = el("div", "chat");
    surface.transcript.forEach(function (m, i) {
      if (m.role === "system") {
        wrap.appendChild(bubble("system", m.text, { error: m.error }));
        return;
      }
      var role = m.role === "user" ? "user" : m.role === "agent" ? "agent" : "bot";
      var who = m.role === "bot" ? "bot" : m.role === "agent" ? "human agent" : null;
      var meta = who ? who + " · " + chatTime(i) : chatTime(i);
      wrap.appendChild(msgRow(role, m.text, { flag: m.flag, error: m.error, meta: meta }));
    });
    return wrap;
  }

  function renderWhatsapp(surface) {
    var wa = el("div", "wa");
    var head = el("div", "wa-head");
    var av = el("div", "wa-avatar", (surface.contact || "?")[0].toUpperCase());
    av.setAttribute("aria-hidden", "true");
    head.appendChild(av);
    var who = el("div");
    who.appendChild(el("div", "wa-contact", surface.contact));
    who.appendChild(el("div", "wa-sub", "online"));
    head.appendChild(who);
    wa.appendChild(head);

    var thread = el("div", "chat");
    surface.transcript.forEach(function (m, i) {
      if (m.role === "system") {
        thread.appendChild(bubble("system", m.text, { error: m.error }));
        return;
      }
      var role = m.role === "user" ? "user" : m.role === "agent" ? "agent" : "bot";
      var meta = "11:" + (52 + i) + " AM" + (role === "user" ? " ✓✓" : "");
      // whatsapp threads have no per-message avatar; the header carries it
      var row = el("div", "msg-row " + (role === "user" ? "user" : "other"));
      row.appendChild(bubble(role, m.text, { flag: m.flag, error: m.error, meta: meta }));
      thread.appendChild(row);
    });
    wa.appendChild(thread);
    return wa;
  }

  function renderLeads(surface) {
    var list = el("div", "lead-list");
    surface.leads.forEach(function (lead) {
      var correct = lead.route === lead.should_be;
      var card = el("div", "lead " + (correct ? "ok" : "bad"));
      var top = el("div", "lead-top");
      top.appendChild(el("span", "lead-name", lead.name));
      top.appendChild(el("span", "lead-verdict", correct ? "correct" : "misrouted"));
      card.appendChild(top);
      card.appendChild(el("div", "lead-co", lead.company));
      card.appendChild(el("div", "lead-msg", "“" + lead.message + "”"));
      var routes = el("div", "lead-routes");
      routes.appendChild(el("span", "chip", lead.route));
      routes.appendChild(icon("arrow"));
      routes.appendChild(el("span", "chip", lead.should_be));
      card.appendChild(routes);
      card.appendChild(el("div", "lead-note", lead.note));
      list.appendChild(card);
    });
    return list;
  }

  function renderOutage(surface) {
    var list = el("div", "svc-list");
    if (surface.incident) {
      var banner = el("div", "incident " + surface.incident.level);
      banner.appendChild(icon(surface.incident.level === "critical" ? "alert" : "bolt"));
      var body = el("div");
      body.appendChild(el("div", "incident-label", surface.incident.label));
      body.appendChild(el("div", "incident-note", surface.incident.note));
      banner.appendChild(body);
      list.appendChild(banner);
    }
    surface.services.forEach(function (s) {
      var row = el("div", "svc " + s.status);
      var dot = el("span", "svc-dot");
      dot.setAttribute("aria-hidden", "true");
      row.appendChild(dot);
      var mid = el("div");
      mid.appendChild(el("div", "svc-name", s.name));
      mid.appendChild(el("div", "svc-detail", s.detail));
      row.appendChild(mid);
      row.appendChild(el("span", "cat-chip cat-" + s.failure_category, s.failure_category));
      list.appendChild(row);
    });
    return list;
  }

  function renderModeration(surface) {
    var list = el("div", "mod-list");
    surface.queue.forEach(function (item) {
      var card = el("div", "mod");
      card.appendChild(el("div", "mod-author", item.author));
      card.appendChild(el("div", "mod-text", "“" + item.text + "”"));
      var foot = el("div", "mod-foot");
      foot.appendChild(el("span", "chip", item.route));
      foot.appendChild(el("span", "lead-note", item.note));
      card.appendChild(foot);
      list.appendChild(card);
    });
    return list;
  }

  function renderSurface(surface) {
    switch (surface.kind) {
      case "web_chat": return renderChat(surface);
      case "whatsapp": return renderWhatsapp(surface);
      case "leads": return renderLeads(surface);
      case "outage": return renderOutage(surface);
      case "moderation": return renderModeration(surface);
      default: return el("p", "subtitle", "No surface for this case.");
    }
  }

  // ---- analysis -----------------------------------------------------------

  var SVG_NS = "http://www.w3.org/2000/svg";

  function polar(fraction, r) {
    var a = fraction * 2 * Math.PI;
    return [60 + r * Math.cos(a), 60 + r * Math.sin(a)];
  }

  function svgLine(fraction, r1, r2, cls) {
    var p1 = polar(fraction, r1);
    var p2 = polar(fraction, r2);
    var line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", p1[0].toFixed(1));
    line.setAttribute("y1", p1[1].toFixed(1));
    line.setAttribute("x2", p2[0].toFixed(1));
    line.setAttribute("y2", p2[1].toFixed(1));
    line.setAttribute("class", cls);
    return line;
  }

  function gauge(entropy, stability) {
    var wrap = el("div", "gauge-wrap");
    var svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", "0 0 120 120");
    svg.setAttribute("class", "gauge");
    svg.setAttribute("aria-hidden", "true");

    var track = document.createElementNS(SVG_NS, "circle");
    track.setAttribute("cx", "60"); track.setAttribute("cy", "60"); track.setAttribute("r", "52");
    track.setAttribute("class", "gauge-track");
    svg.appendChild(track);

    // major ticks at 0 / .25 / .5 / .75, threshold markers at .25 and .60
    [0, 0.25, 0.5, 0.75].forEach(function (f) { svg.appendChild(svgLine(f, 55, 59.5, "gauge-tick")); });
    [0.25, 0.6].forEach(function (f) { svg.appendChild(svgLine(f, 43, 49, "gauge-threshold")); });

    if (entropy > 0) {
      var value = document.createElementNS(SVG_NS, "circle");
      value.setAttribute("cx", "60"); value.setAttribute("cy", "60"); value.setAttribute("r", "52");
      value.setAttribute("class", "gauge-value");
      value.style.stroke = "var(--" + stability + ")";
      svg.appendChild(value);
      requestAnimationFrame(function () {
        requestAnimationFrame(function () {
          value.style.strokeDashoffset = String(CIRCUMFERENCE * (1 - entropy));
        });
      });
    } else {
      var p = polar(0, 52);
      var dot = document.createElementNS(SVG_NS, "circle");
      dot.setAttribute("cx", p[0]);
      dot.setAttribute("cy", p[1]);
      dot.setAttribute("r", "3.2");
      dot.setAttribute("class", "gauge-zero-dot");
      dot.style.fill = "var(--" + stability + ")";
      svg.appendChild(dot);
    }

    wrap.appendChild(svg);
    var center = el("div", "gauge-center");
    center.appendChild(el("span", "gauge-score", entropy.toFixed(2)));
    center.appendChild(el("span", "gauge-label", "entropy"));
    wrap.appendChild(center);
    return wrap;
  }

  function counterfactual(label, textNode) {
    var cf = el("p", "counterfactual");
    cf.appendChild(el("span", "cf-label", label));
    cf.appendChild(textNode);
    return cf;
  }

  function variantTable(originalRoute, variantRoutes) {
    var table = el("table", "attr-table");
    var thead = el("thead");
    var hr = el("tr");
    ["Variant", "Route", "Result"].forEach(function (h) { hr.appendChild(el("th", null, h)); });
    thead.appendChild(hr); table.appendChild(thead);
    var tbody = el("tbody");
    VARIANTS.forEach(function (key) {
      var route = variantRoutes[key];
      var changed = route !== originalRoute;
      var tr = el("tr", changed ? "flip" : "");
      tr.appendChild(el("td", null, VARIANT_LABELS[key]));
      var td = el("td"); td.appendChild(el("code", null, route)); tr.appendChild(td);
      var res = el("span", "res " + (changed ? "res-flip" : "res-hold"));
      res.appendChild(icon(changed ? "flip" : "check"));
      res.appendChild(el("span", null, changed ? "flipped" : "holds"));
      var tdRes = el("td"); tdRes.appendChild(res); tr.appendChild(tdRes);
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return table;
  }

  function panelHead(label, iconName, tag) {
    var head = el("div", "analysis-head");
    var left = el("span", "h-label");
    left.appendChild(icon(iconName));
    left.appendChild(el("span", null, label));
    head.appendChild(left);
    if (tag) head.appendChild(el("span", "h-tag", tag));
    return head;
  }

  function renderRoutingAnalysis(c, phase) {
    var a = c.analysis[phase];
    var body = el("div", "analysis-body");

    var top = el("div", "result-top");
    top.appendChild(gauge(a.entropy_score, a.stability));
    var verdict = el("div", "verdict");
    verdict.appendChild(el("span", "badge badge-" + a.stability, a.stability));
    if (a.counterfactual_route) {
      var cfText = el("span");
      cfText.textContent = "Remove the driving dimension and the agent takes the other path: ";
      cfText.appendChild(el("code", null, a.counterfactual_route));
      cfText.appendChild(document.createTextNode("."));
      verdict.appendChild(counterfactual("Counterfactual", cfText));
    } else {
      verdict.appendChild(counterfactual(
        "Stable",
        el("span", null, "No variant flips the route. The decision holds across all three perturbations.")
      ));
    }
    top.appendChild(verdict);
    body.appendChild(top);

    body.appendChild(panelHead("Attribution", "radar", "priority: opposite, neutral, similar"));
    body.appendChild(variantTable(a.route, a.variant_routes));
    var driver = el("p", "attr-driver");
    driver.appendChild(el("span", null, "First flip: "));
    driver.appendChild(el("strong", null, a.attribution_dimension || "none detected"));
    body.appendChild(driver);

    if (phase === "before" && c.alerts) {
      body.appendChild(panelHead("Failure signals", "alert", null));
      var ul = el("ul", "alerts");
      c.alerts.forEach(function (al) {
        var li = el("li");
        var head = el("div", "alert-head");
        head.appendChild(el("code", null, al.node_id));
        head.appendChild(el("span", "cat-chip cat-" + al.failure_category, al.failure_category));
        li.appendChild(head);
        li.appendChild(el("div", "alert-detail", al.detail));
        ul.appendChild(li);
      });
      body.appendChild(ul);
    }

    body.appendChild(costLine(c, phase));
    return body;
  }

  function renderFailureAnalysis(c, phase) {
    var body = el("div", "analysis-body");
    if (phase === "after") {
      var label = c.fix.after_label || "recovered";
      var cls = label === "recovered" ? "confident" : "boundary";
      var ok = el("div", "result-top");
      var verdict = el("div", "verdict");
      verdict.appendChild(el("span", "badge badge-" + cls, label));
      verdict.appendChild(counterfactual(
        "Outcome",
        el("span", null, c.fix.after_note || "The incident is classified and handled.")
      ));
      ok.appendChild(verdict);
      body.appendChild(ok);
      body.appendChild(costLine(c, phase));
      return body;
    }

    var d = c.analysis;
    var top = el("div", "result-top");
    var v = el("div", "verdict");
    v.appendChild(el("span", "badge badge-fragile", d.failure_category));
    var err = el("span");
    var strong = el("strong", null, d.error_type || "error");
    err.appendChild(strong);
    err.appendChild(document.createTextNode(": " + (d.error_message || "")));
    v.appendChild(counterfactual("Error", err));
    top.appendChild(v);
    body.appendChild(top);

    body.appendChild(panelHead("Affected nodes", "radar", "classified from SDK signals"));

    var table = el("table", "attr-table");
    var thead = el("thead");
    var hr = el("tr");
    ["Node", "Failure", "Events"].forEach(function (h) { hr.appendChild(el("th", null, h)); });
    thead.appendChild(hr); table.appendChild(thead);
    var tbody = el("tbody");
    d.affected_nodes.forEach(function (n) {
      var tr = el("tr");
      var td0 = el("td"); td0.appendChild(el("code", null, n.node_id)); tr.appendChild(td0);
      var td1 = el("td"); td1.appendChild(el("span", "cat-chip cat-" + n.failure_category, n.failure_category)); tr.appendChild(td1);
      var td2 = el("td", null, String(n.count)); td2.style.fontFamily = "var(--font-mono)"; tr.appendChild(td2);
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    body.appendChild(table);

    body.appendChild(costLine(c, phase));
    return body;
  }

  function renderAnalysis(c, phase) {
    if (c.detection.kind === "routing") return renderRoutingAnalysis(c, phase);
    return renderFailureAnalysis(c, phase);
  }

  function costLine(c, phase) {
    var wrap = el("p", "cost-line");
    var cost = c.cost || {};
    var parts = [];
    if (phase === "before" && cost.incident) {
      var wt = fmtTokens(cost.incident.wasted_tokens);
      var wc = fmtUsd(cost.incident.wasted_cost_usd);
      if (wt) parts.push("incident waste: " + wt + (wc ? " / " + wc : ""));
    }
    if (cost.analysis_overhead) {
      var ac = fmtUsd(cost.analysis_overhead.cost_usd);
      var at = fmtTokens(cost.analysis_overhead.total_tokens);
      parts.push("this analysis cost " + (ac || at) + ". The observer's bill is measured too");
    }
    wrap.textContent = parts.join("  ·  ");
    return wrap;
  }

  // ---- fix ----------------------------------------------------------------

  function deltaChip(value, label, cls) {
    var chip = el("span", "delta-chip delta-" + cls);
    chip.appendChild(el("b", null, value));
    if (label) chip.appendChild(el("span", "d-label", label));
    return chip;
  }

  function renderFix(c) {
    var wrap = el("div", "fix");
    var top = el("div", "fix-top");

    var toggle = el("div", "fix-toggle" + (state.phase === "after" ? " after" : ""));
    toggle.setAttribute("role", "group");
    toggle.setAttribute("aria-label", "Incident or after fix");
    var beforeBtn = el("button", state.phase === "before" ? "active" : "", "Incident");
    var afterBtn = el("button", state.phase === "after" ? "active" : "", "After fix");
    beforeBtn.type = "button"; afterBtn.type = "button";
    beforeBtn.setAttribute("aria-pressed", state.phase === "before" ? "true" : "false");
    afterBtn.setAttribute("aria-pressed", state.phase === "after" ? "true" : "false");
    beforeBtn.addEventListener("click", function () { state.phase = "before"; renderCase(); });
    afterBtn.addEventListener("click", function () { state.phase = "after"; renderCase(); });
    toggle.appendChild(beforeBtn); toggle.appendChild(afterBtn);
    top.appendChild(toggle);

    if (c.delta) {
      var delta = el("div", "fix-delta");
      delta.appendChild(deltaChip(c.delta.entropy_before.toFixed(2), c.delta.stability_before, c.delta.stability_before));
      delta.appendChild(icon("arrow"));
      delta.appendChild(deltaChip(c.delta.entropy_after.toFixed(2), c.delta.stability_after, c.delta.stability_after));
      top.appendChild(delta);
    } else if (c.surface.incident && c.surface_after && c.surface_after.incident) {
      var ack = el("div", "fix-delta");
      ack.appendChild(deltaChip(c.surface.incident.label, null, "fragile"));
      ack.appendChild(icon("arrow"));
      ack.appendChild(deltaChip(c.surface_after.incident.label, null, "boundary"));
      top.appendChild(ack);
    }
    wrap.appendChild(top);

    wrap.appendChild(el("p", "fix-summary", c.fix.summary));
    return wrap;
  }

  function renderSources(c) {
    if (!c.source_refs || !c.source_refs.length) return null;
    var wrap = el("div", "source-refs");
    wrap.appendChild(el("span", null, "Pattern seen in:"));
    var ul = el("ul");
    c.source_refs.forEach(function (ref) {
      var li = el("li");
      var a = el("a", null, ref.label);
      a.href = ref.url; a.rel = "noopener"; a.target = "_blank";
      li.appendChild(a);
      ul.appendChild(li);
    });
    wrap.appendChild(ul);
    return wrap;
  }

  // ---- case render --------------------------------------------------------

  function renderCase() {
    var c = state.cases[state.index];
    var view = $("case-view");
    view.innerHTML = "";

    var head = el("div", "case-head");
    var meta = el("div", "case-meta");
    meta.appendChild(el("span", null, "Case 0" + (state.index + 1) + " / 0" + state.cases.length));
    meta.appendChild(el("span", null, c.company));
    meta.appendChild(el("span", null, c.surface.channel_label));
    if (c.detection.node_id) {
      var nodeChip = el("span", "chip", c.detection.node_id);
      meta.appendChild(nodeChip);
    }
    head.appendChild(meta);
    head.appendChild(el("h3", null, c.title));
    head.appendChild(el("p", "case-tagline", c.tagline));
    head.appendChild(el("p", "case-incident", c.incident));
    view.appendChild(head);

    var grid = el("div", "case-grid");

    var surfaceData = state.phase === "after" && c.surface_after ? c.surface_after : c.surface;
    var surface = el("div", "surface");
    var sHead = el("div", "surface-head");
    var sLabel = el("span", "h-label");
    sLabel.appendChild(icon("user"));
    sLabel.appendChild(el("span", null, state.phase === "after" ? "What the user sees now" : "What the user saw"));
    sHead.appendChild(sLabel);
    sHead.appendChild(el("span", "h-tag", surfaceData.channel_label));
    surface.appendChild(sHead);
    var sBody = el("div", "surface-body");
    sBody.appendChild(renderSurface(surfaceData));
    surface.appendChild(sBody);
    grid.appendChild(surface);

    var analysis = el("div", "analysis");
    var aHead = el("div", "analysis-head");
    var aLabel = el("span", "h-label");
    aLabel.appendChild(icon("radar"));
    aLabel.appendChild(el("span", null, "What Conntrail saw"));
    aHead.appendChild(aLabel);
    aHead.appendChild(el("span", "h-tag", state.phase === "after" ? "after fix" : "at incident"));
    analysis.appendChild(aHead);
    analysis.appendChild(renderAnalysis(c, state.phase));
    grid.appendChild(analysis);

    view.appendChild(grid);
    view.appendChild(renderFix(c));

    var sources = renderSources(c);
    if (sources) view.appendChild(sources);

    // retrigger the entrance transition (suppressed under reduced motion)
    view.classList.remove("enter");
    void view.offsetWidth;
    view.classList.add("enter");
  }

  // ---- contact form -------------------------------------------------------

  function initContact() {
    var form = $("contact-form");
    if (!form) return;
    var status = $("form-status");
    var submit = $("contact-submit");

    function field(name) {
      var control = form.elements.namedItem(name);
      return control ? control.value.trim() : "";
    }

    form.addEventListener("submit", function (event) {
      event.preventDefault();
      status.className = "form-status";
      status.textContent = "";

      var data = {
        name: field("name"),
        email: field("email"),
        company: field("company"),
        message: field("message"),
        website: field("website")
      };
      if (!data.name || !data.email || !data.message) {
        status.className = "form-status err";
        status.textContent = "Please add your name, email and a short message.";
        return;
      }
      if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(data.email)) {
        status.className = "form-status err";
        status.textContent = "That email doesn't look right.";
        return;
      }

      submit.disabled = true;
      status.textContent = "Sending...";
      fetch("/api/contact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      })
        .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, status: r.status, body: j }; }); })
        .then(function (res) {
          if (res.ok && res.body.ok) {
            form.reset();
            status.className = "form-status ok";
            status.textContent = "Thanks. We'll be in touch by email.";
          } else if (res.status === 503) {
            status.className = "form-status err";
            status.textContent = "The form is not connected yet. Please email hello@ibzie.dev.";
          } else {
            status.className = "form-status err";
            status.textContent = res.body.error || "Something went wrong. Please email hello@ibzie.dev.";
          }
        })
        .catch(function () {
          status.className = "form-status err";
          status.textContent = "Network error. Please email hello@ibzie.dev.";
        })
        .finally(function () { submit.disabled = false; });
    });
  }

  // ---- boot ---------------------------------------------------------------

  function initDemo() {
    var view = $("case-view");
    if (!$("case-picker") || !view) return;
    view.setAttribute("role", "tabpanel");
    fetch("data/demo.json")
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (bundle) {
        state.cases = bundle.use_cases || [];
        if (!state.cases.length) throw new Error("no use cases");
        renderPicker();
        renderCase();
      })
      .catch(function (err) {
        view.innerHTML = "";
        view.appendChild(el("p", "subtitle", "Could not load the precomputed fixtures (" + err.message + ")."));
      });
  }

  function markActiveNav() {
    var here = window.location.pathname.split("/").pop() || "index.html";
    var links = document.querySelectorAll(".topbar nav a");
    for (var i = 0; i < links.length; i++) {
      var href = links[i].getAttribute("href");
      if (href === here) {
        links[i].classList.add("active");
        links[i].setAttribute("aria-current", "page");
      }
    }
  }

  markActiveNav();
  initDemo();
  initContact();
})();
