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

  // ---- case picker ------------------------------------------------------

  function renderPicker() {
    var picker = $("case-picker");
    picker.innerHTML = "";
    state.cases.forEach(function (c, i) {
      var btn = el("button", i === state.index ? "active" : "", c.title);
      btn.type = "button";
      btn.addEventListener("click", function () {
        state.index = i;
        state.phase = "before";
        renderPicker();
        renderCase();
      });
      picker.appendChild(btn);
    });
  }

  // ---- surfaces ---------------------------------------------------------

  function bubble(role, text, opts) {
    opts = opts || {};
    var b = el("div", "bubble " + role + (opts.flag ? " flag" : "") + (opts.error ? " error" : ""));
    b.appendChild(el("span", "bubble-text", text));
    if (opts.meta) b.appendChild(el("span", "meta", opts.meta));
    return b;
  }

  function renderChat(surface) {
    var wrap = el("div", "chat");
    surface.transcript.forEach(function (m) {
      var role = m.role === "user" ? "user" : m.role === "system" ? "system" : m.role === "agent" ? "agent" : "bot";
      var meta = m.role === "bot" ? "bot" : m.role === "agent" ? "human agent" : null;
      wrap.appendChild(bubble(role, m.text, { flag: m.flag, error: m.error, meta: meta }));
    });
    return wrap;
  }

  function renderWhatsapp(surface) {
    var wa = el("div", "wa");
    var head = el("div", "wa-head");
    head.appendChild(el("div", "wa-avatar", (surface.contact || "?")[0].toUpperCase()));
    var who = el("div");
    who.appendChild(el("div", "wa-contact", surface.contact));
    who.appendChild(el("div", "wa-sub", "online"));
    head.appendChild(who);
    wa.appendChild(head);
    surface.transcript.forEach(function (m, i) {
      var role = m.role === "user" ? "user" : m.role === "system" ? "system" : m.role === "agent" ? "agent" : "bot";
      wa.appendChild(bubble(role, m.text, { flag: m.flag, error: m.error, meta: "11:" + (52 + i) + " AM" }));
    });
    return wa;
  }

  function renderLeads(surface) {
    var list = el("div", "lead-list");
    surface.leads.forEach(function (lead) {
      var correct = lead.route === lead.should_be;
      var card = el("div", "lead " + (correct ? "ok" : "bad"));
      var top = el("div", "lead-top");
      top.appendChild(el("span", "lead-name", lead.name));
      card.appendChild(top);
      card.appendChild(el("div", "lead-co", lead.company));
      card.appendChild(el("div", "lead-msg", "\u201C" + lead.message + "\u201D"));
      card.appendChild(el("div", "lead-routes", "routed: " + lead.route + "  ·  should be: " + lead.should_be));
      card.appendChild(el("div", "lead-note", lead.note));
      list.appendChild(card);
    });
    return list;
  }

  function renderOutage(surface) {
    var list = el("div", "svc-list");
    if (surface.incident) {
      var banner = el("div", "incident " + surface.incident.level);
      banner.appendChild(el("div", "incident-label", surface.incident.label));
      banner.appendChild(el("div", "incident-note", surface.incident.note));
      list.appendChild(banner);
    }
    surface.services.forEach(function (s) {
      var row = el("div", "svc " + s.status);
      row.appendChild(el("span", "svc-dot"));
      var mid = el("div");
      mid.appendChild(el("div", "svc-name", s.name));
      mid.appendChild(el("div", "svc-detail", s.detail));
      row.appendChild(mid);
      row.appendChild(el("span", "cat-chip", s.failure_category));
      list.appendChild(row);
    });
    return list;
  }

  function renderModeration(surface) {
    var list = el("div", "mod-list");
    surface.queue.forEach(function (item) {
      var card = el("div", "mod");
      card.appendChild(el("div", "mod-author", item.author));
      card.appendChild(el("div", "mod-text", "\u201C" + item.text + "\u201D"));
      var foot = el("div", "mod-foot");
      foot.appendChild(el("span", "lead-routes", "bot: " + item.route));
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

  // ---- analysis ---------------------------------------------------------

  function gauge(entropy, stability) {
    var wrap = el("div", "gauge-wrap");
    var svgNs = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(svgNs, "svg");
    svg.setAttribute("viewBox", "0 0 120 120");
    svg.setAttribute("class", "gauge");
    var track = document.createElementNS(svgNs, "circle");
    track.setAttribute("cx", "60"); track.setAttribute("cy", "60"); track.setAttribute("r", "52");
    track.setAttribute("class", "gauge-track");
    var value = document.createElementNS(svgNs, "circle");
    value.setAttribute("cx", "60"); value.setAttribute("cy", "60"); value.setAttribute("r", "52");
    value.setAttribute("class", "gauge-value");
    value.style.stroke = "var(--" + stability + ")";
    value.style.strokeDashoffset = String(CIRCUMFERENCE * (1 - entropy));
    svg.appendChild(track); svg.appendChild(value);
    wrap.appendChild(svg);
    var center = el("div", "gauge-center");
    center.appendChild(el("span", "gauge-score", entropy.toFixed(2)));
    center.appendChild(el("span", "gauge-label", "entropy"));
    wrap.appendChild(center);
    return wrap;
  }

  function variantTable(originalRoute, variantRoutes) {
    var table = el("table", "attr-table");
    var thead = el("thead");
    var hr = el("tr");
    ["Variant", "Route", "Changed?"].forEach(function (h) { hr.appendChild(el("th", null, h)); });
    thead.appendChild(hr); table.appendChild(thead);
    var tbody = el("tbody");
    VARIANTS.forEach(function (key) {
      var route = variantRoutes[key];
      var changed = route !== originalRoute;
      var tr = el("tr", changed ? "flip" : "");
      tr.appendChild(el("td", null, VARIANT_LABELS[key]));
      var td = el("td"); td.appendChild(el("code", null, route)); tr.appendChild(td);
      tr.appendChild(el("td", null, changed ? "yes" : "no"));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return table;
  }

  function renderRoutingAnalysis(c, phase) {
    var a = c.analysis[phase];
    var body = el("div", "analysis-body");

    var top = el("div", "result-top");
    top.appendChild(gauge(a.entropy_score, a.stability));
    var verdict = el("div", "verdict");
    verdict.appendChild(el("span", "badge badge-" + a.stability, a.stability));
    var cf = el("p", "counterfactual");
    if (a.counterfactual_route) {
      cf.textContent = "Remove the driving dimension and the agent takes the other path: " + a.counterfactual_route + ".";
    } else {
      cf.textContent = "No variant flips the route — the decision is stable across all three.";
    }
    verdict.appendChild(cf);
    top.appendChild(verdict);
    body.appendChild(top);

    var attrHead = el("div", "analysis-head");
    attrHead.appendChild(el("span", null, "Attribution"));
    attrHead.appendChild(el("span", null, "opposite > neutral > similar"));
    body.appendChild(attrHead);
    body.appendChild(variantTable(a.route, a.variant_routes));
    var driver = el("p", "attr-driver");
    driver.appendChild(el("span", null, "First flip: "));
    driver.appendChild(el("strong", null, a.attribution_dimension || "none detected"));
    body.appendChild(driver);

    if (phase === "before" && c.alerts) {
      var alertsHead = el("div", "analysis-head");
      alertsHead.appendChild(el("span", null, "Failure signals"));
      body.appendChild(alertsHead);
      var ul = el("ul", "alerts");
      c.alerts.forEach(function (al) {
        var li = el("li");
        li.appendChild(el("code", null, al.node_id + " · " + al.failure_category));
        li.appendChild(el("div", null, al.detail));
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
      verdict.appendChild(
        el("p", "counterfactual", c.fix.after_note || "The incident is classified and handled.")
      );
      ok.appendChild(verdict);
      body.appendChild(ok);
      body.appendChild(costLine(c, phase));
      return body;
    }

    var d = c.analysis;
    var top = el("div", "result-top");
    var v = el("div", "verdict");
    v.appendChild(el("span", "badge badge-fragile", d.failure_category));
    var em = el("p", "counterfactual");
    em.appendChild(el("strong", null, d.error_type || "error"));
    em.appendChild(el("span", null, " — " + (d.error_message || "")));
    v.appendChild(em);
    top.appendChild(v);
    body.appendChild(top);

    var head = el("div", "analysis-head");
    head.appendChild(el("span", null, "Affected nodes"));
    head.appendChild(el("span", null, "classified from SDK signals"));
    body.appendChild(head);

    var table = el("table", "attr-table");
    var thead = el("thead");
    var hr = el("tr");
    ["Node", "Failure", "Events"].forEach(function (h) { hr.appendChild(el("th", null, h)); });
    thead.appendChild(hr); table.appendChild(thead);
    var tbody = el("tbody");
    d.affected_nodes.forEach(function (n) {
      var tr = el("tr");
      var td0 = el("td"); td0.appendChild(el("code", null, n.node_id)); tr.appendChild(td0);
      tr.appendChild(el("td", null, n.failure_category));
      tr.appendChild(el("td", null, String(n.count)));
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
      if (wt) parts.push("estimated incident waste: " + wt + (wc ? " / " + wc : ""));
    }
    if (cost.analysis_overhead) {
      var ac = fmtUsd(cost.analysis_overhead.cost_usd);
      var at = fmtTokens(cost.analysis_overhead.total_tokens);
      parts.push("this analysis itself cost " + (ac || at) + " — the observer's bill is measured too");
    }
    wrap.textContent = parts.join(" · ");
    return wrap;
  }

  // ---- fix --------------------------------------------------------------

  function renderFix(c) {
    var wrap = el("div", "fix");
    var toggle = el("div", "fix-toggle");
    var beforeBtn = el("button", state.phase === "before" ? "active" : "", "Incident");
    var afterBtn = el("button", state.phase === "after" ? "active" : "", "After fix");
    beforeBtn.type = "button"; afterBtn.type = "button";
    beforeBtn.addEventListener("click", function () { state.phase = "before"; renderCase(); });
    afterBtn.addEventListener("click", function () { state.phase = "after"; renderCase(); });
    toggle.appendChild(beforeBtn); toggle.appendChild(afterBtn);
    wrap.appendChild(toggle);

    wrap.appendChild(el("p", "fix-summary", c.fix.summary));

    if (c.delta) {
      var delta = el("div", "fix-delta");
      var d1 = el("span");
      d1.appendChild(el("span", null, "entropy: "));
      d1.appendChild(el("b", null, c.delta.entropy_before.toFixed(2) + " → " + c.delta.entropy_after.toFixed(2)));
      delta.appendChild(d1);
      var d2 = el("span");
      d2.appendChild(el("span", null, "stability: "));
      d2.appendChild(el("b", null, c.delta.stability_before + " → " + c.delta.stability_after));
      delta.appendChild(d2);
      wrap.appendChild(delta);
    }
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

  // ---- case render ------------------------------------------------------

  function renderCase() {
    var c = state.cases[state.index];
    var view = $("case-view");
    view.innerHTML = "";

    var head = el("div", "case-head");
    head.appendChild(el("h3", null, c.title));
    head.appendChild(el("div", "case-company", c.company + " · " + c.surface.channel_label));
    head.appendChild(el("p", "case-tagline", c.tagline));
    head.appendChild(el("p", "case-incident", c.incident));
    view.appendChild(head);

    var grid = el("div", "case-grid");

    var surfaceData = state.phase === "after" && c.surface_after ? c.surface_after : c.surface;
    var surface = el("div", "surface");
    var sHead = el("div", "surface-head");
    sHead.appendChild(el("span", null, state.phase === "after" ? "What the user sees after the fix" : "What the user saw"));
    sHead.appendChild(el("span", null, surfaceData.channel_label));
    surface.appendChild(sHead);
    var sBody = el("div", "surface-body");
    sBody.appendChild(renderSurface(surfaceData));
    surface.appendChild(sBody);
    grid.appendChild(surface);

    var analysis = el("div", "analysis");
    var aHead = el("div", "analysis-head");
    aHead.appendChild(el("span", null, "What Conntrail saw"));
    aHead.appendChild(el("span", null, state.phase === "after" ? "after fix" : "at incident"));
    analysis.appendChild(aHead);
    analysis.appendChild(renderAnalysis(c, state.phase));
    grid.appendChild(analysis);

    view.appendChild(grid);
    view.appendChild(renderFix(c));

    var sources = renderSources(c);
    if (sources) view.appendChild(sources);
  }

  // ---- contact form -----------------------------------------------------

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
      status.textContent = "Sending…";
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
            status.textContent = "Thanks — we'll be in touch by email.";
          } else if (res.status === 503) {
            status.className = "form-status err";
            status.textContent = "The form isn't connected yet — please email hello@ibzie.dev.";
          } else {
            status.className = "form-status err";
            status.textContent = res.body.error || "Something went wrong. Please email hello@ibzie.dev.";
          }
        })
        .catch(function () {
          status.className = "form-status err";
          status.textContent = "Network error — please email hello@ibzie.dev.";
        })
        .finally(function () { submit.disabled = false; });
    });
  }

  // ---- boot -------------------------------------------------------------

  function initDemo() {
    var view = $("case-view");
    if (!$("case-picker") || !view) return;
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
      if (href === here) links[i].classList.add("active");
    }
  }

  markActiveNav();
  initDemo();
  initContact();
})();
