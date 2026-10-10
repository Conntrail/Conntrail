/*
 * Live demo page — runs the real Conntrail analysis against free models.
 * Reuses the render helpers exposed by app.js (window.ConntrailUI).
 */
(function () {
  "use strict";

  var UI = window.ConntrailUI;
  var form = document.getElementById("live-form");
  if (!form || !UI) return;

  function $(id) { return document.getElementById(id); }

  var EXAMPLES = [
    "This is the third time I've asked. I want my money back now, or I'm calling my lawyer.",
    "Can I return an unopened item I bought 5 days ago?",
    "Where is my order? It was supposed to arrive last week and I'm getting frustrated.",
    "I'm reporting you to the consumer protection agency unless someone senior responds today."
  ];

  function apiBase() {
    var h = window.location.hostname;
    // Always use the local service when previewing locally.
    if (h === "localhost" || h === "127.0.0.1") return "http://127.0.0.1:8000";
    var cfg = window.CONNTRAIL_DEMO_API;
    return cfg ? cfg.replace(/\/$/, "") : "";
  }
  var API = apiBase();

  // Example prompts.
  var examples = $("live-examples");
  EXAMPLES.forEach(function (text) {
    var b = UI.el("button", "chip", text.length > 40 ? text.slice(0, 40) + "…" : text);
    b.type = "button";
    b.title = text;
    b.addEventListener("click", function () { $("live-message").value = text; });
    examples.appendChild(b);
  });

  // Optional Turnstile widget.
  if (window.CONNTRAIL_TURNSTILE_SITEKEY) {
    var s = document.createElement("script");
    s.src = "https://challenges.cloudflare.com/turnstile/v0/api.js";
    s.async = true; s.defer = true;
    document.head.appendChild(s);
    var widget = document.createElement("div");
    widget.className = "cf-turnstile";
    widget.setAttribute("data-sitekey", window.CONNTRAIL_TURNSTILE_SITEKEY);
    $("live-turnstile").appendChild(widget);
  }

  function turnstileToken() {
    var el = form.querySelector('[name="cf-turnstile-response"]');
    return el ? el.value : null;
  }

  function setStatus(msg, cls) {
    var el = $("live-status");
    el.className = "live-status" + (cls ? " " + cls : "");
    el.textContent = msg || "";
  }

  // Show trial status.
  fetch(API + "/api/config")
    .then(function (r) { return r.json(); })
    .then(function (cfg) {
      var meta = $("live-meta");
      if (cfg.default_model) {
        meta.textContent = "Free trial · " + cfg.default_provider + " · " + cfg.default_model +
          " · " + cfg.daily_remaining + " analyses left today";
      } else {
        meta.textContent = "Free trial not configured — bring your own free key.";
      }
    })
    .catch(function () { $("live-meta").textContent = "Demo service not reachable right now."; });

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    var message = $("live-message").value.trim();
    if (!message) { setStatus("Type a message first.", "err"); return; }

    var payload = { message: message };
    var key = $("live-key").value.trim();
    if (key) { payload.api_key = key; payload.provider = $("live-provider").value || null; }
    var token = turnstileToken();
    if (token) payload.turnstile_token = token;

    var submit = $("live-submit");
    submit.disabled = true;
    setStatus("Analysing… this makes 5 free model calls and takes a few seconds.");

    fetch(API + "/api/analyse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, status: r.status, body: j }; }); })
      .then(function (res) {
        if (!res.ok) { setStatus(res.body.error || "Something went wrong.", "err"); return; }
        renderResult(res.body);
        setStatus("");
      })
      .catch(function () { setStatus("Could not reach the demo service.", "err"); })
      .finally(function () { submit.disabled = false; });
  });

  function renderResult(p) {
    var view = $("live-result");
    view.hidden = false;
    view.innerHTML = "";

    var head = UI.el("div", "case-head");
    head.appendChild(UI.el("h3", null, "Live analysis"));
    head.appendChild(UI.el("p", "case-incident", "Input: \u201C" + p.original_input + "\u201D"));
    view.appendChild(head);

    var grid = UI.el("div", "case-grid");

    var surface = UI.el("div", "surface");
    var sHead = UI.el("div", "surface-head");
    sHead.appendChild(UI.el("span", null, "Contrast variants"));
    sHead.appendChild(UI.el("span", null, p.provider + " · " + p.model));
    surface.appendChild(sHead);
    var sBody = UI.el("div", "surface-body");
    var vGrid = UI.el("div", "variant-grid");
    ["similar", "neutral", "opposite"].forEach(function (key) {
      var route = p.raw_outputs[key];
      var changed = route !== p.original_route;
      var card = UI.el("div", "variant" + (changed ? " changed" : ""));
      card.appendChild(UI.el("span", "variant-label", key));
      card.appendChild(UI.el("p", "variant-text", p.raw_contrasts[key]));
      card.appendChild(UI.el("span", "variant-route", "\u2192 " + route + (changed ? " (flipped)" : "")));
      vGrid.appendChild(card);
    });
    sBody.appendChild(vGrid);
    surface.appendChild(sBody);
    grid.appendChild(surface);

    var analysis = UI.el("div", "analysis");
    var aHead = UI.el("div", "analysis-head");
    aHead.appendChild(UI.el("span", null, "What Conntrail saw"));
    aHead.appendChild(UI.el("span", null, p.cached ? "cached" : "fresh"));
    analysis.appendChild(aHead);
    var aBody = UI.el("div", "analysis-body");

    var top = UI.el("div", "result-top");
    top.appendChild(UI.gauge(p.entropy_score, p.stability));
    var verdict = UI.el("div", "verdict");
    verdict.appendChild(UI.el("span", "badge badge-" + p.stability, p.stability));
    verdict.appendChild(UI.el("p", "counterfactual", p.counterfactual_route
      ? "Remove the driving dimension and the agent takes the other path: " + p.counterfactual_route + "."
      : "No variant flips the route — the decision is stable across all three."));
    top.appendChild(verdict);
    aBody.appendChild(top);

    aBody.appendChild(UI.variantTable(p.original_route, p.raw_outputs));
    var driver = UI.el("p", "attr-driver");
    driver.appendChild(UI.el("span", null, "First flip: "));
    driver.appendChild(UI.el("strong", null, p.attribution_dimension || "none detected"));
    aBody.appendChild(driver);

    var overhead = p.analysis_overhead || {};
    var cost = overhead.cost_usd ? UI.fmtUsd(overhead.cost_usd) : "$0.00";
    var tokens = UI.fmtTokens(overhead.total_tokens) || "0 tokens";
    var latency = overhead.latency_ms != null ? Math.round(overhead.latency_ms) + " ms" : "—";
    aBody.appendChild(UI.el("p", "cost-line",
      "This analysis cost " + cost + " (" + tokens + ", " + latency + "). The observer's bill is measured too."));

    analysis.appendChild(aBody);
    grid.appendChild(analysis);

    view.appendChild(grid);
    if (typeof view.scrollIntoView === "function") {
      view.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }
})();
