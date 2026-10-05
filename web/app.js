/*
 * Static demo for Conntrail — "Is this routing decision stable?"
 *
 * Loads precomputed TraceRecord fixtures from data/demo.json. No LLM calls,
 * no signup, no server: the entropy/attribution shown are exactly what the
 * SDK measured when the fixtures were frozen.
 */
(function () {
  "use strict";

  var CIRCUMFERENCE = 2 * Math.PI * 52; // gauge radius = 52
  var VARIANTS = ["similar", "neutral", "opposite"];
  var VARIANT_LABELS = { similar: "Similar", neutral: "Neutral", opposite: "Opposite" };

  var state = { records: [], active: null };

  function $(id) { return document.getElementById(id); }

  function fmtCost(usd) {
    if (usd === null || usd === undefined) return null;
    return "$" + Number(usd).toFixed(5);
  }

  function renderPicker() {
    var picker = $("scenario-picker");
    picker.innerHTML = "";
    state.records.forEach(function (record, i) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = record.scenario_title || record.node_id;
      btn.className = i === 0 ? "active" : "";
      btn.addEventListener("click", function () {
        Array.prototype.forEach.call(picker.children, function (c) { c.classList.remove("active"); });
        btn.classList.add("active");
        select(i);
      });
      picker.appendChild(btn);
    });
  }

  function select(index) {
    state.active = state.records[index];
    var record = state.active;

    $("node-id").textContent = record.node_id;
    $("original-input").textContent = record.original_input;
    $("original-route").textContent = record.original_route;

    $("variants").hidden = true;
    $("results").hidden = true;
    $("perturb").disabled = false;
    $("perturb").textContent = "Perturb";
  }

  function buildVariants(record) {
    var grid = $("variant-grid");
    grid.innerHTML = "";
    VARIANTS.forEach(function (key) {
      var route = record.raw_outputs[key];
      var changed = route !== record.original_route;
      var card = document.createElement("div");
      card.className = "variant" + (changed ? " changed" : "");
      card.innerHTML =
        '<span class="variant-label">' + VARIANT_LABELS[key] + "</span>" +
        '<p class="variant-text"></p>' +
        '<span class="variant-route">→ ' + route + (changed ? " (flipped)" : "") + "</span>";
      card.querySelector(".variant-text").textContent = record.raw_contrasts[key];
      grid.appendChild(card);
    });
  }

  function buildAttribution(record) {
    var body = $("attr-body");
    body.innerHTML = "";
    VARIANTS.forEach(function (key) {
      var route = record.raw_outputs[key];
      var changed = route !== record.original_route;
      var tr = document.createElement("tr");
      if (changed) tr.className = "flip";
      tr.innerHTML =
        "<td>" + VARIANT_LABELS[key] + "</td>" +
        "<td><code>" + route + "</code></td>" +
        "<td>" + (changed ? "yes" : "no") + "</td>";
      body.appendChild(tr);
    });
  }

  function revealResults() {
    var record = state.active;
    var entropy = Number(record.entropy_score) || 0;
    var stability = record.stability;

    $("entropy-score").textContent = entropy.toFixed(2);
    var gauge = $("gauge-value");
    gauge.style.strokeDashoffset = String(CIRCUMFERENCE * (1 - entropy));
    gauge.style.stroke = "var(--" + stability + ")";

    var badge = $("stability-badge");
    badge.textContent = stability;
    badge.className = "badge badge-" + stability;

    var cf = record.counterfactual_route;
    $("counterfactual").textContent = cf
      ? "Remove the driving dimension and the agent takes the other path: " + cf + "."
      : "No variant flipped the route; the decision is stable across all three.";

    $("attribution-dimension").textContent = record.attribution_dimension || "none detected";

    var note = $("note");
    if (record.scenario_note) {
      note.textContent = record.scenario_note;
      note.hidden = false;
    } else {
      note.hidden = true;
    }

    var overhead = record.analysis_overhead;
    var costLine = $("cost-line");
    if (overhead) {
      var cost = fmtCost(overhead.cost_usd);
      var tokens = overhead.total_tokens;
      costLine.textContent =
        "This analysis itself cost " +
        (cost || (tokens + " tokens")) +
        " — the observer's bill is measured too, so sampling can be tuned from data.";
      costLine.hidden = false;
    } else {
      costLine.hidden = true;
    }

    $("results").hidden = false;
  }

  function onPerturb() {
    if (!state.active) return;
    buildVariants(state.active);
    $("variants").hidden = false;
    revealResults();
    $("perturb").disabled = true;
    $("perturb").textContent = "Perturbed";
  }

  function fail(message) {
    var demo = $("demo");
    var p = document.createElement("p");
    p.className = "subtitle";
    p.textContent = message;
    demo.appendChild(p);
  }

  fetch("data/demo.json")
    .then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    })
    .then(function (bundle) {
      state.records = bundle.records || [];
      if (!state.records.length) throw new Error("no records");
      renderPicker();
      select(0);
      $("perturb").addEventListener("click", onPerturb);
    })
    .catch(function (err) {
      fail("Could not load the precomputed fixtures (" + err.message + ").");
    });
})();
