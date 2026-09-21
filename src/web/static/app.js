/* Plainstock frontend logic.
   Talks to the Flask API (/api/recommend/<ticker>), manages the panel states
   (loading / error / result), renders the model's own reasoning, and draws the
   price chart with Plotly. No framework — plain DOM, kept readable for the
   report.

   Note on the result panel: the API now reports whether the model was behaving
   like a predictor at all. When it has collapsed to one class, there is no
   reasoning to show, so the reasoning block is hidden entirely and only the
   explanation of *why* nothing is shown appears. */

(function () {
  "use strict";

  const form = document.getElementById("lookup-form");
  const input = document.getElementById("ticker");
  const goBtn = document.getElementById("go");

  const loadingPanel = document.getElementById("loading");
  const errorPanel = document.getElementById("error");
  const resultPanel = document.getElementById("result");

  const loadingTicker = document.getElementById("loading-ticker");
  const errorMessage = document.getElementById("error-message");

  function showOnly(panel) {
    [loadingPanel, errorPanel, resultPanel].forEach((p) => {
      p.hidden = p !== panel;
    });
  }

  function setBusy(busy) {
    goBtn.disabled = busy;
    input.disabled = busy;
    goBtn.textContent = busy ? "Reading…" : "Get the read";
  }

  // The served model is a Random Forest, which trains in seconds rather than
  // the tens of seconds the earlier sequence model needed. A 60-second ceiling
  // is ample; it exists only so a stalled request fails visibly rather than
  // hanging.
  const REQUEST_TIMEOUT_MS = 60000;

  async function lookup(ticker) {
    loadingTicker.textContent = ticker;
    showOnly(loadingPanel);
    setBusy(true);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    // Stage 1: the network request. Only failures here mean the server
    // could not be reached.
    let resp, data;
    try {
      resp = await fetch(`/api/recommend/${encodeURIComponent(ticker)}`, {
        signal: controller.signal,
      });
      data = await resp.json();
    } catch (err) {
      console.error("[plainstock] request failed:", err);
      errorMessage.textContent =
        err.name === "AbortError"
          ? "This is taking longer than expected. Give it a moment and try " +
            "again; the second attempt is usually instant."
          : "Couldn't reach the server. Make sure it's running (python -m " +
            "src.web.app) and try again.";
      showOnly(errorPanel);
      clearTimeout(timer);
      setBusy(false);
      return;
    }
    clearTimeout(timer);

    if (!resp.ok) {
      errorMessage.textContent =
        data.error || "Something went wrong. Please try again.";
      showOnly(errorPanel);
      setBusy(false);
      return;
    }

    // Stage 2: rendering. The server answered; a failure here is a page bug,
    // so it is reported as one rather than blamed on the connection.
    try {
      renderResult(data);
      showOnly(resultPanel);
    } catch (err) {
      console.error("[plainstock] render failed:", err);
      errorMessage.textContent =
        "The analysis finished, but the page couldn't display it (" +
        err.message + "). The page template may be out of date.";
      showOnly(errorPanel);
    } finally {
      setBusy(false);
    }
  }

  /* Map a SHAP direction onto the existing bullish/bearish bullet styling. */
  function leaningFromDirection(direction) {
    if (direction === "toward BUY") return "bullish";
    if (direction === "toward SELL") return "bearish";
    return "neutral";
  }

  function renderConfidence(confidence, explained) {
    const banner = document.getElementById("confidence-banner");
    const message = document.getElementById("confidence-message");
    const summary = document.getElementById("verdict-confidence");

    const pct = Math.round(confidence.buy_probability * 100);
    summary.textContent = `Model probability for BUY: ${pct}%`;

    banner.className = "confidence-banner " + confidence.status;

    if (confidence.status === "ok") {
      banner.hidden = true;
    } else {
      message.textContent = confidence.message;
      banner.hidden = false;
    }

    // When the model has collapsed there is no reasoning to display at all.
    document.getElementById("reasoning").hidden = !explained;
  }

  function renderResult(data) {
    document.getElementById("result-ticker").textContent = data.ticker;
    document.getElementById("result-date").textContent =
      " · as of " + data.as_of_date;

    const word = document.getElementById("verdict-word");
    word.textContent = data.recommendation;
    word.className =
      "verdict-word " + (data.recommendation === "BUY" ? "buy" : "sell");

    renderConfidence(data.confidence, data.explained);

    document.getElementById("result-headline").textContent = data.headline;

    const list = document.getElementById("signal-list");
    list.innerHTML = "";
    (data.signals || []).forEach((s) => {
      const li = document.createElement("li");
      li.className = leaningFromDirection(s.direction);

      const text = document.createElement("span");
      text.className = "signal-text";
      text.textContent = s.sentence;

      const weight = document.createElement("span");
      weight.className = "signal-weight";
      weight.textContent =
        `${s.display_name}: ${s.shap_value > 0 ? "+" : ""}${s.shap_value}`;
      weight.title =
        "SHAP value — how far this feature moved the model's decision. " +
        "Positive pushes toward BUY, negative toward SELL.";

      li.appendChild(text);
      li.appendChild(weight);
      list.appendChild(li);
    });

    const caveat = document.getElementById("result-caveat");
    if (data.caveat) {
      caveat.textContent = data.caveat;
      caveat.hidden = false;
    } else {
      caveat.hidden = true;
    }

    drawChartSafely(data.price_history);
  }

  function drawChartSafely(history) {
    try {
      if (typeof Plotly === "undefined") {
        console.warn("[plainstock] Plotly not loaded — skipping chart.");
        return;
      }
      drawChart(history);
    } catch (e) {
      console.error("[plainstock] chart failed (result still shown):", e);
    }
  }

  function drawChart(history) {
    const dates = history.map((p) => p.date);
    const closes = history.map((p) => p.close);

    const trace = {
      x: dates,
      y: closes,
      type: "scatter",
      mode: "lines",
      line: { color: "#2f6f5e", width: 2 },
      fill: "tozeroy",
      fillcolor: "rgba(47,111,94,0.07)",
      hovertemplate: "%{x}<br>$%{y:.2f}<extra></extra>",
    };

    const layout = {
      margin: { l: 44, r: 12, t: 8, b: 32 },
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { family: "Inter, sans-serif", size: 12, color: "#55605a" },
      xaxis: { showgrid: false, tickformat: "%b %Y" },
      yaxis: { showgrid: true, gridcolor: "#eceae3", tickprefix: "$" },
      showlegend: false,
    };

    Plotly.newPlot("chart", [trace], layout, {
      displayModeBar: false,
      responsive: true,
    }).then(function () {
      Plotly.Plots.resize("chart");
    });
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    const ticker = input.value.trim().toUpperCase();
    if (!ticker) {
      errorMessage.textContent = "Type a stock symbol first — like MSFT.";
      showOnly(errorPanel);
      return;
    }
    lookup(ticker);
  });
})();