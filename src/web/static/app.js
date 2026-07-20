/* Plainstock frontend logic.
   Talks to the Flask API (/api/recommend/<ticker>), manages the three states
   (loading / error / result), and draws the price chart with Plotly.
   No framework — plain DOM, kept readable for the report. */

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

  // Training an LSTM on demand can take a while, so we allow a generous window
  // before giving up. Without this, the browser's default behaviour can abandon
  // the request while the server is still working, showing a false "server
  // unreachable" error even though a result is coming.
  const REQUEST_TIMEOUT_MS = 180000; // 3 minutes

  async function lookup(ticker) {
    loadingTicker.textContent = ticker;
    showOnly(loadingPanel);
    setBusy(true);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
      const resp = await fetch(`/api/recommend/${encodeURIComponent(ticker)}`, {
        signal: controller.signal,
      });
      console.log("[plainstock] fetch returned, status:", resp.status);
      const data = await resp.json();
      console.log("[plainstock] parsed JSON:", data);

      if (!resp.ok) {
        // API returned a structured error (400/500).
        errorMessage.textContent =
          data.error || "Something went wrong. Please try again.";
        showOnly(errorPanel);
        return;
      }

      console.log("[plainstock] rendering result…");
      renderResult(data);
      console.log("[plainstock] render complete, showing result panel");
      showOnly(resultPanel);
    } catch (err) {
      console.error("[plainstock] caught error:", err);
      if (err.name === "AbortError") {
        errorMessage.textContent =
          "This is taking longer than expected — the model may still be " +
          "training. Give it a moment and try again; the second attempt is " +
          "usually instant.";
      } else {
        errorMessage.textContent =
          "Couldn't reach the server. Make sure it's running (python -m " +
          "src.web.app) and try again.";
      }
      showOnly(errorPanel);
    } finally {
      clearTimeout(timer);
      setBusy(false);
    }
  }

  function renderResult(data) {
    document.getElementById("result-ticker").textContent = data.ticker;
    document.getElementById("result-date").textContent =
      " · as of " + data.as_of_date;

    const word = document.getElementById("verdict-word");
    word.textContent = data.recommendation;
    word.className =
      "verdict-word " + (data.recommendation === "BUY" ? "buy" : "sell");

    document.getElementById("verdict-tally").textContent =
      `${data.bullish_count} signals leaning positive · ${data.bearish_count} leaning cautious`;

    document.getElementById("result-headline").textContent = data.headline;

    const list = document.getElementById("signal-list");
    list.innerHTML = "";
    data.signals.forEach((s) => {
      const li = document.createElement("li");
      li.className = s.leaning; // bullish | bearish | neutral
      li.textContent = s.phrase;
      list.appendChild(li);
    });

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
      // Plotly can initially size to its own default width, overflowing the
      // card. Force a resize to the container once drawn.
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