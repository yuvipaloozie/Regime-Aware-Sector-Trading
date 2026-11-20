"use strict";

import "./style.css";
import plotlyScript from "plotly.js-dist-min/plotly.min.js?url";
import {mountActivity, mountChartRange} from "./controls.jsx";

const get = (id) => document.getElementById(id);
const theme = getComputedStyle(document.documentElement);
const tone = (name) => theme.getPropertyValue(`--${name}`).trim();
const colors = {strategy: tone("strategy"), spy: tone("spy"), sectors: tone("sectors")};
const names = {strategy: "XGBoost + HMM", spy: "SPY buy-and-hold", sectors: "Equal-weight sectors"};
const regimes = [
  {name: "Low volatility", color: tone("low")},
  {name: "Moderate volatility", color: tone("moderate")},
  {name: "High volatility", color: tone("high")},
  {name: "Extreme volatility", color: tone("extreme")},
];
const usd = new Intl.NumberFormat("en-US", {style: "currency", currency: "USD"});
const shares = new Intl.NumberFormat("en-US", {maximumFractionDigits: 2});
const pct = (value) => `${(100 * value).toFixed(2)}%`;

function appendRow(body, values) {
  const row = body.insertRow();
  values.forEach((value) => { row.insertCell().textContent = value; });
  return row;
}

function maxDrawdown(values) {
  let peak = values[0];
  let worst = 0;
  for (const value of values) {
    peak = Math.max(peak, value);
    worst = Math.min(worst, value / peak - 1);
  }
  return worst;
}

async function draw({dates, series, weights, regimes: recorded, fills, metrics}) {
  const chart = get("equity");
  const datePicker = get("date");
  const last = dates.at(-1);
  const keys = Object.keys(names);
  let activeRange = 0;
  const states = Object.entries(recorded);
  const fillsNewestFirst = [...fills].sort((left, right) => right[0].localeCompare(left[0]));
  const csv = ["Date,Ticker,Side,Shares,Fill price", ...fillsNewestFirst.map((fill) => fill.join(","))].join("\r\n");
  get("download").href = URL.createObjectURL(new Blob([csv], {type: "text/csv;charset=utf-8"}));
  const showFills = mountActivity(get("activity"), fillsNewestFirst.map(([date, ticker, side, units, price]) =>
    [date, ticker, side, shares.format(units), usd.format(price)]));

  const legendTitle = document.createElement("span");
  legendTitle.textContent = "Saved regimes";
  get("regime-legend").append(legendTitle);
  regimes.forEach(({name, color}) => {
    const key = document.createElement("span");
    key.className = "key";
    key.style.setProperty("--key", color);
    key.textContent = name;
    get("regime-legend").append(key);
  });

  datePicker.replaceChildren(...[...dates].reverse().map((date) => new Option(date, date)));
  const baseShapes = states.slice(0, -1).map(([date, [state]], index) => ({
    type: "rect", x0: date, x1: states[index + 1][0], y0: 0, y1: 1, yref: "paper",
    fillcolor: regimes[state].color, opacity: .11, line: {width: 0}, layer: "below",
  }));
  const finalState = states.at(-1);
  const traces = keys.map((key, index) => ({
    type: "scatter", mode: "lines", name: names[key], x: dates, y: series[key].map((value) => value * 100000),
    line: {color: colors[key], width: index === 0 ? 2.5 : 1.7, dash: ["solid", "dash", "dot"][index]},
    hovertemplate: "%{y:$,.0f}",
  }));
  traces.push({
    type: "scatter", mode: "markers", name: "Saved regime", x: states.map(([date]) => date),
    y: states.map(() => .035), yaxis: "y2", showlegend: false,
    marker: {symbol: "square", size: 7, color: states.map(([, [state]]) => regimes[state].color)},
    text: states.map(([, [state, equity]]) => `${regimes[state].name} · ${pct(equity)} equity`),
    hovertemplate: "%{text}<extra></extra>",
  });

  await Plotly.newPlot(chart, traces, {
    margin: {l: 54, r: 18, t: 12, b: 33}, paper_bgcolor: tone("panel"), plot_bgcolor: tone("panel"),
    font: {family: "Segoe UI, sans-serif", color: tone("muted"), size: 12}, showlegend: false,
    dragmode: false, hovermode: "x unified", hoverlabel: {bgcolor: tone("toolbar"), font: {color: tone("ink")}},
    xaxis: {type: "date", fixedrange: true, gridcolor: tone("grid"), hoverformat: "%b %d, %Y", showspikes: true, spikemode: "across", spikecolor: tone("muted"), spikethickness: 1},
    yaxis: {tickprefix: "$", tickformat: "~s", gridcolor: tone("grid"), fixedrange: true, zeroline: false},
    yaxis2: {overlaying: "y", range: [0, 1], visible: false, fixedrange: true},
  }, {responsive: true, displayModeBar: false});

  keys.forEach((key, index) => {
    const label = document.createElement("label");
    const toggle = document.createElement("input");
    const keyText = document.createElement("span");
    toggle.type = "checkbox";
    toggle.checked = true;
    toggle.addEventListener("change", async () => {
      await Plotly.restyle(chart, {visible: toggle.checked}, [index]);
      setRange(activeRange);
    });
    keyText.className = "key";
    keyText.style.setProperty("--key", colors[key]);
    keyText.style.setProperty("--stroke", ["solid", "dashed", "dotted"][index]);
    keyText.textContent = names[key];
    label.append(toggle, keyText);
    get("curves").append(label);
    const values = series[key];
    const metric = metrics[key];
    const row = appendRow(get("metrics"), [names[key], pct(values.at(-1) / values[0] - 1), pct(metric.cagr),
      metric.sharpe.toFixed(2), metric.sortino.toFixed(2), pct(maxDrawdown(values))]);
    row.cells[0].className = "series-name";
    row.cells[0].style.setProperty("--series", colors[key]);
  });

  function selectDate() {
    const date = datePicker.value;
    const state = recorded[date];
    get("selection").textContent = state
      ? `${regimes[state[0]].name} · ${pct(state[1])} sector equity`
      : `Saved regimes start ${states[0][0]}.`;
    get("selection").style.color = state ? regimes[state[0]].color : "";
    const holdings = Object.entries(weights[date]).sort((left, right) => right[1] - left[1]);
    get("weights").replaceChildren(...holdings.map(([ticker, weight]) => {
      const item = document.createElement("li");
      const symbol = document.createElement("span");
      const meter = document.createElement("meter");
      const value = document.createElement("span");
      symbol.textContent = ticker;
      meter.value = weight;
      meter.setAttribute("aria-label", `${ticker} target weight`);
      if (ticker === "TLT") meter.className = "defensive";
      value.textContent = pct(weight);
      item.append(symbol, meter, value);
      return item;
    }));
    Plotly.relayout(chart, {shapes: [...baseShapes, {
      type: "line", x0: date, x1: date, y0: 0, y1: 1, yref: "paper",
      line: {color: date === finalState[0] ? regimes[finalState[1][0]].color : tone("muted"), width: 1, dash: "dot"},
    }]});
    showFills(date, state ? "No fills on this date." : `Saved fills start ${states[0][0]}.`);
  }

  chart.on("plotly_click", ({points}) => {
    const date = points[0].x;
    if (Object.hasOwn(weights, date)) {
      datePicker.value = date;
      selectDate();
    }
  });
  datePicker.addEventListener("change", selectDate);
  function setRange(years) {
    activeRange = years;
    const start = new Date(last);
    start.setUTCFullYear(start.getUTCFullYear() - years);
    const range = years ? [start.toISOString().slice(0, 10), last] : [dates[0], last];
    const visibleValues = keys.flatMap((key, index) => chart.data[index].visible === false ? [] :
      series[key].filter((_, point) => dates[point] >= range[0]).map((value) => value * 100000));
    const low = Math.min(...visibleValues);
    const high = Math.max(...visibleValues);
    const pad = Math.max((high - low) * .08, 1000);
    Plotly.relayout(chart, {"xaxis.range": range, ...(visibleValues.length ? {"yaxis.range": [low - pad, high + pad]} : {})});
  }
  mountChartRange(get("ranges"), setRange);
  selectDate();
}

async function load() {
  const plotlyReady = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = plotlyScript;
    script.addEventListener("load", resolve, {once: true});
    script.addEventListener("error", () => reject(new Error("Could not load Plotly")), {once: true});
    document.head.append(script);
  });
  const [response] = await Promise.all([fetch(`${import.meta.env.BASE_URL}archive.json`), plotlyReady]);
  if (!response.ok) throw new Error("Missing backtest archive");
  const archive = await response.json();
  get("terminal").hidden = false;
  await draw(archive);
  get("status").hidden = true;
}

load().catch((error) => {
  console.error(error);
  get("terminal").hidden = true;
  get("status").textContent = "Could not load the backtest. Reload the page to try again.";
});
