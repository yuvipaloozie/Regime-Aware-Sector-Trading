import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


def load_config(config_path=None):
    """Load settings and resolve the dynamic end date."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("dates", {}).get("end") == "today":
        config["dates"]["end"] = datetime.now().strftime("%Y-%m-%d")
    return config


def splice_proxy_returns(target, proxy):
    """Backfill a target's pre-inception history using proxy returns, not price levels."""
    target = target.astype(float).copy()
    proxy = proxy.astype(float).copy()
    valid_target = target.dropna()
    if valid_target.empty:
        raise ValueError("Target has no observations and cannot be spliced to a proxy.")
    first_date = valid_target.index[0]
    proxy_anchor = proxy.loc[:first_date].dropna()
    if proxy_anchor.empty:
        return target
    history = proxy.loc[:first_date] / proxy_anchor.iloc[-1] * valid_target.iloc[0]
    result = target.copy()
    result.loc[result.index < first_date] = history.loc[history.index < first_date]
    return result


class DataPipeline:
    """Point-in-time ingestion with explicit provenance and demo isolation."""

    def __init__(self, config_path=None, raw_dir=None, data_mode=None):
        self.config = load_config(config_path)
        configured_mode = self.config.get("data", {}).get("mode", "real")
        self.data_mode = (data_mode or os.getenv("REGIME_DATA_MODE") or configured_mode).lower()
        if self.data_mode not in {"real", "demo"}:
            raise ValueError("data mode must be either 'real' or 'demo'")
        default_raw = Path(__file__).resolve().parent.parent / "data" / "raw"
        self.raw_dir = Path(raw_dir or default_raw)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.provenance = {}
        self.daily_equity = pd.DataFrame()

    @staticmethod
    def _utc_now():
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _meta_path(path):
        path = Path(path)
        return path.with_suffix(path.suffix + ".meta.json")

    def _write_cache(self, frame, path, metadata):
        path = Path(path)
        frame.to_csv(path)
        payload = {
            **metadata,
            "fetched_at": self._utc_now(),
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "start": str(frame.index.min().date()) if not frame.empty else None,
            "end": str(frame.index.max().date()) if not frame.empty else None,
        }
        self._meta_path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def _load_valid_cache(self, path, required_columns, requested_end):
        path = Path(path)
        meta_path = self._meta_path(path)
        if not path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("synthetic", False) and self.data_mode != "demo":
                return None
            fetched = pd.Timestamp(meta["fetched_at"])
            max_age = int(self.config.get("data", {}).get("cache_max_age_days", 7))
            if pd.Timestamp.now(tz="UTC") - fetched > pd.Timedelta(days=max_age):
                return None
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if df.empty or set(required_columns) - set(df.columns):
                return None
            tolerance = int(self.config.get("data", {}).get("cache_end_tolerance_days", 10))
            if pd.Timestamp(requested_end).normalize() - df.index.max().normalize() > pd.Timedelta(days=tolerance):
                return None
            return df, meta
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

    def generate_synthetic_equity_data(self, tickers, start_date, end_date):
        """Generate deterministic data only when demo mode was explicitly requested."""
        if self.data_mode != "demo":
            raise RuntimeError("Synthetic equity data is disabled outside explicit demo mode.")
        idx = pd.date_range(start=start_date, end=end_date, freq="B")
        rng = np.random.default_rng(42)
        market = rng.normal(0.0003, 0.009, len(idx))
        df = pd.DataFrame(index=idx)
        df["SPY"] = np.exp(np.log(100.0) + np.cumsum(market))
        vix = np.zeros(len(idx))
        vix[0] = 18.0
        for i in range(1, len(idx)):
            vix[i] = max(8.0, vix[i - 1] + 0.05 * (16 - vix[i - 1]) + rng.normal(0, 1.8) - 400 * min(0, market[i]))
        df["VIX"] = vix
        df["^TNX"] = np.maximum(0.5, 4.5 + np.cumsum(rng.normal(0, 0.04, len(idx))))
        betas = {"XLK": 1.3, "XLE": 1.1, "XLF": 1.2, "XLV": 0.8, "XLI": 1.0,
                 "XLY": 1.2, "XLP": 0.6, "XLU": 0.5, "XLB": 1.0, "XLC": 1.1,
                 "XLRE": 0.9, "VOX": 1.0, "VNQ": 0.9, "TLT": -0.2, "IEF": -0.1, "SHY": 0.0}
        for ticker in tickers:
            normalized = "VIX" if ticker == "^VIX" else ticker
            if normalized in df.columns:
                continue
            beta = betas.get(ticker, 1.0)
            noise = rng.normal(0.0001, 0.005 if beta <= 0 else 0.007, len(idx))
            df[ticker] = np.exp(np.log(50.0) + np.cumsum(beta * market + noise))
        df.index.name = "Date"
        return df

    def generate_synthetic_macro_data(self, start_date, end_date):
        if self.data_mode != "demo":
            raise RuntimeError("Synthetic macro data is disabled outside explicit demo mode.")
        idx = pd.date_range(start=start_date, end=end_date, freq="D")
        rng = np.random.default_rng(101)
        df = pd.DataFrame(index=idx)
        df["T10Y2Y"] = 1.5 + np.cumsum(rng.normal(0, 0.02, len(idx)))
        df["STLFSI4"] = -0.5 + np.cumsum(rng.normal(0, 0.025, len(idx)))
        df["CPIAUCSL"] = 160 + 0.02 * np.arange(len(idx))
        df.index.name = "Date"
        return df

    def fetch_equity_data(self, start_date=None, end_date=None):
        start_date = start_date or self.config["dates"]["start"]
        end_date = end_date or self.config["dates"]["end"]
        sectors = self.config["sectors"]
        proxies = list(self.config.get("proxies", {}).values())
        benchmarks = list(self.config["benchmarks"].values())
        defensive = self.config.get("defensive_assets", [])
        tickers = sorted(set(sectors + proxies + benchmarks + defensive))
        expected = [("VIX" if ticker == "^VIX" else ticker) for ticker in tickers]
        cache_path = self.raw_dir / "yfinance_adjusted.csv"
        cached = self._load_valid_cache(cache_path, expected, end_date)
        if cached:
            df, meta = cached
            self.provenance["equity"] = meta
            return df.loc[start_date:end_date]

        if self.data_mode == "demo":
            df = self.generate_synthetic_equity_data(tickers, start_date, end_date)
            meta = self._write_cache(df, cache_path, {"source": "synthetic-demo", "synthetic": True, "adjusted": True})
            self.provenance["equity"] = meta
            return df

        try:
            downloaded = yf.download(
                tickers,
                start=start_date,
                end=(pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=True,
                actions=False,
                progress=False,
                group_by="column",
            )
            if downloaded.empty:
                raise ValueError("Yahoo Finance returned no observations")
            df = downloaded["Close"] if isinstance(downloaded.columns, pd.MultiIndex) else downloaded
            if "^VIX" in df.columns:
                df = df.rename(columns={"^VIX": "VIX"})
            df.index = pd.to_datetime(df.index).tz_localize(None)
            if missing := set(expected) - set(df.columns):
                raise ValueError(f"Yahoo Finance response is missing columns: {sorted(missing)}")
            meta = self._write_cache(df, cache_path, {"source": "yfinance", "synthetic": False, "adjusted": True})
            self.provenance["equity"] = meta
            return df
        except Exception as exc:
            raise RuntimeError(
                "Real equity ingestion failed. No synthetic fallback was used. "
                "Retry later or run with REGIME_DATA_MODE=demo."
            ) from exc

    def _fetch_fred_initial_release(self, series_id, start_date, end_date, api_key):
        params = urlencode({
            "series_id": series_id, "api_key": api_key, "file_type": "json", "output_type": 4,
            "observation_start": start_date, "observation_end": end_date, "limit": 100000, "sort_order": "asc",
        })
        with urlopen(f"https://api.stlouisfed.org/fred/series/observations?{params}", timeout=30) as response:
            payload = json.load(response)
        observations = pd.DataFrame(payload.get("observations", []))
        if observations.empty:
            raise ValueError(f"FRED returned no observations for {series_id}")
        observations[series_id] = pd.to_numeric(observations["value"].replace(".", np.nan), errors="coerce")
        observations["availability_date"] = pd.to_datetime(observations["realtime_start"])
        return observations.set_index("availability_date")[[series_id]].groupby(level=0).last()

    def fetch_macro_data(self, start_date=None, end_date=None):
        start_date = start_date or self.config["dates"]["start"]
        end_date = end_date or self.config["dates"]["end"]
        fred_ids = self.config["fred_indicators"]
        cache_path = self.raw_dir / "fred_point_in_time.csv"
        cached = self._load_valid_cache(cache_path, fred_ids, end_date)
        if cached:
            df, meta = cached
            self.provenance["macro"] = meta
            return df.loc[start_date:end_date]

        if self.data_mode == "demo":
            df = self.generate_synthetic_macro_data(start_date, end_date)
            meta = self._write_cache(df, cache_path, {"source": "synthetic-demo", "synthetic": True, "point_in_time": False})
            self.provenance["macro"] = meta
            return df

        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            raise RuntimeError(
                "FRED_API_KEY is required for point-in-time macro data. "
                "Set it or explicitly run with REGIME_DATA_MODE=demo."
            )
        try:
            frames = [self._fetch_fred_initial_release(fid, start_date, end_date, api_key) for fid in fred_ids]
            df = pd.concat(frames, axis=1).sort_index().ffill()
            meta = self._write_cache(df, cache_path, {
                "source": "fred-api-initial-release", "synthetic": False, "point_in_time": True,
                "index_semantics": "first-availability-date",
            })
            self.provenance["macro"] = meta
            return df
        except Exception as exc:
            raise RuntimeError("Point-in-time FRED ingestion failed; revised data was not substituted.") from exc

    def get_processed_data(self, start_date=None, end_date=None):
        """Return Friday observations while retaining adjusted daily prices for execution."""
        eq_raw = self.fetch_equity_data(start_date, end_date)
        macro_raw = self.fetch_macro_data(start_date, end_date)
        df_eq = eq_raw.sort_index().copy()
        for sector, proxy in self.config.get("proxies", {}).items():
            if sector in df_eq and proxy in df_eq:
                df_eq[sector] = splice_proxy_returns(df_eq[sector], df_eq[proxy])

        daily_columns = self.config["sectors"] + self.config.get("defensive_assets", []) + [
            self.config["benchmarks"]["equity"]
        ]
        self.daily_equity = df_eq[list(dict.fromkeys(daily_columns))].copy().ffill()
        # Preserve releases that occur off-market and make them available only on
        # the next equity session.
        joined = pd.concat([df_eq, macro_raw], axis=1).sort_index().ffill().reindex(df_eq.index)
        required = ["SPY", "VIX", "T10Y2Y", "STLFSI4", "CPIAUCSL"] + self.config["sectors"]
        joined = joined.dropna(subset=required)
        if joined.empty:
            raise ValueError("No complete point-in-time observations remain after joining data sources.")
        return joined.resample("W-FRI").last()

    def write_run_provenance(self, output_path, extra=None):
        payload = {"generated_at": self._utc_now(), "data_mode": self.data_mode,
                   "sources": self.provenance, **(extra or {})}
        Path(output_path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload
