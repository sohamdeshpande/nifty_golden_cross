#!/usr/bin/env python3
"""
daily_screener.py

Every weekday morning (pre-market), for every NIFTY 50 + NIFTY Midcap 50 stock:
    Stock, CMP, Open, Close, 200DMA, 50DMA, Golden Cross, ROE %
for the last 5 completed sessions -> a CSV, sent to Telegram.

Plus: when a stock's 50/200 DMA trend flips NEGATIVE -> POSITIVE (golden cross)
on the latest session, an audible Telegram message naming the stock(s).

Prices : yfinance (batch download, split/bonus-adjusted closes).
ROE    : screener.in via screenercli, cached in state/roe_cache.json.
         A stock is re-scraped only if its cached ROE is older than
         --roe-ttl-days (default 1 = effectively daily). If today's scrape
         fails, the last cached value is used, so the table is never blank.
CMP    : the latest completed close (pre-market it equals that row's Close).

Golden cross is detected against state/trend_state.json (last trend per
stock), so a skipped scheduled run can't miss a cross and a re-run can't
alert twice.

Env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
Usage:
    python daily_screener.py                    # normal run
    python daily_screener.py --dry-run          # no Telegram, no state write
    python daily_screener.py --limit 8          # first 8 symbols (smoke test)
    python daily_screener.py --roe-ttl-days 5   # re-scrape ROE only if >5 days old
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
from screener_cli.parsers import pros_cons
from screener_cli.scraper import (
    CompanyNotFoundError,
    RateLimitError,
    ScraperError,
    fetch_page_with_fallback,
)

IST = ZoneInfo("Asia/Kolkata")
FAST, SLOW = 50, 200
HISTORY_PERIOD = "2y"
LOOKBACK_ROWS = 5
PRICE_RETRY_WAIT = 30
ROE_DELAY_SECONDS = 8.0          # screener has no API; be gentle
POSITIVE, NEGATIVE = "POSITIVE", "NEGATIVE"

STATE_DIR = Path("state")
TREND_FILE = STATE_DIR / "trend_state.json"
ROE_CACHE_FILE = STATE_DIR / "roe_cache.json"
REPORT_CSV = Path("daily_report.csv")
SITE_DIR = Path("site")
REPORT_HTML = SITE_DIR / "index.html"

NIFTY50_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
NIFTY_MIDCAP50_URL = "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap50list.csv"
NSE_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                             "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


def ist_today() -> date:
    return datetime.now(IST).date()


# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #

def fetch_constituents(url: str, name: str) -> list[str]:
    resp = requests.get(url, headers=NSE_HEADERS, timeout=15)
    if resp.status_code != 200:
        print(f"[warning] {name} list HTTP {resp.status_code} from {url}", file=sys.stderr)
        return []
    return [r["Symbol"].strip() for r in csv.DictReader(StringIO(resp.text)) if r.get("Symbol", "").strip()]


def build_universe() -> dict[str, set[str]]:
    universe: dict[str, set[str]] = {}
    for url, name in [(NIFTY50_URL, "NIFTY50"), (NIFTY_MIDCAP50_URL, "NIFTY_MIDCAP50")]:
        symbols = fetch_constituents(url, name)
        if not symbols:
            sys.exit(f"[error] {name} list came back empty - refusing to run on a partial universe.")
        for s in symbols:
            universe.setdefault(s, set()).add(name)
    return universe


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #

def download(tickers: list[str]) -> pd.DataFrame:
    try:
        return yf.download(tickers, period=HISTORY_PERIOD, interval="1d", group_by="ticker",
                            auto_adjust=False, actions=False, progress=False, threads=True)
    except Exception as exc:
        print(f"yf.download failed outright: {exc}", file=sys.stderr)
        return pd.DataFrame()


def frame_for(data: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if data is None or data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        for level in (0, 1):
            if ticker in data.columns.get_level_values(level):
                return data.xs(ticker, axis=1, level=level)
        return None
    return data


def completed_sessions(frame: pd.DataFrame, today: date) -> pd.DataFrame:
    frame = frame.dropna(subset=["Close"])
    idx = frame.index
    if idx.tz is not None:
        idx = idx.tz_convert(IST)
    return frame.loc[[d < today for d in idx.date]]


# --------------------------------------------------------------------------- #
# ROE (screener.in, cached)
# --------------------------------------------------------------------------- #

def _parse_pct(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value.strip().rstrip("%").replace(",", "").strip())
    except ValueError:
        return None


def scrape_roe(symbol: str, retries_left: int = 2) -> float | None:
    try:
        soup, _ = fetch_page_with_fallback(symbol, view="consolidated")
    except CompanyNotFoundError:
        print(f"  {symbol}: not on screener.in", file=sys.stderr)
        return None
    except RateLimitError:
        if retries_left <= 0:
            print(f"  {symbol}: rate-limited, giving up", file=sys.stderr)
            return None
        time.sleep(30)
        return scrape_roe(symbol, retries_left - 1)
    except ScraperError as exc:
        print(f"  {symbol}: {exc}", file=sys.stderr)
        return None
    return _parse_pct(pros_cons.parse(soup).key_metrics.get("ROE"))


def resolve_roe(symbols: list[str], cache: dict, ttl_days: int, today: date) -> dict[str, float | None]:
    """Return {symbol: roe}, scraping only stocks whose cache is stale/missing,
    falling back to the cached value when a fresh scrape fails."""
    fresh_cutoff = today - timedelta(days=ttl_days)
    out: dict[str, float | None] = {}
    for i, symbol in enumerate(symbols, start=1):
        entry = cache.get(symbol)
        cached_val = entry.get("roe") if entry else None
        cached_day = date.fromisoformat(entry["as_of"]) if entry and entry.get("as_of") else None

        if cached_day is not None and cached_day > fresh_cutoff:
            out[symbol] = cached_val
            continue

        print(f"[ROE {i}/{len(symbols)}] {symbol}", file=sys.stderr)
        val = scrape_roe(symbol)
        if val is not None:
            cache[symbol] = {"roe": val, "as_of": today.isoformat()}
            out[symbol] = val
        else:                                   # scrape failed -> keep last known
            out[symbol] = cached_val
        time.sleep(ROE_DELAY_SECONDS)
    return out


# --------------------------------------------------------------------------- #
# Table + signal
# --------------------------------------------------------------------------- #

def build_rows(symbol: str, frame: pd.DataFrame, indexes: set[str], roe: float | None,
               stored_trend: str | None) -> tuple[list[dict], str | None, str]:
    """(csv rows for last 5 sessions, latest trend, cross) where cross is
    'GOLDEN' (NEG->POS), 'DEATH' (POS->NEG), or '' (no flip on latest bar)."""
    close = frame["Close"]
    cmp_price = float(close.iloc[-1])
    have_sma = len(close) >= SLOW + 1
    sma50 = close.rolling(FAST).mean() if have_sma else None
    sma200 = close.rolling(SLOW).mean() if have_sma else None

    def trend_at(pos: int) -> str | None:
        if not have_sma or pd.isna(sma200.iloc[pos]):
            return None
        return POSITIVE if sma50.iloc[pos] > sma200.iloc[pos] else NEGATIVE

    latest_trend = trend_at(-1)
    prev = stored_trend if stored_trend in (POSITIVE, NEGATIVE) else (trend_at(-2) if have_sma else None)
    cross = ""
    if latest_trend and prev and latest_trend != prev:
        cross = "GOLDEN" if latest_trend == POSITIVE else "DEATH"

    def gap_at(pos: int) -> float | None:
        if not have_sma or pd.isna(sma200.iloc[pos]) or sma200.iloc[pos] == 0:
            return None
        return float((sma50.iloc[pos] - sma200.iloc[pos]) / sma200.iloc[pos] * 100)

    rows = []
    n = len(frame)
    for pos in range(max(0, n - LOOKBACK_ROWS), n):
        t = trend_at(pos)
        g = gap_at(pos)
        rows.append({
            "Stock": symbol,
            "Index": "+".join(sorted(indexes)),
            "Date": frame.index[pos].date().isoformat(),
            "CMP": f"{cmp_price:.2f}",
            "Open": f"{frame['Open'].iloc[pos]:.2f}" if "Open" in frame else "",
            "Close": f"{close.iloc[pos]:.2f}",
            "200DMA": f"{sma200.iloc[pos]:.2f}" if have_sma and not pd.isna(sma200.iloc[pos]) else "",
            "50DMA": f"{sma50.iloc[pos]:.2f}" if have_sma and not pd.isna(sma50.iloc[pos]) else "",
            "Golden Cross": "TRUE" if (pos == n - 1 and cross == "GOLDEN") else "FALSE",
            "Trend": t or "n/a",
            "ROE %": f"{roe:.1f}" if roe is not None else "n/a",
            "_gap": g,                                 # HTML only; CSV writer ignores _-keys
        })
    return rows, latest_trend, cross


CSV_FIELDS = ["Stock", "Index", "Date", "CMP", "Open", "Close",
              "200DMA", "50DMA", "Golden Cross", "Trend", "ROE %"]


# --------------------------------------------------------------------------- #
# Telegram
# --------------------------------------------------------------------------- #

def telegram(method: str, token: str, **kwargs) -> None:
    try:
        resp = requests.post(TELEGRAM_API.format(token=token, method=method), timeout=60, **kwargs)
    except requests.RequestException as exc:
        raise RuntimeError(f"Telegram {method} failed: {type(exc).__name__}") from None
    if not resp.ok:
        raise RuntimeError(f"Telegram {method} failed: HTTP {resp.status_code} {resp.text[:300]}")


def render_html(stocks: list[dict], session: date, generated: datetime,
                crosses: list[dict], missing: list[str]) -> str:
    """Self-contained dashboard. One block per stock = its last 5 sessions,
    so you can watch the 50/200 gap approach zero before a cross. Blocks are
    ordered nearest-to-crossing first (actual crosses pinned on top).
    No token or secret ever reaches this - only the stock rows passed in."""
    import html as _html

    def esc(v) -> str:
        return _html.escape(str(v))

    blocks = []
    for st in stocks:
        cross = st["cross"]
        badge = {"GOLDEN": '<span class="badge golden">Golden cross today</span>',
                 "DEATH": '<span class="badge death">Death cross today</span>'}.get(cross, "")
        rows = st["rows"]
        prev_trend = None
        day_rows = []
        for r in rows:
            trend = r["Trend"]
            tcls = {"POSITIVE": "pos", "NEGATIVE": "neg"}.get(trend, "muted")
            flip = (prev_trend in ("POSITIVE", "NEGATIVE") and trend in ("POSITIVE", "NEGATIVE")
                    and trend != prev_trend)
            if flip and trend == POSITIVE:
                cross_cell = '<span class="badge golden">Golden cross</span>'
            elif flip and trend == NEGATIVE:
                cross_cell = '<span class="badge death">Death cross</span>'
            else:
                cross_cell = '<span class="muted">—</span>'
            prev_trend = trend
            day_rows.append(
                f'<tr class="{"flip" if flip else ""}">'
                f'<td class="day">{esc(r["Date"])}</td>'
                f'<td class="num">{esc(r["Close"])}</td>'
                f'<td class="num">{esc(r["50DMA"] or "—")}</td>'
                f'<td class="num">{esc(r["200DMA"] or "—")}</td>'
                f'<td class="{tcls}">{esc(trend)}</td>'
                f'<td class="cross">{cross_cell}</td>'
                "</tr>"
            )
        latest = rows[-1]
        blocks.append(
            f'<section class="stock {"has-cross" if cross else ""}">'
            f'<div class="shead"><span class="sym">{esc(st["stock"])}</span>'
            f'<span class="muted idx">{esc(st["index"])}</span>'
            f'<span class="cmp">CMP {esc(latest["CMP"])}</span>'
            f'<span class="roe">ROE {esc(latest["ROE %"])}%</span>{badge}</div>'
            '<div class="tscroll"><table><thead><tr>'
            '<th>Date</th><th>Close</th><th>50 DMA</th><th>200 DMA</th><th>Trend</th><th>Cross</th>'
            f'</tr></thead><tbody>{"".join(day_rows)}</tbody></table></div>'
            '</section>'
        )

    cross_note = ""
    if crosses:
        items = ", ".join(f'{c["symbol"]} ({"golden" if c["cross"]=="GOLDEN" else "death"})' for c in crosses)
        cross_note = f'<p class="crosses">Crossovers today: {esc(items)}</p>'
    miss_note = f'<p class="miss">No price data: {esc(", ".join(missing))}</p>' if missing else ""

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>50/200 DMA screener — Nifty 50 + Midcap 50</title>
<style>
  :root {{ color-scheme: light dark;
    --bg:#0f1420; --panel:#151b2b; --line:#26304a; --ink:#e7ecf5; --muted:#8b97b0;
    --pos:#2ec16b; --neg:#ff5d6c; --gold:#f2c14e; box-sizing:border-box; }}
  * {{ box-sizing:inherit; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font:15px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    padding:env(safe-area-inset-top,0) 16px calc(env(safe-area-inset-bottom,0) + 40px); }}
  header {{ max-width:920px; margin:0 auto; padding:28px 0 6px; }}
  h1 {{ font-size:1.4rem; margin:0 0 4px; letter-spacing:-.01em; }}
  .meta {{ color:var(--muted); font-size:.85rem; }}
  .hint {{ color:var(--muted); font-size:.82rem; margin:8px 0 0; }}
  .crosses {{ color:var(--gold); font-weight:600; margin:10px 0 0; }}
  .miss {{ color:var(--muted); font-size:.8rem; margin:6px 0 0; }}
  .grid {{ max-width:960px; margin:16px auto 0; display:grid; gap:14px;
    grid-template-columns:repeat(auto-fill,minmax(440px,1fr)); }}
  .stock {{ border:1px solid var(--line); border-radius:12px; background:var(--panel);
    overflow:hidden; }}
  .stock.has-cross {{ border-color:var(--gold); }}
  .shead {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap;
    padding:10px 14px; border-bottom:1px solid var(--line); }}
  .sym {{ font-weight:700; }}
  .idx {{ font-size:.78rem; }}
  .cmp {{ margin-left:auto; font-size:.82rem; color:var(--muted); }}
  .roe {{ font-size:.82rem; color:var(--muted); }}
  .tscroll {{ overflow-x:auto; }}
  td.cross {{ text-align:left; }}
  table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }}
  th,td {{ padding:6px 12px; text-align:right; white-space:nowrap; font-size:.88rem; }}
  th {{ font-size:.7rem; letter-spacing:.03em; color:var(--muted); font-weight:600;
    border-bottom:1px solid var(--line); }}
  th:first-child,td.day {{ text-align:left; }}
  td.day {{ color:var(--muted); }}
  tbody tr.flip td {{ background:rgba(242,193,78,.12); }}
  tbody tr.flip td.day {{ box-shadow:inset 3px 0 0 var(--gold); }}
  .muted {{ color:var(--muted); }} .pos {{ color:var(--pos); }} .neg {{ color:var(--neg); }}
  .badge {{ font-size:.72rem; font-weight:700; padding:2px 8px; border-radius:999px; }}
  .badge.golden {{ background:rgba(46,193,107,.18); color:var(--pos); }}
  .badge.death {{ background:rgba(255,93,108,.18); color:var(--neg); }}
</style></head><body>
<header>
  <h1>50/200 DMA screener</h1>
  <div class="meta">Nifty 50 + Nifty Midcap 50 · {len(stocks)} stocks · last close {session:%d %b %Y}
    · generated {generated:%d %b %Y %H:%M} IST</div>
  <p class="hint">Ordered nearest-to-crossing first. A flip in Trend is a cross —
    the flip day is flagged in the Cross column and highlighted.</p>
  {cross_note}{miss_note}
</header>
<div class="grid">{"".join(blocks)}</div>
</body></html>
"""


def alert_text(crosses: list[dict], session: date) -> str:
    lines = [f"50/200 DMA crossover · {session:%d %b %Y} close", ""]
    for c in sorted(crosses, key=lambda x: (x["cross"] != "GOLDEN", x["symbol"])):
        tag = "🟢 GOLDEN" if c["cross"] == "GOLDEN" else "🔴 DEATH"
        roe = f"{c['roe']:.1f}%" if c["roe"] is not None else "n/a"
        lines.append(f"{tag}  {c['symbol']} ({c['index']})  close {c['close']:.2f} · ROE {roe}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--roe-ttl-days", type=int, default=1,
                    help="re-scrape a stock's ROE only if cached value is older than this (default 1)")
    args = ap.parse_args()

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not args.dry_run and not (token and chat_id):
        sys.exit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (or use --dry-run).")

    today = ist_today()
    universe = build_universe()
    symbols = sorted(universe)
    if args.limit:
        symbols = symbols[: args.limit]

    # Prices: one batch, one retry for stragglers.
    frames: dict[str, pd.DataFrame] = {}
    pending = symbols
    for attempt in (1, 2):
        data = download([f"{s}.NS" for s in pending])
        missing = []
        for s in pending:
            fr = frame_for(data, f"{s}.NS")
            fr = completed_sessions(fr, today) if fr is not None else None
            if fr is None or fr.empty:
                missing.append(s)
            else:
                frames[s] = fr
        pending = missing
        if not pending or attempt == 2:
            break
        print(f"No price data for {pending}; retrying in {PRICE_RETRY_WAIT}s", file=sys.stderr)
        time.sleep(PRICE_RETRY_WAIT)

    if not frames:
        msg = f"⚠️ Daily screener: no price data for any of {len(symbols)} stocks."
        print(msg, file=sys.stderr)
        if not args.dry_run:
            telegram("sendMessage", token, data={"chat_id": chat_id, "text": msg})
        return 1

    # ROE: cached, scrape only stale/missing.
    roe_cache = load_json(ROE_CACHE_FILE)
    roe = resolve_roe(sorted(frames), roe_cache, args.roe_ttl_days, today)

    # Build table + detect crosses.
    trend_state = load_json(TREND_FILE)
    all_rows: list[dict] = []
    stock_blocks: list[dict] = []
    new_trend = {s: v for s, v in trend_state.items() if s in universe}
    crosses = []
    for symbol in sorted(frames):
        stored = (trend_state.get(symbol) or {}).get("trend")
        rows, latest, cross = build_rows(symbol, frames[symbol], universe[symbol],
                                         roe.get(symbol), stored)
        all_rows.extend(rows)
        stock_blocks.append({"stock": symbol, "index": "+".join(sorted(universe[symbol])),
                             "cross": cross, "latest_gap": rows[-1].get("_gap"), "rows": rows})
        if latest is not None:
            new_trend[symbol] = {"trend": latest, "as_of": rows[-1]["Date"]}
        if cross:
            crosses.append({"symbol": symbol, "index": "+".join(sorted(universe[symbol])),
                            "close": float(frames[symbol]["Close"].iloc[-1]),
                            "roe": roe.get(symbol), "cross": cross})

    # nearest-to-crossing first: crosses pinned, then smallest |gap|, n/a last
    def order_key(b):
        g = b["latest_gap"]
        return (not b["cross"], g is None, abs(g) if g is not None else 0.0, b["stock"])
    stock_blocks.sort(key=order_key)

    with REPORT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    session = max(date.fromisoformat(r["Date"]) for r in all_rows)
    missing = [s for s in symbols if s not in frames]

    SITE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_HTML.write_text(render_html(
        stock_blocks, session, datetime.now(IST), crosses, missing))

    print(f"{len(frames)} stocks, {len(all_rows)} rows, {len(crosses)} cross(es), "
          f"session {session}. Missing: {missing or 'none'}", file=sys.stderr)

    if args.dry_run:
        print(f"[dry-run] wrote {REPORT_CSV}; state not written.")
        return 0

    # 1) Alert first (the part that matters).
    if crosses:
        telegram("sendMessage", token, data={"chat_id": chat_id, "text": alert_text(crosses, session)})

    # 2) Persist state only after a successful alert.
    save_json(TREND_FILE, new_trend)
    save_json(ROE_CACHE_FILE, roe_cache)

    # 3) Full CSV, silently.
    caption = f"Daily 50/200 DMA screener · {session:%d %b %Y} · {len(frames)} stocks · {len(crosses)} cross(es)"
    if missing:
        caption += f"\nNo data: {', '.join(missing)}"
    with REPORT_CSV.open("rb") as f:
        telegram("sendDocument", token,
                 data={"chat_id": chat_id, "caption": caption[:1024], "disable_notification": "true"},
                 files={"document": (f"screener_{session:%Y%m%d}.csv", f, "text/csv")})
    return 0


if __name__ == "__main__":
    sys.exit(main())