"""Collect additional REAL Mozilla Bugzilla reports from the public REST API.

Usage:
    python scripts/collect_bugzilla.py                   # default: 2006-01-01 .. 2009-01-01
    python scripts/collect_bugzilla.py --start 2007-01-01 --end 2007-07-01 --out data/x.csv

Source: https://bugzilla.mozilla.org/rest/bug (public, read-only, no login).
Selection mirrors the original ``summaryList.csv`` sample, whose 99 rows were all
verified against the live API: products Core / Core Graveyard / Firefox Build
System, status VERIFIED, created 2007-01-02 .. 2007-02-11.

Label provenance
----------------
Bugzilla moved "enhancement" out of the *severity* field into a separate *type*
field (the 6 original ``enhancement`` rows are now ``severity=normal,
type=enhancement``). To keep the ORIGINAL label vocabulary we store:

* ``severity_live`` / ``type_live``  - exactly what the API returns today
* ``severity``                       - the original-vocabulary label:
  ``enhancement`` if ``type_live == "enhancement"`` and ``severity_live == "normal"``,
  otherwise ``severity_live`` unchanged.

Labels outside the original vocabulary (e.g. the post-2020 ``S1``..``S4``) are kept
in the file as-is; the training loader drops and counts them, it never guesses.

Privacy: ``assigned_to`` / reporter e-mail addresses are NOT requested or stored.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

API = "https://bugzilla.mozilla.org/rest/bug"
PRODUCTS = ["Core", "Core Graveyard", "Firefox Build System"]
FIELDS = ["id", "summary", "severity", "type", "status", "resolution",
          "product", "component", "creation_time"]
USER_AGENT = "severity-prediction-research/1.0 (academic, read-only)"
ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = ROOT / "summaryList.csv"


def original_label(severity: str, bug_type: str) -> tuple[str, str]:
    """Return (label in the original vocabulary, how it was derived)."""
    if bug_type == "enhancement" and severity == "normal":
        return "enhancement", "type=enhancement"
    return severity, "severity"


def month_windows(start: date, end: date):
    cur = start
    while cur < end:
        nxt = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        yield cur, min(nxt, end)
        cur = nxt


def fetch(lo: date, hi: date, retries: int = 3) -> list[dict]:
    params = {"product": PRODUCTS, "bug_status": "VERIFIED",
              "f1": "creation_ts", "o1": "greaterthaneq", "v1": lo.isoformat(),
              "f2": "creation_ts", "o2": "lessthan", "v2": hi.isoformat(),
              "include_fields": ",".join(FIELDS), "limit": 0}
    url = API + "?" + urllib.parse.urlencode(params, doseq=True)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["bugs"]
        except Exception as exc:  # network hiccup: back off and retry
            if attempt == retries - 1:
                raise
            print(f"  retry {lo}: {exc}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    return []


def original_ids() -> set[int]:
    with open(ORIGINAL, encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f, quotechar="'")
        return {int(r["bug_id"]) for r in rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2006-01-01")
    ap.add_argument("--end", default="2009-01-01", help="exclusive")
    ap.add_argument("--out", default=str(ROOT / "data" / "bugzilla_core_verified_2006_2008.csv"))
    args = ap.parse_args(argv)
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    orig = original_ids()
    bugs: dict[int, dict] = {}
    for lo, hi in month_windows(start, end):
        got = fetch(lo, hi)
        for b in got:
            bugs[b["id"]] = b
        print(f"{lo}..{hi}: {len(got)} bugs (total {len(bugs)})")
        time.sleep(1.0)  # be polite to the public server

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["bug_id", "summary", "severity", "label_source", "severity_live", "type_live",
            "status", "resolution", "product", "component", "creation_time", "in_original_sample"]
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(cols)
        for bid in sorted(bugs):
            b = bugs[bid]
            label, source = original_label(b["severity"], b["type"])
            w.writerow([bid, b["summary"], label, source, b["severity"], b["type"], b["status"],
                        b["resolution"], b["product"], b["component"], b["creation_time"],
                        int(bid in orig)])

    prov = {"source": API, "products": PRODUCTS, "bug_status": "VERIFIED",
            "created_from": args.start, "created_before": args.end,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rows": len(bugs), "rows_also_in_original_sample": sum(b in orig for b in bugs),
            "fields": FIELDS, "label_rule": "enhancement if type=enhancement and severity=normal, else live severity",
            "emails_stored": False}
    out.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(f"Wrote {len(bugs)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
