"""Monte Carlo simulation CLI.

Usage:
    python -m mc breach --sector Healthcare --market-cap 5000 --beta 1.1 --growth 0.10
    python -m mc cost   --market-cap 5000 --sector Healthcare --breach-score 65
    python -m mc dcf    --revenue 500 --growth 0.08,0.14,0.20 --margin 0.10,0.15,0.22 \
                        --discount 0.10,0.02 --terminal 0.025,0.005

Add --json to any subcommand for machine-readable output.
"""
from __future__ import annotations
import argparse
import json
import sys


def _parse_triple(s: str) -> tuple[float, float, float]:
    parts = s.split(",")
    if len(parts) != 3:
        raise ValueError(
            f"Expected 'low,mode,high' (3 comma-separated values), got: {s!r}"
        )
    try:
        vals = tuple(float(p.strip()) for p in parts)
    except ValueError:
        raise ValueError(f"Could not parse floats from: {s!r}")
    return vals  # type: ignore[return-value]


def _parse_pair(s: str) -> tuple[float, float]:
    parts = s.split(",")
    if len(parts) != 2:
        raise ValueError(
            f"Expected 'mean,std' (2 comma-separated values), got: {s!r}"
        )
    try:
        vals = tuple(float(p.strip()) for p in parts)
    except ValueError:
        raise ValueError(f"Could not parse floats from: {s!r}")
    return vals  # type: ignore[return-value]


def _format_result(result: dict, title: str) -> str:
    unit = result.get("unit", "")
    suf = f" {unit}" if unit else ""
    lo, hi = result["ci_90"]

    lines = [f"\n{title}", "-" * max(len(title), 30)]

    if "tier" in result:
        lines.append(f"  Tier:           {result['tier']}")

    lines.append(f"  p10:            {result['p10']}{suf}")
    lines.append(f"  p50 (median):   {result['p50']}{suf}")
    lines.append(f"  p90:            {result['p90']}{suf}")
    lines.append(f"  90% CI:         {lo}-{hi}{suf}")

    if "prob_critical_pct" in result:
        lines.append(f"  Prob Critical:  {result['prob_critical_pct']}%")

    if result.get("net_debt", 0) != 0:
        nd = result["net_debt"]
        lines.append(f"\n  Net debt:       ${nd:,.1f}M")
        lines.append(f"  Equity p10:     {result['equity_p10']}{suf}")
        lines.append(f"  Equity p50:     {result['equity_p50']}{suf}")
        lines.append(f"  Equity p90:     {result['equity_p90']}{suf}")

    if "narrative" in result:
        lines.append(f"\n  {result['narrative']}")

    return "\n".join(lines)


def _clean(result: dict) -> dict:
    """Strip non-JSON-serialisable numpy arrays (scores, pnl_scores)."""
    return {k: v for k, v in result.items() if not hasattr(v, "dtype")}


def _cmd_breach(args: argparse.Namespace) -> dict:
    from mc.models.breach import BreachLikelihoodModel
    company_info = {
        "name":         args.name,
        "sector":       args.sector,
        "market_cap_m": args.market_cap,
        "beta":         args.beta,
        "rev_growth":   args.growth,
    }
    peers = [{"ticker": f"PEER{i}"} for i in range(args.peers)]
    return BreachLikelihoodModel(company_info, peers, n=args.n, seed=args.seed).run()


def _cmd_cost(args: argparse.Namespace) -> dict:
    from mc.models.cyber_cost import CyberCostModel
    return CyberCostModel(
        market_cap_m=args.market_cap,
        sector=args.sector,
        breach_score=args.breach_score,
        n=args.n,
        seed=args.seed,
    ).run()


def _cmd_dcf(args: argparse.Namespace) -> dict:
    from mc.models.dcf import DCFModel
    growth2 = _parse_triple(args.growth2) if args.growth2 else None
    capex   = _parse_triple(args.capex) if args.capex else None
    return DCFModel(
        revenue=args.revenue,
        revenue_growth=_parse_triple(args.growth),
        operating_margin=_parse_triple(args.margin),
        discount_rate=_parse_pair(args.discount),
        terminal_growth=_parse_pair(args.terminal),
        years=args.years,
        n=args.n,
        seed=args.seed,
        revenue_growth_phase2=growth2,
        phase1_years=args.phase1_years,
        capex_pct_revenue=capex,
        net_debt=args.net_debt,
    ).run()


def main() -> None:
    # Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError for
    # non-ASCII chars in narratives and CI ranges).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="python -m mc",
        description="Monte Carlo risk simulation — breach likelihood, cyber cost, DCF valuation",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    def _add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--n", type=int, default=10_000, metavar="N",
                       help="Simulations (default: 10000)")
        p.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
        p.add_argument("--json", action="store_true",
                       help="Output raw JSON")

    # ── breach ────────────────────────────────────────────────────────────────
    pb = sub.add_parser("breach", help="Breach likelihood score 0–100")
    _add_common(pb)
    pb.add_argument("--sector",     required=True)
    pb.add_argument("--market-cap", type=float, required=True, metavar="M",
                    help="Market cap in $M")
    pb.add_argument("--beta",       type=float, required=True)
    pb.add_argument("--growth",     type=float, required=True, metavar="RATE",
                    help="Annual revenue growth (e.g. 0.10 for 10%%)")
    pb.add_argument("--peers",      type=int,   default=0,
                    help="Peer companies with disclosed breaches (default: 0)")
    pb.add_argument("--name",       default="Company")

    # ── cost ──────────────────────────────────────────────────────────────────
    pc = sub.add_parser("cost", help="Cyber breach cost in $M (if breach occurs)")
    _add_common(pc)
    pc.add_argument("--market-cap",    type=float, required=True, metavar="M")
    pc.add_argument("--sector",        default="Healthcare")
    pc.add_argument("--breach-score",  type=float, default=None, metavar="SCORE",
                    help="Breach likelihood score 0–100 (from 'breach' subcommand)")

    # ── dcf ───────────────────────────────────────────────────────────────────
    pd_ = sub.add_parser("dcf", help="DCF enterprise valuation in $M")
    _add_common(pd_)
    pd_.add_argument("--revenue",      type=float, required=True, metavar="M",
                     help="Current annual revenue in $M")
    pd_.add_argument("--growth",       required=True, metavar="L,M,H",
                     help="Revenue growth Triangular: low,mode,high e.g. 0.08,0.14,0.20")
    pd_.add_argument("--margin",       required=True, metavar="L,M,H",
                     help="Operating margin Triangular: low,mode,high e.g. 0.10,0.15,0.22")
    pd_.add_argument("--discount",     required=True, metavar="MEAN,STD",
                     help="Discount rate Normal: mean,std e.g. 0.10,0.02")
    pd_.add_argument("--terminal",     required=True, metavar="MEAN,STD",
                     help="Terminal growth Normal: mean,std e.g. 0.025,0.005")
    pd_.add_argument("--years",        type=int,   default=10)
    pd_.add_argument("--growth2",      default=None, metavar="L,M,H",
                     help="Phase-2 revenue growth — enables two-stage DCF")
    pd_.add_argument("--phase1-years", type=int,   default=None, metavar="YEARS",
                     help="Phase-1 duration in years (required with --growth2)")
    pd_.add_argument("--capex",        default=None, metavar="L,M,H",
                     help="CapEx as %% of revenue Triangular e.g. 0.04,0.06,0.10")
    pd_.add_argument("--net-debt",     type=float, default=0.0, metavar="M",
                     help="Net debt $M — adds equity bridge to output (negative = net cash)")

    args = parser.parse_args()

    try:
        if args.command == "breach":
            result = _cmd_breach(args)
            title  = f"Breach Likelihood — {args.name} ({args.sector})"
        elif args.command == "cost":
            result = _cmd_cost(args)
            title  = f"Cyber Breach Cost — {args.sector} (${args.market_cap:,.0f}M cap)"
        else:
            result = _cmd_dcf(args)
            title  = f"DCF Valuation — Revenue ${args.revenue:,.0f}M"
    except (ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    clean = _clean(result)
    if args.json:
        print(json.dumps(clean, indent=2))
    else:
        print(_format_result(clean, title))


if __name__ == "__main__":
    main()
