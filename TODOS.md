# TODOS

## High priority — app.py

- [x] **Breach: show factor breakdown chart**
  - What: After running breach model, show a bar chart of what's driving the score (sector weight, size, peer incidents, growth, beta).
  - Why: Executives want to know WHY a company scores high, not just that it does. "Your sector risk accounts for 30 points" is actionable. A number alone isn't.

- [x] **All models: make graphs bigger and more prominent**
  - What: Move histogram above the metrics row, or give it more vertical space (figsize height 5+). Add a subtitle explaining what the x-axis means in plain English.
  - Why: The distribution IS the point of Monte Carlo. It should be the first thing you see, not below four metric boxes.

- [x] **BreachIQ consulting CTA — opt-in only**
  - What: After breach + cost results, add a subtle "Want a full assessment?" link to breachiq.io. No email collection in the app itself.
  - Why: The app is a free tool. If they want to go deeper, BreachIQ is the product. Don't ask for their email — send them to the site where they can choose to engage.

## Medium priority

- [x] **`operating_margin_phase2` in DCFModel**
  - What: Add optional `operating_margin_phase2: tuple[float,float,float] | None = None` param to DCFModel.
  - Why: High-growth companies typically have compressed margins in phase 1 (heavy R&D/sales investment) that expand as they mature. A single margin distribution covers both phases by widening the range, which understates the phase-structure and overstates early-phase FCF.
  - How to apply: Same Triangular pattern as `capex_pct_revenue`. If None, use `operating_margin` for both phases.
  - Depends on: two-stage DCF already landed

- [x] **DCF: add revenue input guidance**
  - What: Add helper text or example under the revenue field: "Where to find this: 10-K income statement, Yahoo Finance financials tab."
  - Why: Students don't know where to get the number. One line of help text removes the friction.
