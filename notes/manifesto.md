# Quant Research Developer Manifesto

## Purpose

This document defines the principles and boundaries that govern development within this quant trading system. It is intended to minimize churn, prevent repeated rewrites, and promote disciplined, scientific experimentation—like a professional quant would approach strategy development.

---

## I. Project Philosophy

1. **The Engine is a Platform, Not a Strategy**
   The core system (data loader, backtester, model interfaces, logging, and metrics) exists to enable experimentation—not to be rewritten with every new idea.

2. **Experimentation is Modular**
   New ideas must live in isolated modules (e.g., new alpha models or execution styles). Do not alter `core/` just to test something.

3. **Code Must Serve the Research Loop**
   This system should make it easy to:

   * Load data
   * Generate signals
   * Construct portfolios
   * Execute trades
   * Measure performance

---

## II. Daily Research Workflow

1. **Start With a Hypothesis**
   Every session begins with a question:

   > “What would happen if I used X factor in my alpha model?”

2. **Work in a Branch**
   Create a Git branch named after the experiment:
   `git checkout -b exp-meanrev-volatility`

3. **Code in Isolation**
   Place experimental models in `experiments/` or as new files in `models/alpha/`, `models/risk/`, etc. Only interface with the core system.

4. **Backtest Immediately**
   Run the end-to-end system with the new component plugged in.
   Record metrics, trades, and observations.

5. **Log Results in a Lab Notebook**
   Append a markdown entry in `research/lab_notes.md`:

   ```markdown
   ## 2025-05-12 – Mean-Reversion Volatility Alpha
   - Hypothesis: Volatility-normalized price deviations mean-revert
   - Method: `alpha/meanrev_vol.py`
   - Sharpe: 0.89, MaxDD: -7.2%
   - Notes: Performs better in high-vol environments
   ```

6. **Decide: Promote or Archive**
   If the model is promising, consider promoting it to the main model suite. If not, tag the branch and leave it.

---

## III. Cursor and Code Tools Discipline

1. **No Bulk Refactor on Uncommitted Code**
   Never let Cursor apply wide-scope changes unless current changes are committed.

2. **No Deleting Working Code**
   If the code works and is used, refactor only with justification.

3. **Never Rewrite Architecture Without a Postmortem**
   If a rewrite is needed, document why the last system failed in `notes/postmortems.md` first.

---

## IV. When In Doubt, Preserve Progress

* Tag good states: `git tag v1.0.0`
* Freeze good models: `models/alpha/trend_following_v1.py`
* Keep everything that worked, even if imperfect.

Working code is more valuable than ideal code.

---

## V. Closing Principle

> "Build your tools like you're going to live with them. Because you are."

This is not an art project. It’s a research machine.
The measure of progress is **new, tested insight**, not beautiful architecture.

Stick to this, and you’ll outpace 90% of undisciplined quant devs.
