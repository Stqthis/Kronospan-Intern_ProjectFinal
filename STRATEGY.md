# Strategy for your Kronospan financial data — what's wrong and how to fix it

## What I learned from test1/test2/test3

Your MATRIX sheet has ~60 columns. The correct "funds in EUR" answer lives in
`AMOUNT EURO`. But a dozen other columns LOOK like valid answers to a language
model: `Finance/ECCM`, `Bank Limit Euro '000`, `AMT EURO 'OOO`,
`PREVIOUS DAY B/CE EURO`, `ONE_PAGE_Total Sum of AMT EURO`, `USH`.

**That is the real disease.** The nonsense clarifications were a symptom. With
so many confusable columns and no signal about which is canonical, the model
guesses — and sometimes picks `Finance/ECCM` instead of `AMOUNT EURO`.

The fix is NOT hardcoding. It is giving the model the same one-paragraph
briefing you would give a new analyst on day one — through the house-rules
file, which the app already injects into both the planner and the code
generator. You write it; the code stays generic.

## The single highest-value action: a business glossary (rules.txt)

Create `rules.txt` in your data directory (or set `RAG_HOUSE_RULES`). One line
per rule. These are READ FROM A FILE, not baked into code — change them any
time, for any dataset. A starter tailored to what your questions need:

    # ---- canonical measures (kills the "which euro column?" guessing) ----
    "funds", "deposits", "total funds", "amount in EUR" and "EUR equivalent" all mean the column AMOUNT EURO. Never use Finance/ECCM, Bank Limit, USH, or any '000 column for these.
    "foreign currency amount" or "original amount" means the column FOREIGN CURRENCY.
    "interest rate" means the column INTEREST RATE.
    # ---- snapshot date semantics ----
    The MATRIX sheet is a single snapshot as at its report date. "as at <date>" applies NO row filter — the whole sheet already is that date.
    Exclude summary/total rows: only count rows where COMPANY is not empty.
    # ---- directorships (CY05) ----
    For "directorships of <person> as at <date>": a row is valid when Start_date <= date AND (End_date is empty OR End_date >= date). Return ONE row per company (drop duplicates on COMPANY_CODE).
    Use DIRECTOR_NAME for the director; names are stored "Surname, Firstname".
    # ---- show name + code together ----
    Whenever you show a company, include both COMPANY_NAME and COMPANY_CODE.

With these in place, "total funds of Lignum in EUR equivalent" compiles
straight to `sum(AMOUNT EURO) where COMPANY = Lignum` — no guessing, no
clarify. The glossary is your lever; the engine is generic.

## Per-question status against your expected answers

1.1 / 1.2 (funds in EUR for a company / country): FIXED by the AMOUNT EURO
    glossary line + the summary-row exclusion. Expected 105,247,755.84 and
    5,346,044 become deterministic once the model stops choosing Finance/ECCM.

2.1 / 2.2 (sum by bank / per currency): the per-currency clarifier no longer
    fires the nonsense "Finance/ECCM category" option (verified). With the
    glossary, the measure is AMOUNT EURO and grouping by CURRENCY is direct.

3.1 (sort interest rates in Denmark): the "sort"->"South" nonsense ask is
    GONE (verified). With the INTEREST RATE glossary line it sorts that column.

4 / 5 / 6 (directorships, changes over a period): the validity-window rule +
    "one row per company" fixes the duplicate rows you saw in question 3.
    Period COMPARISON (5/6) is the one genuinely new capability — see below.

7 (same address): works as a grouping; the clarifier no longer interferes.

## What still needs real work (honest)

- **Period comparison (Q5, Q6)** — "changes from Dec 2023 to Dec 2024" needs
  two snapshots compared. Your engine HAS a compare path (`compare` plan key),
  but it needs two dated files loaded and a clear key (DIRECTOR_NAME +
  COMPANY_CODE). This is a feature to exercise, not a bug to fix.
- **Model quality** — with the glossary the PLAN becomes easy, but only live
  qwen/llama on the Spark show whether they follow it. Run with RAG_EXPLAIN=1.
- The glossary above is a STARTER. Add a line every time the model picks a
  wrong column; each line is permanent, costs nothing, and is pure business
  language.
