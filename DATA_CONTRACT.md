# Data-drop contract (monthly exports)

The assistant is exactly as good as the snapshots it holds. This one-page
contract keeps the feed healthy.

## What to drop, where
One folder (mounted as `/files` in the api container). Monthly, as-of month
end:
- `CY01-DDR-<dd-mm-yyyy>-DATA.xlsx` — daily deposit report
- `CY05-Group_Company-<mm-yyyy>-DATA.xlsx` — companies, directors, addresses
- `WCR_<dd_mm_yyyy>.xlsx` — working-capital facilities
- `LTL_Data.xlsx` — long-term loans daily series (full refresh)
- `CY28-3RD-PARTY-<dd-mm-yyyy>-DATA.xlsx` — third-party facilities
PDF report versions are welcome as extra (they feed text search) but are NOT
a substitute for the DATA .xlsx exports — comparisons across dates need the
tables.

## Rules
1. **The date in the file name must match the report date inside.** The
   backend validates this automatically and raises a warning on the System
   page / web UI when they disagree (this has happened: a July-named file
   containing December data).
2. Keep entity spellings stable (e.g. "Kronospan CR, spol s r.o." with or
   without the dot — pick one).
3. Never edit a dropped file in place; replace it with a corrected copy
   (the watcher picks up the change automatically).
4. DATA exports must contain the raw table, not a pivot of it.

## What happens automatically
The backend watches the folder (default every 60 s): new or changed files
are date-validated and indexed incrementally; the System page and web UI
show the updated inventory and any warnings. No manual re-index step.
