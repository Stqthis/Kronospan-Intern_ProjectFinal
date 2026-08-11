from jarvisman.planning import code_resolver

# grab the live dataframes the same way the app does — adjust if your app
# exposes them differently
import pandas as pd, glob, os

# EITHER point at your two excel files directly:
files = glob.glob(os.path.expanduser("~/**/*.xlsx"), recursive=True)
print("Excel files found:", len(files))
dfs = {}
for f in files:
    try:
        dfs[os.path.basename(f)] = pd.read_excel(f)
    except Exception as e:
        print("skip", f, e)

print("\n-- mapping tables the resolver can see --")
for df, code_c, name_c in code_resolver._mapping_columns(dfs):
    print("  code col:", code_c, "| name col:", name_c, "| rows:", len(df))

print("\n-- resolve NT01 --")
name = code_resolver.resolve_code("NT01", dfs)
print("  resolved to:", repr(name))

print("\n-- does looks_like_code fire? --")
print("  NT01 ->", code_resolver.looks_like_code("NT01"))