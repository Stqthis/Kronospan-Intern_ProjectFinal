import pickle, os
from jarvisman import config as cfg

t = pickle.load(open(os.path.join(cfg.INDEX_DIR, 'tables.pkl'), 'rb'))
mat = qry = None
for n, df in t.items():
    cols = [str(c) for c in df.columns]
    if 'AMOUNT EURO' in cols:
        mat = df
    if 'COMPANY_NAME' in cols and 'COMPANY_COUNTRY_NAME' in cols:
        qry = df

COUNTRY = "Croatia"          # <-- put the real country name here

names = qry.loc[
    qry['COMPANY_COUNTRY_NAME'].astype(str).str.strip()
       .str.contains(COUNTRY, case=False, na=False),
    'COMPANY_NAME'
].dropna().unique()

rows = mat[mat['COMPANY'].astype(str).str.strip().isin(names)]

print(f"companies matched: {len(names)}")
print(f"MATRIX rows:       {len(rows)}")
print(f"AMOUNT EURO sum:   {rows['AMOUNT EURO'].sum():,.2f}")
