import sys, pandas as pd

f1 = sys.argv[1]   # mapping file: Company_code + Company
f2 = sys.argv[2]   # amounts file: Company + AMOUNT EURO

df1 = pd.read_excel(f1)
df2 = pd.read_excel(f2)
print("File 1 columns:", list(df1.columns))
print("File 2 columns:", list(df2.columns))

def norm(s):
    return set(s.dropna().astype(str).str.strip().str.upper())

# find the name column in each (case-insensitive 'company', not 'company_code')
def name_col(df):
    for c in df.columns:
        cu = c.strip().upper()
        if cu == "COMPANY" or (("COMPANY" in cu or "NAME" in cu) and "CODE" not in cu):
            return c
    return None

n1, n2 = name_col(df1), name_col(df2)
print("\nName column in File 1:", n1)
print("Name column in File 2:", n2)

if n1 and n2:
    v1, v2 = norm(df1[n1]), norm(df2[n2])
    shared = v1 & v2
    print(f"\nFile1 distinct names: {len(v1)}")
    print(f"File2 distinct names: {len(v2)}")
    print(f"SHARED (overlap): {len(shared)}")
    print("Sample File1:", sorted(v1)[:5])
    print("Sample File2:", sorted(v2)[:5])
    print("Sample SHARED:", sorted(shared)[:5])
