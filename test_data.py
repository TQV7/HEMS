from data import build_merged_df

df = build_merged_df()

print(df.head())
print()
print(df.columns)
print()
print(df.shape)
print()
print(df.isna().sum())