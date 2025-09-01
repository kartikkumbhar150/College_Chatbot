import pandas as pd
from tabulate import tabulate

# Path to your xlsx file
xlsx_file = "data/mht_cet_cutoff.xlsx"   # change this to your actual file
# Path to your text file
txt_file = "data/college.txt"

# Read the Excel file
df = pd.read_excel(xlsx_file)

# Convert DataFrame to a pretty table string
table_str = tabulate(df, headers="keys", tablefmt="grid", showindex=False)

# Append to the text file
with open(txt_file, "a", encoding="utf-8") as f:
    f.write("\n\n")  # spacing before new data
    f.write(table_str)
    f.write("\n")
