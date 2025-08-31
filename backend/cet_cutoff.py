import pandas as pd
from tabulate import tabulate

xlsx_file = "data/mht_cet_cutoff.xlsx"  
txt_file = "data/college.txt"

df = pd.read_excel(xlsx_file)


table_str = tabulate(df, headers="keys", tablefmt="grid", showindex=False)


with open(txt_file, "a", encoding="utf-8") as f:
    f.write("\n\n")  # spacing before new data
    f.write(table_str)
    f.write("\n")
