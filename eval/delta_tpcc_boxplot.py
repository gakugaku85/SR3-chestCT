import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# CSVファイルを読み込む
csv_file = "wd.csv"  # 適宜ファイル名を変更
df = pd.read_csv(csv_file)


# データを縦持ち（long format）に変換
df_melted = df.melt(var_name="Category", value_name="#(GTCC)-#(TPCC)")

# Seabornで箱ひげ図を描画
plt.figure(figsize=(6, 6))
sns.boxplot(x="Category", y="#(GTCC)-#(TPCC)", data=df_melted, width=0.6, showmeans=True,  meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": 8})

# グラフのタイトルとラベル
plt.title("Comparison of originalLoss and wdLoss #(GTCC)-#(TPCC)")
plt.xlabel("")
plt.ylabel("#(GTCC)-#(TPCC)")
plt.savefig('boxplot.png')