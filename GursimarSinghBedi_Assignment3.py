import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(
    "https://raw.githubusercontent.com/resbaz/r-novice-gapminder-files/master/data/gapminder-FiveYearData.csv")
print("HEAD")
print(df.head())  # to read the first five lines of the data
print("\nCOLUMN DESCRIPTIONS")
print(df.describe())  # to gain insights about the columns and make sure that there are no absurd values
print("\nNULL VALUES")
print(df.isnull().sum())
table = pd.pivot_table(df, values='lifeExp', index=['country'], columns=['year'])
print("\nPIVOT TABLE")
print(table)
sns.heatmap(table)
plt.show()
