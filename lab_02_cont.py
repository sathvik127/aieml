# A5
print("A5")
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
def find_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
    return outliers

df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
df.replace('?', np.nan, inplace=True)
print("the data types of the following data are:\n",df.dtypes)
dtypes_o=np.array(df.select_dtypes(include=['object']).columns)
print(dtypes_o)
print("for this types of data we can use one hot encoding as it is nominal data")
dtypes_i=np.array(df.select_dtypes(include=['int64']).columns)
dtypes_f=np.array(df.select_dtypes(include=['float64']).columns)
numdata_c=[dtypes_f,dtypes_i]
print(dtypes_i)
print("for this types of data we can use lable encoding as it is ordinal data")
dtypes_i_full_col=np.array(df.select_dtypes(include=['int64']))
int_data=np.array(dtypes_i_full_col)
# print(int_data)
range_c1=int_data[:,0:1]
print("range for numerical data in column1 is:",np.max(range_c1)-np.min(range_c1))
range_c2=int_data[:,1:2]
# print(range_c2)
print("range for numerical data in column2 is:",np.max(range_c2)-np.min(range_c2))
dtypes_f_full_col=np.array(df.select_dtypes(include=['float64']))
float_data=np.array(dtypes_f_full_col)
# print(float_data)
for i in range(6):
    print("range for numerical data in column",i+1," is:", np.max(float_data[:,i:i+1]) - np.min(float_data[:,i:i+1]))
dfarray=np.array(df)
cols_with_nan = df.columns[df.isna().any()].tolist()
print("these are the cols with nan values",cols_with_nan)

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
outlier_cols = [col for col in numeric_cols if any(find_outliers(df[col].dropna()))]
print("Numeric Columns with Outliers:", outlier_cols)
print("Numeric Columns without Outliers:", list(set(numeric_cols) - set(outlier_cols)))
for i in df[numeric_cols]:
    print("mean for numeric values",np.mean(np.array(df[i])))

for i in df[numeric_cols]:
    print("std for numeric values",np.std(np.array(df[i])))
#------------------------------------------------------------------------------------------------------------#
# A6
print("A6")

df1=df[outlier_cols]
print("numeric data with outliners\n")
print(df1)
df1.fillna(df1.median(), inplace=True)
print("numeric data with outliners after filling\n")
df[outlier_cols]=df1
print(df1)
df2=df[list(set(numeric_cols) - set(outlier_cols))]
print("numeric data without outliners\n")
print(df2)
df2.fillna(df2.mean(), inplace=True)
print("numeric data without outliners after filling\n")
df[list(set(numeric_cols) - set(outlier_cols))]=df2
print(df2)
catog=(df.select_dtypes(include=['object']).columns)
print(catog)
df3=df[catog]
print("categorical data\n")
print(df3)
df3.fillna(df3.mode(), inplace=True)
print("categorical data after filling\n")
df[catog]=df3
print(df3)
# -------------------------------------------------------------------------------------------------------#
# A7
print("A7")

df_scaled = df
for col in numeric_cols:
    if col in outlier_cols:
        df_scaled[col] = RobustScaler().fit_transform(df_scaled[[col]])
    else:
        df_scaled[col] = MinMaxScaler().fit_transform(df_scaled[[col]])

print("Scaled data:\n", df_scaled.head())
# --------------------------------------------------------------------------------------------------------------------#
# A8
print("A8")

rowone=df[catog].replace('t', 1, inplace=False)
row1=rowone.replace('f', 0, inplace=False)
obs1=list((np.array(row1)[1]))
obs2=list((np.array(row1)[2]))
# print(obs1)
o1=[]
o2=[]
for i in range(len(obs1)):
    if obs1[i]==0 or obs1[i]==1:
        o1.append(obs1[i])
        o2.append(obs2[i])
# print(o1)
# print(o2)
oo=0
ii=0
io=0
oi=0
for i in range(len(o1)):
    if o1[i]==1 and o2[i]==1:
        ii+=1
    if o1[i] == 1 and o2[i] == 0:
        io += 1
    if o1[i] == 0 and o2[i] == 1:
        oi += 1
    if o1[i] == 0 and o2[i] == 0:
        oo += 1

print("jaccard index",(ii+oo)/(ii+oo+oi+io))
print("Simple Matching Coefficient",(ii)/(ii+io+oi))

# ------------------------------------------------------------------------------------------------------------#
# A9
print("A9")

numm=df[numeric_cols]
i1=np.array(numm)[1]
i2=np.array(numm)[2]
# print(i1)
x1=np.append(o1,np.array(i1))
x2=np.append(o2,np.array(i2))
# print(x1)
magx1=np.linalg.norm(x1)
magx2=np.linalg.norm(x2)
mag3=magx1+magx2
print(magx1)
print("the cosine similarity of the two vectors:",np.dot(x1,x2)/mag3)
# -------------------------------------------------------------------------------------------------------------------------------------------#
# A10
print("A10")

df_subset = df.iloc[:20]
binary_cols = [col for col in df.columns if set(df[col].dropna().unique()).issubset({0, 1})]
df_binary = df_subset[binary_cols].astype(int)
df_numeric = df_subset.select_dtypes(include=['int64', 'float64']).fillna(0)
df_values = df_numeric.to_numpy()

jc_matrix = np.zeros((20, 20))
smc_matrix = np.zeros((20, 20))
cos_matrix = cosine_similarity(df_values)
def compute_jc_smc(vec1, vec2):
    intersection = np.sum((vec1 == 1) & (vec2 == 1))
    union = np.sum((vec1 == 1) | (vec2 == 1))
    matches = np.sum(vec1 == vec2)
    total_attributes = len(vec1)
    jc = intersection / union if union != 0 else 0
    smc = matches / total_attributes
    return jc, smc

for i in range(20):
    for j in range(20):
        jc_matrix[i, j], smc_matrix[i, j] = compute_jc_smc(df_binary.iloc[i].values, df_binary.iloc[j].values)
jc_df = pd.DataFrame(jc_matrix, index=range(1, 21), columns=range(1, 21))
smc_df = pd.DataFrame(smc_matrix, index=range(1, 21), columns=range(1, 21))
cos_df = pd.DataFrame(cos_matrix, index=range(1, 21), columns=range(1, 21))
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.heatmap(jc_df, ax=axes[0], cmap="coolwarm", annot=False)
axes[0].set_title("Jaccard Coefficient (JC)")
sns.heatmap(smc_df, ax=axes[1], cmap="coolwarm", annot=False)
axes[1].set_title("Simple Matching Coefficient (SMC)")
sns.heatmap(cos_df, ax=axes[2], cmap="coolwarm", annot=False)
axes[2].set_title("Cosine Similarity (COS)")
plt.tight_layout()
plt.show()