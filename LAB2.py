import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def load_data(file_path):
    sheets = pd.ExcelFile(file_path)
    purchase_data = sheets.parse('Purchase data')
    stock_data = sheets.parse('IRCTC Stock Price')
    thyroid_data = sheets.parse('thyroid0387_UCI')
    return purchase_data, stock_data, thyroid_data
# A1
print("A1")
def dividing_as_AC(matrix):
    A=matrix[:,0:3]
    B=matrix[:,3:4]
    return A,B
def rank(A):
    return np.linalg.matrix_rank(A)
df = pd.read_excel("Lab Session Data.xlsx")
# print(df)
matrix = df.iloc[:, 1:5].apply(pd.to_numeric, errors='coerce').to_numpy()
a,c=dividing_as_AC(matrix)
rank=rank(a)
print("the dimentionality of the vector space is:",rank)
print("the number of vectors exists in the vectro space are",np.shape(a[1]))
ainv=np.linalg.pinv(a)
ainv=np.array(ainv)
c=np.array(c)
X=ainv @ c
print("The individual prices",X)
# A2
print("A2")
print("verification/prediction of the prices",a @ X)
# A3
print("A3")
classi = [0] * c
for i in range(len(c)):
    if c[i]>200:
        classi[i]=1
classi=np.array(classi)
Xc=ainv @ classi

for i in a:
    dot=np.dot(i,Xc)
    if dot > 0.5:
        print("1-rich")
    else:
        print("0-poor")
# print("hello",np.dot(a[1],Xc))
# A4
print("A4")
def mean_calc(data):
    return np.mean(data)

def varience_calc(data):
    return np.var(data)

purchase,stock,thyroid=load_data("Lab Session Data.xlsx")
df1=stock
stock1=df1.iloc[:, 1:9].to_numpy()
# print(stock1)
stock_purchase=stock1[:, 2:3]
print("mean of the price",mean_calc(stock_purchase))
print("varience of the price",varience_calc(stock_purchase))
wed_data=[]
for i in stock1:
    if i[1]=="Wed":
        wed_data.append(i)
wed_data=np.array(wed_data)
# print(wed_data)
stock_purchase_wed=wed_data[:, 2:3]
print("mean of the wednesday purchase data",mean_calc(stock_purchase_wed))
print("the mean is almost similar to population mean just a bit less")
april_data=[]
for i in stock1:
    if i[0]=="Apr":
        april_data.append(i)
april_data=np.array(april_data)
# print(april_data)
stock_purchase_apr=april_data[:, 2:3]
print("mean of the april purchase data",mean_calc(stock_purchase_apr))
print("the mean is almost similar to population mean just a bit more")
chg=stock1[:,7:8]
neg=0
for i in chg:
    if i<0:
        neg+=1
loss_prob=neg/len(chg)
print("probability of loss is:",loss_prob)
chg_wed=wed_data[:,7:8]
pos=0
for i in chg_wed:
    if i>0:
        pos+=1
profit_prob=pos/len(chg)
print("probability of profit is on a wednes day is:",profit_prob)
wed_prob=len(wed_data)/len(stock1)
print("probability of getting a wednes day is:",wed_prob)
print("the required conditional probability is:",profit_prob/wed_prob)
days=stock1[:,1:2]
day=[]
ll=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
for i in days:
    day.append(ll.index(i))
# print(stock1)
chg_plot=stock1[:,7:8]
plt.figure(figsize=(10, 5))
plt.scatter(day,chg_plot)
plt.xlabel("Day of the Week")
plt.ylabel("Chg%")
plt.title("Scatter Plot of Chg% vs Day of the Week")
plt.show()

