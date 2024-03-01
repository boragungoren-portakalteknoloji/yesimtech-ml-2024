# ODTÜ BA4318 ders örneğinden adapte edilmiştir -- https://github.com/boragungoren-portakalteknoloji/METU-BA4318-Fall2019/tree/master

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import Holt 
# from sklearn.metrics import mean_squared_error

def estimate_holt(df, seriesname, alpha=0.2, slope=0.1, trend="add"):
    numbers = np.asarray(df[seriesname], dtype='float')
    model = Holt(numbers)
#    fit = model.fit(alpha, slope, trend)
    fit = model.fit()
    estimate = fit.forecast(1)[-1]
    return estimate

def decomp(frame,name,f,mod='Additive'):
    #frame['Date'] = pd.to_datetime(frame['Date'])
    series = frame[name]
    array = np.asarray(series, dtype=float)
    result = sm.tsa.seasonal_decompose(array,period=f,model=mod,two_sided=False) # Eski sürüm freq=f şeklinde
    # Additive model means y(t) = Level + Trend + Seasonality + Noise
    result.plot()
    plt.show() # Uncomment to reshow plot, saved as Figure 1.
    return result

#Series Descriptions	
#TP.KTF10	Personal (TRY)(Flow Data, %)-Level
#TP.KTF101	Personal (TRY)(Including Real Person Overdraft Account)(Flow Data, %)-Level
#TP.KTF11	Vehicle (TRY)(Flow Data, %)-Level
#TP.KTF12	Housing (TRY)(Flow Data, %)-Level
#TP.KTF17	Commercial (TRY)(Flow Data, %)-Level
#TP.KTF17.EUR	Commercial Loans (EUR)(Flow Data, %)-Level
#TP.KTF17.USD	Commercial Loans (USD)(Flow Data, %)-Level
#TP.KTF18	Commercial Loans (TRY)(Excluding Corporate Overdraft Account and Corporate Credit Cards)(Flow Data, %)-Level
#TP.KTFTUK	Consumer Loan (TRY)(Personal+Vehicle+Housing)(Flow Data, %)-Level
#TP.KTFTUK01	Consumer Loan (TRY)(Personal+Vehicle+Housing)(Including Real Person Overdraft Account)(Flow Data, %)-Level

# The following lines are to suppress warning messages.
import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("faiz-orani.txt", sep='\t')
#  , yerine . 
df = df.stack().str.replace(',','.').unstack()
df.set_index('Date')
# df.info() # Tüm sütunlarda eşit sayıda veri olmayabilir. 
print("Veri eksiği olan satırlar nedeni ile eksik veri yüzdesi:")
print( df.isna().mean().round(4)*100)  # Eksik veri yüzde olarak
seriesname = 'TP KTF18' # KMH ve Kredi Kartı hariç ticari krediler. 

print("Metod 1 - dropna()")
df2 = df.dropna() # Eksik veri olan tüm satırlar devre dışı kaldı.
df2.info()
# print("Percentage of missing values")
# print( df2.isna().mean().round(4)*100)  # get percentage of missing values
result = decomp(df2,seriesname,f=52) # 52 weeks per year
# Tahmin oluştur
faiz = round( estimate_holt(df2, seriesname, alpha=0.2, slope=0.1, trend="add"), 2)
print("Önümüzdeki hafta için faiz oranı tahmini:", faiz)

print("Metod 2 - fillna")
# Load data
df = pd.read_csv("faiz-orani.txt", sep='\t')
#  , yerine .
df = df.stack().str.replace(',','.').unstack()
df.set_index('Date')
# forward fill, eksik değerin daha önceki değerin ileri yayınımı ile doldurulması (son gördüğü değer geçerli)
# df3 = df.fillna(method ='ffill')
# backward fill, eksik değerin sonraki değerin geri yayınımı ile doldurulması (yerine geçen değer geçerli)
df3 = df.fillna(method ='bfill') 
# alternative, use number # df.fillna(0, inplace = True)
# df3.info()
# print("Percentage of missing values")
# print( df3.isna().mean().round(4)*100)  # get percentage of missing values
result = decomp(df3,seriesname,f=52) # 52 weeks per yearEstimate 'TP KTF18' for next 1 period

faizler = round( estimate_holt(df3, seriesname, alpha=0.2, slope=0.1, trend="add"), 2)
print("Önümüzdeki hafta için faiz oranı tahmini:", faizler)





