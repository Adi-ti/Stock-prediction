import pandas as pd
import math
from sklearn import preprocessing,cross_validation
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
df = pd.read_csv("SBI.csv",
                 header=0, 
                  index_col='Date',
                  parse_dates=True)

df = df[["Open", "Close", "High", "Low", "Volume"]]
df["PCT_Change"] =(df["Close"] - df["Open"])/ df["Open"] * 100.0
df["HL_PCT"] = (df["High"] - df["Close"]) / df["Close"] * 100.0
forecast_col = "Close"
df.fillna(-99999, inplace = True)
forecast_out = int(math.ceil(0.1 *len(df)))
df["label"] = df[forecast_col].shift(-forecast_out)
X = np.array(df.drop(["label"], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace = True)
y = np.array(df["label"])
y = np.array(df["label"])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)
clf = LinearRegression(n_jobs = -1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)
print(forecast_set,accuracy,forecast_out)
df["Forecast"] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
for i  in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] +[i]
print(df.tail()) 
df["Close"].plot()
df["Forecast"].plot()
plt.legend(loc = 4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()