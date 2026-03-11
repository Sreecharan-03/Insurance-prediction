import pandas as pd
from sklearn.model_selection import train_test_split
df=pd.read_csv('insurance.csv') 
y=df['Annual_Premium_Thousands']
x=df.drop(['Annual_Premium_Thousands','Customer_ID'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(df.head())
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
# save the processed data and scaler 
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train_scaled,y_train)
y_pred=model.predict(x_test_scaled)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
import pickle
# Save the model and scaler
with open('insurance_model.pkl','wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl','wb') as f:
    pickle.dump(scaler, f)

