import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler

def load_data(path = 'data/insurance_eda.csv'):
    
   df = pd.read_csv(path)   
   scaler = StandardScaler()
   
 
   X_raw = df[['age', 'bmi', 'children', 'smoker']]
   Y = df['charges']
   
   X = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)

   X_train, X_test, Y_train, Y_test = train_test_split(
       X, Y, test_size= 0.2, random_state= 42
   )
   
   return  X_train, X_test, Y_train, Y_test
    
if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = load_data()
    print(f'X_train: {X_train.shape} , Y_train: {Y_train.shape}')
    