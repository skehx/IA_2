import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 

def load_data(path = 'data/insurance.csv'):
    
   df = pd.read_csv(path)
   encoder = LabelEncoder()
   
   df['smoker'] = encoder.fit_transform(df['smoker'])
   X = df[['age', 'bmi', 'children', 'smoker']]
   Y = df['charges']
   
   X_train, X_test, Y_train, Y_test = train_test_split(
       X, Y, test_size= 0.2, random_state= 42
   )
   
   return  X_train, X_test, Y_train, Y_test
    
if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = load_data()
    print(f'X_train: {X_train.shape} , Y_train: {Y_train.shape}')
    