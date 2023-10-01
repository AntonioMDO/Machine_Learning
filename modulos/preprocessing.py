import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.utils import shuffle
from main import df

#Arreglar el nombre de las columnas
df.columns = df.columns.str.lower()
df.columns

#Reemplazar los valores ausentes
df['tenure'] = df['tenure'].fillna(0)

#Codificación de etiquetas
encoder = OrdinalEncoder()

df_ord = pd.DataFrame(encoder.fit_transform(df), columns=df.columns)

#Separar los datos
features = df_ord.drop(['exited'], axis = 1)
target = df_ord['exited']

#Datos de entrenamiento y validación
x_train, x_valid, y_train, y_valid = train_test_split(features, target, random_state=42, test_size = 0.25)

#Datos de testeo
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state = 42, test_size = 0.25)

#Separar las características numéricas para aplicar el escalado
numeric = ['rownumber', 'customerid', 'creditscore',
           'age', 'tenure', 'balance', 'numofproducts', 'hascrcard',
           'isactivemember', 'estimatedsalary']

#Modelo a utilizar para el escalado
scaler = StandardScaler()
scaler.fit(x_train[numeric])

#Aplicar el solo para los datos de entrenamiento
x_train[numeric] = scaler.transform(x_train[numeric])
x_valid[numeric] = scaler.transform(x_valid[numeric])
x_test[numeric] = scaler.transform(x_test[numeric])

def upsample(x, y, repeat):
    '''
    Aplicar de forma automática el sobremuestreo de las clases
    '''
    #Dividir el conjunto de datos en notas negativas y positivas
    x_zeros = x[y == 0]
    x_ones = x[y == 1]
    y_zeros = y[y == 0]
    y_ones = y[y == 1]
    
    #Concatenar y duplicar los datos
    x_upsampled = pd.concat([x_zeros]+ [x_ones] * repeat)
    y_upsampled = pd.concat([y_zeros] + [y_ones] * repeat)
    
    #Mezclar los datos
    x_upsampled, y_upsampled = shuffle(x_upsampled, y_upsampled, random_state = 42)
    
    return x_upsampled, y_upsampled

#Aplicar la funcion de upsample
x_upsampled, y_upsampled = upsample(x_train, y_train, 5)
