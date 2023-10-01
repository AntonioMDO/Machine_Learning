import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,roc_auc_score
from preprocessing import x_upsampled, y_upsampled, x_valid, y_valid

#Buscar mejor los mejores hiperparametros para el modelo árbol de decisiones

best_f1_score = 0
best_depth = 0

for depth in range(1, 16):
    dtc = DecisionTreeClassifier(random_state=42, max_depth= depth, class_weight= 'balanced')
    dtc.fit(x_upsampled, y_upsampled)
    predict = dtc.predict(x_valid)
    f1 = f1_score(y_valid, predict)
    roc = roc_auc_score(y_valid, predict)
    if f1 > best_f1_score:
        best_f1_score = f1
        best_depth = depth

#Mejor modelo en árbol de decisiones
best_dtc = DecisionTreeClassifier(random_state=42, max_depth= best_depth, class_weight='balanced')

#Entrenar modelo
best_dtc.fit(x_upsampled, y_upsampled)
probabilities_valid = best_dtc.predict_proba(x_valid) #Predicciones con predict_proba para ajustar el umbral y mejorar las métricas del modelo
probabilities_one_valid = probabilities_valid[:, 1]

best_f1_dtc = 0
best_threshold_dtc = 0

for threshold in np.arange(0, 0.5, 0.02):
    predicted_valid = probabilities_one_valid > threshold
    f1 = f1_score(y_valid, predicted_valid)
    roc_dtc = round(roc_auc_score(y_valid, predicted_valid),3)
    if f1 > best_f1_dtc:
        best_f1_dtc = round(f1,3)
        best_threshold_dtc = threshold

print('Valores de las métricas del mejor modelo en Árbol de decisiones')
print('Árbol de decisiones, valor de F1:',best_f1_dtc)
print('Árbol de decisiones, valor de ROC:',roc_dtc)
print('Árbol de decisiones, umbral final:',best_threshold_dtc)
print()

#Buscar mejor los mejores hiperparametros para el modelo bosque aleatorio

best_f1_score = 0
best_est = 0
best_depth = 0

for est in range(1, 50, 10):
    for depth in range(1, 14):
        rfc= RandomForestClassifier(random_state= 42, n_estimators= est, max_depth= depth, class_weight= 'balanced')
        rfc.fit(x_upsampled, y_upsampled)
        predict = rfc.predict(x_valid)
        f1 = f1_score(y_valid, predict)
        if f1 > best_f1_score:
            best_f1_score = f1
            best_est = est
            best_depth = depth

#Mejor modelo Bosque aleatorio
rfc = RandomForestClassifier(random_state= 42, max_depth= best_depth, n_estimators= best_est, class_weight= 'balanced')

rfc.fit(x_upsampled, y_upsampled)
probabilities_valid = rfc.predict_proba(x_valid) #Predicciones con predict_proba para ajustar el umbral y mejorar las métricas del modelo
probabilities_one_valid = probabilities_valid[:, 1]

best_f1_rfc = 0

#Calcular métricas
for threshold in np.arange(0, 0.5, 0.02):
    predicted_valid = probabilities_one_valid > threshold
    f1 = f1_score(y_valid, predicted_valid)
    roc_rfc = round(roc_auc_score(y_valid, predicted_valid),3)
    if f1 > best_f1_rfc:
        best_f1_rfc = round(f1,3)
        best_threshold_rfc = threshold
        best_roc_rfc = roc_rfc
        
print('Valores de las métricas del mejor modelo en Bosque aleatorio')
print('Bosque aleatorio, valor de F1:',best_f1_rfc)
print('Bosque aleatorio, valor de ROC:',best_roc_rfc)
print('Bosque aleatorio, umbral final:',best_threshold_rfc)
print()

#Modelo de regresión logística
lr= LogisticRegression(random_state=12345, solver='liblinear')

#Entrenar modelo
lr.fit(x_upsampled, y_upsampled)
probabilities_valid = lr.predict_proba(x_valid) #Predicciones con predict_proba para ajustar el umbral y mejorar las métricas del modelo
probabilities_one_valid = probabilities_valid[:, 1]

best_f1_lr = 0

#Calcular métricas
for threshold in np.arange(0, 0.45, 0.02):
    predicted_valid = probabilities_one_valid > threshold
    f1 = f1_score(y_valid, predicted_valid)
    roc_lr = round(roc_auc_score(y_valid, predicted_valid),3)
    if f1 > best_f1_lr:
        best_f1_lr = round(f1,3)
        best_threshold_lr = threshold

print('Valores de las métricas del mejor modelo en Regresión logística')
print('Regresión logística, valor de F1:',best_f1_lr)
print('Regresión logística, valor de ROC:',roc_lr)
print('Regresión logística, umbral final:',best_threshold_lr)
print()

if (best_f1_dtc > best_f1_rfc) & (best_f1_dtc > best_f1_lr):
    print('El mejor modelo es',dtc)
    print('F1 con un valor de:', best_f1_dtc)
    print('Curva ROC con un valor de:', roc_dtc)
elif (best_f1_rfc > best_f1_dtc) & (best_f1_rfc > best_f1_lr):
    print('El mejor modelo es', rfc)
    print('F1 con un valor de:', best_f1_rfc)
    print('Curva ROC con un valor de:', best_roc_rfc)
elif (best_f1_lr > best_f1_dtc) & (best_f1_lr > best_f1_rfc):
    print('El mejor modelo es', lr)
    print('F1 con un valor de:', best_f1_lr)
    print('Curva ROC con un valor de:', roc_lr)
else:
    print('Revisar')