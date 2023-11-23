# Instalando o TensorFlow
pip install tensorflow
 
# Importando bibliotecas
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler



# Carregando o conjunto de dados (usando Iris dataset como exemplo)
iris = load_iris()
X = iris.data

y = iris.target

print(X)

print("\n")

print(y)


# Pré-processamento dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Construindo o modelo
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))


# Compilando o modelo
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Treinando o modelo
model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test))


# Avaliando o modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Acurácia do modelo: {accuracy}")