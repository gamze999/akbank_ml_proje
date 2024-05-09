#%%
# Kütüphaneleri import etmek:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
#%%
# Veri çerçevesinin datasını tanımlamak:
data = pd.read_csv("winequality.csv")

# Veri çerçevesini oluşturmak:
df = pd.DataFrame(data)

df

df.info()

df.describe()

df.head()
#%%
# Korelasyon matrisi:
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True)
plt.title('Verilerin Korelasyon Matrisi')
plt.show()

df.corr(numeric_only=True)["quality"].sort_values(ascending=False)

# 0'a yakın değerleri çıkarmak:
threshold = 0.1
korelasyon = abs(corr_matrix["quality"])
filtered_data = korelasyon[korelasyon>threshold]
filtered_data

# Kullanmayacağımız kısım:
kullanma = korelasyon[korelasyon<0.1]
kullanma

#%%
# Veri tipini değiştirme:
kullanma_frame = kullanma.to_frame()
kullanma_frame
row_names = kullanma_frame.index
row_names_list = list(row_names)
row_names_list.append('quality')
print(row_names_list)

# x ve y değerleri atamak:
X = data.loc[:, ['residual sugar', 'free sulfur dioxide', 'pH', 'quality']]
y = data.loc[:, ["quality"]]
encoder = LabelEncoder()
for i in X.columns:
    X[i] = encoder.fit_transform(X[i])

y = encoder.fit_transform(y.values.ravel())
#%%
# Karar ağacı ve hiperparametre optimizasyonu:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

logistic_classifier_model = LogisticRegression()
ridge_classifier_model = RidgeClassifier()
decision_tree_model = DecisionTreeClassifier()
naive_bayes_model = GaussianNB()
neural_network_model = MLPClassifier()

logistic_pred = logistic_classifier_model.predict(X_test)
ridge_pred = ridge_classifier_model.predict(X_test)
tree_pred = decision_tree_model.predict(X_test)
naive_bayes_pred = naive_bayes_model.predict(X_test)
neural_network_pred = neural_network_model.predict(X_test)

logistic_report = classification_report(y_test, logistic_pred)
print(logistic_report)
ridge_report = classification_report(y_test, ridge_pred)
print(ridge_report)
tree_report = classification_report(y_test, tree_pred)
print(tree_report)
naive_bayes_report = classification_report(y_test, naive_bayes_pred)
print(naive_bayes_report)
neural_network_report = classification_report(y_test, neural_network_pred)
print(neural_network_report)

steps = [('scaler', StandardScaler()),
('dec_tree', DecisionTreeClassifier())]

pipeline = Pipeline(steps)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
params = {"dec_tree__criterion":['gini', 'entropy'],
         "dec_tree__max_depth":np.arange(3, 15)
         }

from sklearn.model_selection import GridSearchCV
for cv in range(3,10):
    cv_grid = GridSearchCV(pipeline, param_grid=params,cv=cv)
    cv_grid.fit(X_train, y_train)
    print("%d fold score: %3.2f" %(cv,cv_grid.score(X_test, y_test)))
    print("Best parameters: ", cv_grid.best_params_)

# Kesinliğe karar verilmesi (F1 score):
karar_agaci = DecisionTreeClassifier(criterion='gini', max_depth=3)
karar_agaci.fit(X_train, y_train)
y_pred = karar_agaci.predict(X_test)
print(classification_report(y_test, y_pred))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("X_train_scaled shape:",X_train.shape)
print("X_test_scaled shape:",X_test.shape)
print("y_train shape:",y_train.shape)
print("y_test shape:",y_test.shape)

rfc = RandomForestClassifier(random_state=123)

param_grid = {
    'n_estimators': range(100, 1001, 100),
    'max_depth': [None, 10, 20, 30],
}

rfc_g = GridSearchCV(estimator = rfc, param_grid = param_grid,
                           scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

rfc_g.fit(X_train_scaled, y_train)

best_params = rfc_g.best_params_
print("Best parameters:", best_params)