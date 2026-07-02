import pandas as pd
import numpy as np
import joblib

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import VotingClassifier, BaggingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# 1. PRZYGOTOWANIE ŚRODOWISKA I DANYCH

print("Generowanie dużego zbioru danych")
# Generowanie 10 000 próbek, 20 cech, 3 klasy
X_array, y_array = make_classification(
    n_samples=10000,      # Liczba wierszy (kilka tysięcy)
    n_features=20,        # Liczba kolumn (cech)
    n_informative=15,     # Liczba cech niosących faktyczną informację
    n_classes=3,          # Klasyfikacja wieloklasowa (jak w Wine)
    random_state=42
)

# Konwersja do Pandas DataFrame, aby reszta kodu działała bez zmian
feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]
X = pd.DataFrame(X_array, columns=feature_names)
y = pd.Series(y_array, name="target")

print(f"Rozmiar zbioru X: {X.shape}") # Zwróci (10000, 20)

# Podział na zbiór treningowy i testowy z zachowaniem proporcji klas (stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 2. PREPROCESING (Potok transformacji danych)

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), list(range(X.shape[1])))
])


# 3. KROSWALIDACJA I OPTYMALIZACJA PARAMETRÓW

models = {
    "kNN": Pipeline([("prep", preprocessor), ("model", KNeighborsClassifier())]),
    "DT": Pipeline([("prep", preprocessor), ("model", DecisionTreeClassifier(random_state=42))]),
    "NN": Pipeline([("prep", preprocessor), ("model", MLPClassifier(max_iter=500, random_state=42))]),
    "LR": Pipeline([("prep", preprocessor), ("model", LogisticRegression(max_iter=2000))])
}

param_grids = {
    "kNN": {
        "model__n_neighbors": [3, 5, 7, 11],
        "model__weights": ["uniform", "distance"]
    },
    "DT": {
        "model__max_depth": [3, 5, 10, None],
        "model__min_samples_leaf": [1, 3, 5]
    },
    "NN": {
        "model__hidden_layer_sizes": [(32,), (64,), (64, 32)],
        "model__alpha": [1e-4, 1e-3]
    },
    "LR": {
        "model__C": [0.1, 1, 10]
    }
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_models = {}
results = []

def eval_model(name, model):
    pred = model.predict(X_test)
    return {
        "model": name,
        "acc": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred, average="macro")
    }

print("Rozpoczynanie GridSearch")
for name in models:
    gs = GridSearchCV(
        models[name],
        param_grids[name],
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )
    gs.fit(X_train, y_train)

    # Zapisujemy najlepszy estymator
    best_models[name] = gs.best_estimator_

    res = eval_model(name, gs.best_estimator_)
    res["best_params"] = gs.best_params_
    results.append(res)
    print(f" Zoptymalizowano model: {name}")

results_df = pd.DataFrame(results)
print("\nWYNIKI OPTYMALIZACJI MODELI BAZOWYCH")
display(results_df[["model", "acc", "f1", "best_params"]])


# 4. KOMITETY

print("Trenowanie komitetów")

voting_estimators = [
    ("knn", best_models["kNN"]),
    ("dt", best_models["DT"]),
    ("nn", best_models["NN"])
]

# a) Voting
voting_hard = VotingClassifier(estimators=voting_estimators, voting="hard")
voting_soft = VotingClassifier(estimators=voting_estimators, voting="soft")

voting_hard.fit(X_train, y_train)
voting_soft.fit(X_train, y_train)

# b) Bagging
bag_default_base = Pipeline([
    ("prep", preprocessor),
    ("model", DecisionTreeClassifier(random_state=42))
])

bag_default = BaggingClassifier(
    estimator=bag_default_base,
    n_estimators=200,
    random_state=42
)

bag_best = BaggingClassifier(
    estimator=best_models["DT"],
    n_estimators=200,
    random_state=42
)

bag_default.fit(X_train, y_train)
bag_best.fit(X_train, y_train)

# c) Stacking
stack_lr = StackingClassifier(
    estimators=voting_estimators,
    final_estimator=LogisticRegression(max_iter=2000),
    cv=cv
)

stack_rf = StackingClassifier(
    estimators=voting_estimators,
    final_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    cv=cv
)

stack_lr.fit(X_train, y_train)
stack_rf.fit(X_train, y_train)

print(" Wszystkie komitety zostały wyuczone.\n")


# 5. PORÓWNANIE I OCENA

all_committees = {
    "Voting hard": voting_hard,
    "Voting soft": voting_soft,
    "Bagging default": bag_default,
    "Bagging best": bag_best,
    "Stacking (LR)": stack_lr,
    "Stacking (RF)": stack_rf
}

committee_results = []
print("WYNIKI KOMITETÓW")
for name, model in all_committees.items():
    metrics = eval_model(name, model)
    committee_results.append(metrics)
    print(f"{name:<18} -> Accuracy: {metrics['acc']:.4f} | F1-Macro: {metrics['f1']:.4f}")


# 6. ZAPIS NAJLEPSZEGO MODELU

best_committee_name = max(committee_results, key=lambda x: x["f1"])["model"]
best_model_object = all_committees[best_committee_name]

print(f"\n Najwyższy wynik uzyskał komitet: {best_committee_name}")

joblib.dump(best_model_object, "model.pkl")
print(" Model został pomyślnie przygotowany i zapisany jako 'model.pkl'")


# 7. WIZUALIZACJA WYNIKÓW (Wykresy)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Ustawienie stylu wykresów
sns.set_theme(style="whitegrid")

# 1. Wykres modeli bazowych
plt.figure(figsize=(10, 5))
df_base = pd.DataFrame(results)
df_base_melted = df_base.melt(id_vars="model", value_vars=["acc", "f1"], var_name="Metryka", value_name="Wynik")
sns.barplot(data=df_base_melted, x="model", y="Wynik", hue="Metryka", palette="Blues")
plt.title("Wydajność zoptymalizowanych modeli bazowych", fontsize=14)
plt.ylim(0.0, 1.1)
plt.legend(title="Metryka", labels=["Accuracy", "F1-Macro"])
plt.tight_layout()
plt.show()

# 2. Wykres komitetów
plt.figure(figsize=(12, 6))
df_comm = pd.DataFrame(committee_results)
df_comm_melted = df_comm.melt(id_vars="model", value_vars=["acc", "f1"], var_name="Metryka", value_name="Wynik")
sns.barplot(data=df_comm_melted, x="model", y="Wynik", hue="Metryka", palette="viridis")
plt.title("Porównanie wydajności komitetów modeli (Ensembles)", fontsize=14)
plt.ylim(0.0, 1.1)
plt.xticks(rotation=45)
plt.legend(title="Metryka", labels=["Accuracy", "F1-Macro"])
plt.tight_layout()
plt.show()

# 3. Macierz pomyłek dla najlepszego modelu
plt.figure(figsize=(6, 5))
best_pred = best_model_object.predict(X_test)
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title(f"Macierz pomyłek: {best_committee_name}", fontsize=14)
plt.xlabel("Przewidziana klasa")
plt.ylabel("Rzeczywista klasa")
plt.tight_layout()
plt.show()