import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import shap
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Charger le modèle
model_path = os.path.join(os.path.dirname(__file__), 'model', 'lgbm_modelee.pkl')
model = joblib.load(model_path)

# Charger les données
data_path = os.path.join(os.path.dirname(__file__), 'model', 'test_data.csv')
df = pd.read_csv(data_path)
df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)

# Calculer l'âge
df['AGE'] = (df['DAYS_BIRTH'] / -365).apply(lambda x: int(x))
data_path = os.path.join(os.path.dirname(__file__), 'model', 'test_data_73.csv')
df_73 = pd.read_csv(data_path)
df_73['SK_ID_CURR'] = df_73['SK_ID_CURR'].astype(int)

# Vérifier et formater les données du client
def verifier_donnees_client(df, ID, model):
    if df[df['SK_ID_CURR'] == ID].empty:
        return None, "Client non répertorié"

    X = df[df['SK_ID_CURR'] == ID].drop(['SK_ID_CURR'], axis=1)
    expected_columns = model.feature_name_

    missing_cols = set(expected_columns) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
    
    X = X[expected_columns]
    if X.shape[1] != model.n_features_in_:
        return None, f"Nombre de caractéristiques incorrect: attendu {model.n_features_in_}, reçu {X.shape[1]}"
    
    return X, None


# Effectuer la prédiction
def effectuer_prediction(model, X, seuil=0.625):
    try:
        probability_default_payment = model.predict_proba(X)[:, 1][0]
        prediction = "Prêt NON Accordé, risque de défaut" if probability_default_payment >= seuil else "Prêt Accordé"
        return probability_default_payment, prediction
    except Exception as e:
        return None, str(e)

# Fonction pour créer l'explainer SHAP et calculer les valeurs SHAP
def compute_shap_values(model, data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(data)
    return shap_values

# Fonction pour obtenir les caractéristiques les plus importantes selon SHAP
def get_top_features(shap_values, data, top_n=20):
    mean_shap_values = np.abs(shap_values.values).mean(axis=0)
    sorted_idx = np.argsort(mean_shap_values)[::-1]
    top_n = min(top_n, len(sorted_idx))
    top_features = sorted_idx[:top_n]
    features = data.columns[top_features]
    importances = mean_shap_values[top_features]
    return features, importances

# Fonction pour afficher les caractéristiques les plus importantes avec SHAP
def plot_feature_importance(features, importances):
    plt.figure(figsize=(10, 8))
    plt.barh(features, importances, color='skyblue')
    plt.xlabel('Valeur d\'importance moyenne (valeurs SHAP)', fontsize=14)
    plt.title('Top Importances des Caractéristiques (Moyenne SHAP)', fontsize=16)
    plt.gca().invert_yaxis()
    img_path = "static/feature_importance.png"
    plt.savefig(img_path)
    plt.close()
    return img_path

# Fonction pour afficher un résumé des valeurs SHAP pour toutes les données
def plot_summary(shap_values, data, feat_number=20):
    plt.figure()
    shap.summary_plot(shap_values, data, plot_type="bar", max_display=feat_number, color_bar=False)
    img_path = "static/shap_summary.png"
    plt.savefig(img_path)
    plt.close()
    return img_path

# Route pour l'affichage du formulaire et la prédiction
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        client_id = int(request.form['client_id'])
        X, erreur = verifier_donnees_client(df, client_id, model)
        if erreur:
            return render_template("index.html", erreur=erreur)

        # Afficher les informations du client
        client_data = df[df['SK_ID_CURR'] == client_id].iloc[0]

        # Effectuer la prédiction
        probability_default_payment, prediction = effectuer_prediction(model, X)

        # Calculer les valeurs SHAP
        df_73_copy = df_73[df_73['SK_ID_CURR'] == client_id].drop(['SK_ID_CURR'], axis=1)
        shap_values = compute_shap_values(model, df_73_copy)

        features, importances = get_top_features(shap_values, df, top_n=20)
        feature_importance_img = plot_feature_importance(features, importances)
        shap_summary_img = plot_summary(shap_values, df_73_copy)

        return render_template("index.html", 
                               client_data=client_data, 
                               prediction=prediction, 
                               probability=probability_default_payment,
                               feature_importance_img=feature_importance_img,
                               shap_summary_img=shap_summary_img)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
