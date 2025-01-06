import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import shap

# Charger le modèle
model_path = os.path.join(os.path.dirname(__file__), 'model', 'lgbm_model (2).pkl')
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


# Afficher les informations client
def display_client_info(client, df):
    idx_client = df.index[df['SK_ID_CURR'] == client][0]
    st.sidebar.markdown("### Informations du client sélectionné")
    st.sidebar.markdown(f"**ID client :** {client}")
    st.sidebar.markdown(f"**Sexe :** {df.loc[idx_client, 'CODE_GENDER']}")
    st.sidebar.markdown(f"**Âge :** {df.loc[idx_client, 'AGE']}")
    st.sidebar.markdown(f"**Statut familial :** {df.loc[idx_client, 'NAME_FAMILY_STATUS']}")
    st.sidebar.markdown(f"**Enfants :** {df.loc[idx_client, 'CNT_CHILDREN']}")
    st.sidebar.markdown(f"**Statut professionnel :** {df.loc[idx_client, 'NAME_INCOME_TYPE']}")
    st.sidebar.markdown(f"**Niveau d'études :** {df.loc[idx_client, 'NAME_EDUCATION_TYPE']}")
    return idx_client

# Effectuer la prédiction
def predict_client(ID, seuil):
    # Vérifier si l'ID existe dans la base de données
    if df[df['SK_ID_CURR'] == ID].empty:
        st.error("Ce client n'est pas répertorié")
        return

    # Extraire les données du client
    X = df[df['SK_ID_CURR'] == ID].drop(['SK_ID_CURR'], axis=1)

    # Vérifier et réorganiser les colonnes pour correspondre au modèle
    expected_columns = model.feature_name_
    missing_cols = set(expected_columns) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
    X = X[expected_columns]

    if X.shape[1] != model.n_features_in_:
        st.error(f"Nombre de caractéristiques incorrect: attendu {model.n_features_in_}, reçu {X.shape[1]}")
        return

    # Prédiction
    try:
        probability_default_payment = model.predict_proba(X)[:, 1][0]
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        return

    prediction = "Prêt NON Accordé, risque de défaut" if probability_default_payment >= seuil else "Prêt Accordé"
    st.success(f"Probabilité de défaut de paiement: {probability_default_payment:.4f}")
    st.write(f"Prédiction: {prediction}")

    # Afficher la jauge
    afficher_jauge(probability_default_payment, seuil)

# Afficher une jauge de score
def afficher_jauge(score, seuil):
    fig, ax = plt.subplots(figsize=(6, 3))
    color = 'red' if score >= seuil else 'green'
    ax.barh([0], [score], color=color)
    ax.set_xlim(0, 1)
    ax.text(score, 0, f"{score:.2f}", ha='center', va='center', color='white', fontsize=12)
    ax.set_title("Probabilité de défaut de paiement", fontsize=14)
    ax.set_yticks([])
    st.pyplot(fig)

# Fonction pour calculer le nombre de personnes à risque avec un seuil ajusté
def calculer_risque(df, model, target_percentage=0.08):
    X = df.drop(['SK_ID_CURR'], axis=1)
    # Préparer les données (similaire à la fonction 'verifier_donnees_client')
    expected_columns = model.feature_name_
    missing_cols = set(expected_columns) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
    X = X[expected_columns]
    
    # Prédire les probabilités de défaut
    probas = model.predict_proba(X)[:, 1]
    
    # Calculer le seuil pour obtenir target_percentage des clients à risque
    seuil = np.percentile(probas, (1 - target_percentage) * 100)
    
    # Compter le nombre de clients avec une probabilité supérieure au seuil
    clients_risque = np.sum(probas >= seuil)
    return clients_risque, len(df), probas, seuil

# Fonction pour calculer les valeurs SHAP
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

# Fonction pour afficher les caractéristiques les plus importantes
def display_top_features(features, importances):
    st.write("### Top caractéristiques les plus importantes")
    for feature, importance in zip(features, importances):
        st.write(f"- {feature}: {importance:.4f}")

# Fonction pour visualiser les importances des caractéristiques avec SHAP
def plot_feature_importance(features, importances):
    plt.figure(figsize=(10, 8))
    plt.barh(features, importances, color='skyblue')
    plt.xlabel('Valeur d\'importance moyenne (valeurs SHAP)', fontsize=14)
    plt.title('Top Importances des Caractéristiques (Moyenne SHAP)', fontsize=16)
    plt.gca().invert_yaxis()
    st.pyplot(plt)

# Fonction pour afficher un résumé des valeurs SHAP pour toutes les données
def plot_summary(shap_values, data, feat_number=20):
    plt.figure()
    shap.summary_plot(shap_values, data, plot_type="bar", max_display=feat_number, color_bar=False)
    st.pyplot(plt)

# Fonction pour comparer les caractéristiques des clients
def plot_client_comparison(df, feature):
    fig = px.histogram(df, x=feature, title=f'Comparaison des {feature} pour tous les clients')
    st.plotly_chart(fig)

# Fonction principale de l'application Streamlit
def main():
    st.title("Application de Prédiction de Crédit")
    st.markdown("Cette application permet de prédire l'approbation d'un prêt et d'analyser les caractéristiques importantes.")
    
    if 'data' not in st.session_state:
        st.session_state.data = df.copy()

    # Afficher le nombre de personnes à risque
    if st.button("Calculer le nombre de personnes à risque de paiement"):
        clients_risque, total_clients, probas, seuil = calculer_risque(df, model, target_percentage=0.08)
        st.write(f"Nombre total de clients : {total_clients}")
        st.write(f"Nombre de clients avec un risque de défaut supérieur au seuil : {clients_risque}")
        st.write(f"Proportion de clients à risque : {clients_risque / total_clients * 100:.2f}%")
        
        # Optionnel : Afficher un graphique de la distribution des probabilités de défaut
        fig, ax = plt.subplots()
        ax.hist(probas, bins=20, color='skyblue', edgecolor='black')
        ax.axvline(seuil, color='red', linestyle='--', label=f"Seuil de risque ajusté")
        ax.set_title("Distribution des probabilités de défaut")
        ax.set_xlabel("Probabilité de défaut")
        ax.set_ylabel("Nombre de clients")
        ax.legend()
        st.pyplot(fig)

    unique_features = ['CODE_GENDER', 'NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE']
    selected_feature = st.selectbox("Sélectionnez une caractéristique pour la comparaison :", unique_features)

    ID = st.number_input("Entrez l'ID du client :", min_value=100001, max_value=200000, step=1)
    
    if st.button("Prédire"):
        print('stdata', st.session_state.data.head())
        X, erreur = verifier_donnees_client(st.session_state.data, ID, model)
        if erreur:
            st.error(erreur)
        else:
            idx_client = display_client_info(ID, df)
            predict_client(ID, seuil)  # Appel à la fonction existante pour effectuer la prédiction et afficher les résultats

if __name__ == "__main__":
    main()
