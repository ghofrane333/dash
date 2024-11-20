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
def effectuer_prediction(model, X, seuil=0.625):
    try:
        probability_default_payment = model.predict_proba(X)[:, 1][0]
        prediction = "Prêt NON Accordé, risque de défaut" if probability_default_payment >= seuil else "Prêt Accordé"
        return probability_default_payment, prediction
    except Exception as e:
        return None, str(e)

# Afficher une jauge de score
def afficher_jauge(score, seuil):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    color = 'red' if score >= seuil else 'green'
    ax.barh([0], [score], color=color)
    ax.text(score, 0, f"{score:.2f}", ha='center', va='center', color='white', fontsize=12)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks([])
    ax.set_title("Probabilité de défaut de paiement", fontsize=14)
    st.pyplot(fig)

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

def plot_waterfall(shap_values, sample_index):
    if 0 <= sample_index < len(shap_values):
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[sample_index], show=False)
        plt.title("Graphique en cascade pour le client", fontsize=16)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.error(f"L'indice {sample_index} est hors des limites. Veuillez sélectionner un indice valide.")

# Fonction pour afficher un résumé des valeurs SHAP pour toutes les données
def plot_summary(shap_values, data, feat_number=20):
    plt.figure()
    shap.summary_plot(shap_values, data, plot_type="bar", max_display=feat_number, color_bar=False)
    st.pyplot(plt)

# Fonction pour comparer les caractéristiques des clients
def plot_client_comparison(df, feature):
    fig = px.histogram(df, x=feature, title=f'Comparaison des {feature} pour tous les clients')
    st.plotly_chart(fig)



# Fonction pour afficher les informations du client
def modify_client_info(client_id, data):
    st.write("### Informations du client")
    st.write(data[data['SK_ID_CURR'] == client_id])




# Fonction principale de l'application Streamlit
def main():
    st.title("Application de Prédiction de Crédit")
    st.markdown("Cette application permet de prédire l'approbation d'un prêt et d'analyser les caractéristiques importantes.")
    
    if 'data' not in st.session_state:
        st.session_state.data = df.copy()
    
    unique_features = ['CODE_GENDER', 'NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE']
    selected_feature = st.selectbox("Sélectionnez une caractéristique pour la comparaison :", unique_features)

    ID = st.number_input("Entrez l'ID du client :", min_value=100001)

    if st.button("Prédire"):
        print('stdata', st.session_state.data.head())
        X, erreur = verifier_donnees_client(st.session_state.data, ID, model)
        if erreur:
            st.error(erreur)
        else:
            idx_client = display_client_info(ID, st.session_state.data)
            
            probability_default_payment, prediction = effectuer_prediction(model, X)
            afficher_jauge(probability_default_payment, 0.625)
            
            st.success(prediction)

            df_73_copy =df_73[df_73['SK_ID_CURR'] == ID].drop(['SK_ID_CURR'], axis=1)

            shap_values = compute_shap_values(model, df_73_copy)
           
        
            features, importances = get_top_features(shap_values, st.session_state.data, top_n=20)
            
            display_top_features(features, importances)
            plot_feature_importance(features, importances)

            sample_ind = st.session_state.data.index[st.session_state.data['SK_ID_CURR'] == ID][0]
            plot_waterfall(shap_values, sample_ind)
            plot_summary(shap_values, df_73_copy)


            st.subheader(f"Comparaison des caractéristiques des clients par rapport à {selected_feature}")
            plot_client_comparison(st.session_state.data, selected_feature)

if __name__ == "__main__":
    main()
