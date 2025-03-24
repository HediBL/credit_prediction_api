import streamlit as st # type: ignore
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
# Connexion à l'API
API_URL = "https://credit-prediction-api.onrender.com"

def get_prediction(client_id):
    # Appel à l'API pour obtenir la prédiction pour un client
    response = requests.get(f"{API_URL}/prediction?client_id={client_id}")
    return response.json()

def get_client_info(client_id):
    # Appel à l'API pour obtenir les informations d'un client
    response = requests.get(f"{API_URL}/client_info/{client_id}")
    return response.json()

# Charger les données de test si nécessaire pour les comparaisons
test_data = pd.read_csv('/home/utilisateur/Documents/About_Me/Academiques/Formation/Formation_Data_Scientist/P8/test_data.csv')

# Interface Streamlit
# Interface Streamlit
st.title("Dashboard de Scoring Crédit")

client_id = st.number_input("ID du client", min_value=1)

if st.button("Obtenir les informations"):
    # Récupérer les informations du client et la prédiction
    client_info = get_client_info(client_id)
    prediction = get_prediction(client_id)
    
    # Affichage du score
    score = prediction["prediction"][0][1] * 100  # Calculer le score
    st.metric(label="Score de crédit", value=score)
    

    # Informations client
    with st.expander("Informations du client"):
        st.subheader("Détails du client")
        st.write(client_info)
    
    # Comparaison avec d'autres clients
    st.subheader("Comparaison avec d'autres clients")
    
    # Graphique interactif 1 : Scatter plot avec client sélectionné
    st.subheader("Scatter Plot avec Client Sélectionné")
    feature_x = st.selectbox("Sélectionner la caractéristique X", list(client_info.keys()), key="scatter_x")
    feature_y = st.selectbox("Sélectionner la caractéristique Y", list(client_info.keys()), key="scatter_y")
    
    # Préparer les données pour le scatter plot
    fig_scatter = px.scatter(test_data, x=feature_x, y=feature_y, title="Scatter Plot")
    
    # Ajouter le client sélectionné en couleur
    client_data = pd.DataFrame([client_info])  # Convertir en DataFrame pour Plotly
    fig_scatter.add_trace(px.scatter(client_data, x=feature_x, y=feature_y).data[0]) # Ajouter le client en couleur
    
    st.plotly_chart(fig_scatter)
    
    # Gauge Chart avec st.progress
    st.subheader("Score du Client (Gauge)")
    
    # Définir les seuils et les couleurs
    if score < 50:
        color = "red"
    elif score < 75:
        color = "yellow"
    else:
        color = "green"
    
    # Afficher la jauge de progression avec la couleur appropriée
    st.markdown(
        f'<div style="background-color:{color}; width:{score}%; text-align:right; padding:5px; color:white;">{score:.2f}%</div>',
        unsafe_allow_html=True,
    )
    
    # Interprétation SHAP (à intégrer)
    st.subheader("Interprétation du score")
    # Intégrer ici l'explication SHAP
    
    # Modification des informations client
    st.subheader("Modifier les informations")
    new_info = {}
    for key, value in client_info.items():
        new_info[key] = st.text_input(key, value)
    
    if st.button("Mettre à jour et recalculer"):
        # Appel à l'API pour mettre à jour les informations et obtenir un nouveau score
        pass
