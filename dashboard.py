import streamlit as st  # type: ignore
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

# Connexion à l'API
API_URL = "https://credit-prediction-api.onrender.com"

def get_prediction(client_id):
    response = requests.get(f"{API_URL}/prediction?client_id={client_id}")
    return response.json()

def get_client_info(client_id):
    response = requests.get(f"{API_URL}/client_info/{client_id}")
    return response.json()

# Charger les données de test
test_data = pd.read_csv('test_data2.csv')

# Interface Streamlit
st.title("Dashboard de Scoring Crédit")

client_id = st.number_input("ID du client", min_value=1)

if st.button("Obtenir les informations"):
    client_info = get_client_info(client_id)
    prediction = get_prediction(client_id)
    
    # Affichage du score
    score = prediction["prediction"][0][1] * 100  # Score en pourcentage
    st.metric(label="Score de crédit", value=score)
    
    # Informations client
    with st.expander("Informations du client"):
        st.subheader("Détails du client")
        st.write(client_info)
    
    # Comparaison avec d'autres clients
    st.subheader("Comparaison avec d'autres clients")
    
    # Graphique interactif : Scatter plot
    feature_x = st.selectbox("Sélectionner la caractéristique X", list(client_info.keys()), key="scatter_x")
    feature_y = st.selectbox("Sélectionner la caractéristique Y", list(client_info.keys()), key="scatter_y")
    
    fig_scatter = px.scatter(test_data, x=feature_x, y=feature_y, title="Scatter Plot")
    client_data = pd.DataFrame([client_info])
    fig_scatter.add_trace(px.scatter(client_data, x=feature_x, y=feature_y).data[0])
    
    st.plotly_chart(fig_scatter)
    
    # Gauge Chart améliorée
    st.subheader("Score du Client (Gauge)")
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Score de Crédit"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "green"}
            ],
            'bar': {'color': "black"}  # Aiguille
        }
    ))
    
    st.plotly_chart(fig_gauge)
    
    # Interprétation SHAP (à intégrer)
    st.subheader("Interprétation du score")
    
    # Modification des informations client
    st.subheader("Modifier les informations")
    new_info = {key: st.text_input(key, value) for key, value in client_info.items()}
    
    if st.button("Mettre à jour et recalculer"):
        pass
