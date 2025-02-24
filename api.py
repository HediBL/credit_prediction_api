import flask
from flask import request, jsonify
import joblib
import pandas as pd
import sklearn
import gzip
import os

# Initialisation de Flask
app = flask.Flask(__name__)
app.config["DEBUG"] = True

# Charger le modèle compressé
model_path = 'best_rf.pkl.gz'  # Utilise le chemin relatif du fichier
with gzip.open(model_path, 'rb') as f:
    model = joblib.load(f)
print(model)

# Charger les données de test (exemple)
try:
    test_data = pd.read_csv('test_data2.csv')
    print(test_data.head())  # Affiche les 5 premières lignes du fichier pour vérifier
except Exception as e:
    print(f"Erreur lors du chargement du fichier : {e}")


# Démarrer l'API
@app.route('/')
def home():
    return 'API de prédiction de crédit'

# Endpoint pour vérifier si un client existe
@app.route('/check_client/<int:client_id>', methods=['GET'])
def check_client(client_id):
    # Ajouter un print pour vérifier que la fonction est appelée
    print(f"Vérification du client avec client_id : {client_id}")
    
    if client_id in test_data['SK_ID_CURR'].values:
        print("Client trouvé : True")
        return jsonify(True), 200
    else:
        print("Client trouvé : False")
        return jsonify(False), 404

# Endpoint pour récupérer les informations d'un client
@app.route('/client_info/<int:client_id>', methods=['GET'])
def get_client_info(client_id):
    if client_id in test_data['SK_ID_CURR'].values:
        client_data = test_data[test_data['SK_ID_CURR'] == client_id].to_dict(orient='records')[0]
        return jsonify(client_data), 200
    else:
        return jsonify({"error": "Client not found"}), 404

# Endpoint pour mettre à jour les informations d'un client
@app.route('/client_info/<int:client_id>', methods=['PUT'])
def update_client_info(client_id):
    if client_id in test_data['SK_ID_CURR'].values:
        updated_data = request.get_json()
        # Mettre à jour les données dans test_data
        test_data.loc[test_data['SK_ID_CURR'] == client_id, updated_data.keys()] = updated_data.values()
        return jsonify({"message": "Client info updated"}), 200
    else:
        return jsonify({"error": "Client not found"}), 404

# Endpoint pour soumettre un nouveau client (POST)
@app.route('/client_info', methods=['POST'])
def submit_new_client():
    new_client_data = request.get_json()
    new_client_id = max(test_data['SK_ID_CURR']) + 1  # Crée un nouvel ID unique
    new_client_data['SK_ID_CURR'] = new_client_id
    test_data = test_data.append(new_client_data, ignore_index=True)
    return jsonify({"client_id": new_client_id}), 201

# Endpoint pour la prédiction
@app.route('/prediction', methods=['GET'])
def get_prediction():

        client_id= request.args.get("client_id")
        client_id= int(client_id)

        # Vérifier si le client existe dans les données de test
        if client_id not in test_data['SK_ID_CURR'].values:
            return jsonify({"error": "Client not found"}), 404

        # Extraire les données du client à partir de test_data
        client_data = test_data[test_data['SK_ID_CURR'] == client_id].iloc[0]

        # Normaliser les noms de colonnes dans test_data
        test_data.columns = test_data.columns.str.replace(' ', '_').str.replace(':', '_').str.replace('-', '_')

        # Convertir les données en DataFrame pour la prédiction
        client_df = pd.DataFrame([client_data])
        print(client_df)

        # Assurez-vous d'avoir uniquement les colonnes que le modèle attend
        client_df = client_df.reindex(columns=model.feature_names_in_, fill_value=0)
        
        # Déboguer les colonnes
        print("Colonnes d'entraînement :", model.feature_names_in_)
        print("Colonnes de test :", client_df.columns)

        # Effectuer la prédiction
        prediction = model.predict_proba(client_df)  # model.predict_proba(client_df) pour les probabilités ou model.predict(client_df)

        return jsonify({"prediction": prediction.tolist()}), 200


# Endpoint pour tester l'état de l'API
@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "API en ligne et prête à prédire!"}), 200

# Démarrer l'application Flask
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
