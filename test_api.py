import pytest
from flask import Flask
import json
from api import app, test_data
import requests

# URL de l'API
BASE_URL = "http://127.0.0.1:8000"  # Changez cela en fonction de votre configuration

# Exemple de requête utilisant requests
def test_api_requests():
    result = requests.get(f"{BASE_URL}/")
    assert result.status_code == 200
    assert result.text == "API de prédiction de crédit"

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

@pytest.mark.parametrize("client_id, expected_status", [
    (1, 200),     # ID d'un client existant
    (9999, 404)  # ID d'un client qui n'existe pas
])
def test_check_client(client, client_id, expected_status):
    result = client.get(f'/check_client/{client_id}')
    assert result.status_code == expected_status

@pytest.mark.parametrize("client_id", [1])  # Changez ici pour ajouter des IDs existants
def test_prediction(client, client_id):
    result = client.get(f'/prediction?client_id={client_id}')
    assert result.status_code == 200
    assert 'prediction' in result.json
