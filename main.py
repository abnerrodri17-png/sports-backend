from fastapi import FastAPI
from scipy.stats import poisson
import requests
import os

app = FastAPI()

API_KEY = os.getenv("SPORTS_API_KEY")
API_HOST = "v3.football.api-sports.io"

headers = {
    "x-apisports-key": API_KEY
}

@app.get("/")
def root():
    return {"message": "Sports Predictor API Running with Real Data"}

@app.get("/matches/{league_id}")
def get_matches(league_id: int):
    url = f"https://{API_HOST}/fixtures?league={league_id}&season=2024"

    response = requests.get(url, headers=headers)
    return response.json()


@app.get("/predict/{fixture_id}")
def predict_fixture(fixture_id: int):

    url = f"https://{API_HOST}/fixtures?id={fixture_id}"
    response = requests.get(url, headers=headers)
    data = response.json()

    home_team = data["response"][0]["teams"]["home"]["name"]
    away_team = data["response"][0]["teams"]["away"]["name"]

    # Aquí normalmente sacaríamos estadísticas reales,
    # pero para MVP usamos promedio fijo temporal
    home_avg = 1.6
    away_avg = 1.2

    lambda_home = home_avg
    lambda_away = away_avg

    home_probs = [poisson.pmf(i, lambda_home) for i in range(6)]
    away_probs = [poisson.pmf(i, lambda_away) for i in range(6)]

    home_win = sum(home_probs[i] * sum(away_probs[:i]) for i in range(6))
    draw = sum(home_probs[i] * away_probs[i] for i in range(6))
    away_win = 1 - home_win - draw

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_%": round(home_win * 100, 2),
        "draw_%": round(draw * 100, 2),
        "away_win_%": round(away_win * 100, 2),
        "expected_shots": round((lambda_home + lambda_away) * 1.4, 2),
        "expected_yellow_cards": round((lambda_home + lambda_away) * 1.2, 2)
    }
