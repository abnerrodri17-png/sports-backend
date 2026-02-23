from fastapi import FastAPI
from scipy.stats import poisson
import requests
import os

app = FastAPI()

API_KEY = os.getenv("huRfII6y1i9bWOpby5y9K8CfwECyVQiJoYdCSG2l")  # Tu key de SportDB.dev
BASE_URL = "https://sportdb.dev/api"

headers = {
    "x-api-key": API_KEY
}


def get_team_avg_goals(team_id, league_slug, season):
    """Obtiene media de goles de un equipo"""
    url = f"{BASE_URL}/football/{league_slug}/{season}/teams/{team_id}/stats"
    res = requests.get(url, headers=headers).json()

    # Ajusta según estructura de la API
    goals_for = res.get("goals_for", 1.2)
    goals_against = res.get("goals_against", 1.2)
    matches_played = res.get("matches_played", 1)

    avg_for = goals_for / matches_played
    avg_against = goals_against / matches_played

    return avg_for, avg_against


@app.get("/")
def root():
    return {"message": "Sports Predictor API Running with SportDB.dev"}


@app.get("/predict/{league_slug}/{season}/{home_id}/{away_id}")
def predict(league_slug: str, season: str, home_id: int, away_id: int):
    home_avg_for, home_avg_against = get_team_avg_goals(home_id, league_slug, season)
    away_avg_for, away_avg_against = get_team_avg_goals(away_id, league_slug, season)

    lambda_home = (home_avg_for + away_avg_against) / 2
    lambda_away = (away_avg_for + home_avg_against) / 2

    home_probs = [poisson.pmf(i, lambda_home) for i in range(6)]
    away_probs = [poisson.pmf(i, lambda_away) for i in range(6)]

    home_win = sum(home_probs[i] * sum(away_probs[:i]) for i in range(6))
    draw = sum(home_probs[i] * away_probs[i] for i in range(6))
    away_win = 1 - home_win - draw

    return {
        "home_avg_goals": round(home_avg_for, 2),
        "away_avg_goals": round(away_avg_for, 2),
        "home_win_%": round(home_win * 100, 2),
        "draw_%": round(draw * 100, 2),
        "away_win_%": round(away_win * 100, 2),
        "expected_goals_home": round(lambda_home, 2),
        "expected_goals_away": round(lambda_away, 2),
        "expected_shots": round((lambda_home + lambda_away) * 1.5, 2),
        "expected_yellow_cards": round((lambda_home + lambda_away) * 1.3, 2)
    }
