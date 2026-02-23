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


def get_team_stats(team_id, league_id, season):
    url = f"https://{API_HOST}/teams/statistics?team={team_id}&league={league_id}&season={season}"
    response = requests.get(url, headers=headers)
    data = response.json()

    goals_for = data["response"]["goals"]["for"]["total"]["total"]
    goals_against = data["response"]["goals"]["against"]["total"]["total"]
    matches_played = data["response"]["fixtures"]["played"]["total"]

    if matches_played == 0:
        return 1.2, 1.2

    avg_for = goals_for / matches_played
    avg_against = goals_against / matches_played

    return avg_for, avg_against


@app.get("/")
def root():
    return {"message": "Sports Predictor AI Running with Real Stats"}


@app.get("/predict/{fixture_id}")
def predict_fixture(fixture_id: int):

    # 1️⃣ Obtener info del partido
    fixture_url = f"https://{API_HOST}/fixtures?id={fixture_id}"
    fixture_response = requests.get(fixture_url, headers=headers)
    fixture_data = fixture_response.json()["response"][0]

    home_team = fixture_data["teams"]["home"]["name"]
    away_team = fixture_data["teams"]["away"]["name"]

    home_id = fixture_data["teams"]["home"]["id"]
    away_id = fixture_data["teams"]["away"]["id"]

    league_id = fixture_data["league"]["id"]
    season = fixture_data["league"]["season"]

    # 2️⃣ Obtener estadísticas reales
    home_avg_for, home_avg_against = get_team_stats(home_id, league_id, season)
    away_avg_for, away_avg_against = get_team_stats(away_id, league_id, season)

    # 3️⃣ Calcular fuerza ofensiva ajustada
    lambda_home = (home_avg_for + away_avg_against) / 2
    lambda_away = (away_avg_for + home_avg_against) / 2

    # 4️⃣ Modelo Poisson
    home_probs = [poisson.pmf(i, lambda_home) for i in range(6)]
    away_probs = [poisson.pmf(i, lambda_away) for i in range(6)]

    home_win = sum(home_probs[i] * sum(away_probs[:i]) for i in range(6))
    draw = sum(home_probs[i] * away_probs[i] for i in range(6))
    away_win = 1 - home_win - draw

    return {
        "home_team": home_team,
        "away_team": away_team,
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
