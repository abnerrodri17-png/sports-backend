from fastapi import FastAPI
from scipy.stats import poisson

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Sports Predictor API Running"}

@app.get("/predict/{home}/{away}/{home_avg}/{away_avg}")
def predict(home: str, away: str, home_avg: float, away_avg: float):

    lambda_home = home_avg
    lambda_away = away_avg

    home_probs = [poisson.pmf(i, lambda_home) for i in range(6)]
    away_probs = [poisson.pmf(i, lambda_away) for i in range(6)]

    home_win = sum(home_probs[i] * sum(away_probs[:i]) for i in range(6))
    draw = sum(home_probs[i] * away_probs[i] for i in range(6))
    away_win = 1 - home_win - draw

    return {
        "home_team": home,
        "away_team": away,
        "home_win_%": round(home_win * 100, 2),
        "draw_%": round(draw * 100, 2),
        "away_win_%": round(away_win * 100, 2),
        "expected_goals_home": round(lambda_home, 2),
        "expected_goals_away": round(lambda_away, 2),
        "expected_shots": round((lambda_home + lambda_away) * 1.4, 2),
        "expected_yellow_cards": round((lambda_home + lambda_away) * 1.2, 2)
    }