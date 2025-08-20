# Prédiction buy/sell sur séries temporelles (LSTM) — Demo éducative

Repo minimal démontrant un pipeline de prédiction directionnelle (J+1) sur données publiques (Yahoo Finance), avec LSTM (TensorFlow), backtest simple avec coûts, et métriques financières.

Fonctionnalités
- Données: téléchargement OHLCV (yfinance).
- Features: retours, RSI.
- Labels: direction de rendement J+1.
- Modèle: LSTM binaire (Keras).
- Évaluation: AUC-PR, Sharpe annualisé, Max Drawdown, equity curve.
- Backtest: positions long/short, coûts par changement de position.

Installation
- Python 3.10+ recommandé
- Crée un venv puis installe les dépendances:
  ```
  python -m venv .venv
  source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
  pip install -r requirements.txt
  ```

Utilisation
- Par défaut, télécharge AAPL (quotidien) 2015–2024, fenêtre LSTM=30 jours:
  ```
  python main.py --ticker AAPL --start 2015-01-01 --end 2024-12-31 --window 30 --cost_bps 10
  ```
- Sorties:
  - Métriques AUC-PR, Sharpe, MDD
  - Graphique equity: outputs/equity_curve.png
  - Logs d’entraînement: console

Notes méthodologiques
- Split temporel (train/val/test) pour éviter la fuite; normalisation fit sur train uniquement.
- Backtest simple, à but pédagogique: ne pas utiliser pour décision d’investissement.
- Améliorations possibles: walk-forward, coûts/slippage plus réalistes, calibration seuils, gestion class imbalance.

Licence
- MIT. Projet éducatif.
