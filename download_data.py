# -*- coding: utf-8 -*-
"""
download_data.py
=================
Baixa os dados históricos de preços ajustados (Adjusted Close) do universo
de ativos do projeto, via Yahoo Finance, e salva em historical_prices.csv.

Uso:
    python download_data.py

Observação: este script requer conexão à internet e a biblioteca yfinance
instalada (pip install yfinance). O arquivo historical_prices.csv já
incluído no projeto foi gerado por este mesmo script e cobre o período de
2019-12-02 até 2026-06-26 — rode novamente apenas se quiser atualizar os
dados para uma data mais recente.
"""

import pandas as pd
import yfinance as yf

TICKERS = ["URA", "NLR", "CCJ", "URNM", "SPY"]
START_DATE = "2019-12-01"


def main():
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    print(f"Baixando dados de {START_DATE} ate {end_date} para {TICKERS}...")

    data = yf.download(TICKERS, start=START_DATE, end=end_date, auto_adjust=True)

    # yfinance recente retorna MultiIndex de colunas (Price, Ticker)
    if "Close" in data.columns.get_level_values(0):
        prices = data["Close"]
    elif "Adj Close" in data.columns.get_level_values(0):
        prices = data["Adj Close"]
    else:
        raise ValueError("Coluna de preco de fechamento nao encontrada no retorno do yfinance.")

    prices = prices[TICKERS]
    prices = prices.ffill().dropna()

    prices.to_csv("historical_prices.csv", encoding="utf-8-sig")
    print(f"Dados salvos em historical_prices.csv ({len(prices)} pregoes, "
          f"{prices.index[0].date()} a {prices.index[-1].date()}).")


if __name__ == "__main__":
    main()
