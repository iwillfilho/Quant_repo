# -*- coding: utf-8 -*-
"""
quant_models.py
================
Núcleo Quant - Liga de Mercado Financeiro - UFU - 2026.1

Módulo com as implementações estatísticas usadas no projeto de estratégia
quantitativa para o setor de urânio. Tudo aqui é implementado "na mão" com
numpy/scipy (sem usar pacotes prontos como `arch` ou `statsmodels`, que não
estavam disponíveis no ambiente de execução), o que tem a vantagem de deixar
absolutamente transparente o que cada modelo está calculando.

Modelos implementados:
    1. ARIMA(p,d,q) via mínimos quadrados (regressão linear nos termos
       autorregressivos e de médias móveis, com diferenciação para tornar
       a série estacionária).
    2. GARCH(1,1) via máxima verossimilhança (otimização numérica com
       scipy.optimize), para modelar a variância condicional dos resíduos.
    3. Otimização de Markowitz (fronteira eficiente clássica, média-variância,
       SEM Black-Litterman, exatamente como pedido).
    4. Simulação de Monte Carlo via bootstrap em blocos (block bootstrap) dos
       retornos históricos, para projetar a distribuição de capital futuro.
    5. VaR e CVaR (históricos e paramétricos).

Cada função tem docstring explicando a matemática por trás dela.
"""

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm


# ============================================================
# 1. ARIMA(p, d, q) implementado via mínimos quadrados
# ============================================================

def difference(series, d=1):
    """Aplica diferenciação de ordem d para tornar a série estacionária.

    d=1: y'_t = y_t - y_{t-1} (remove tendência linear/random walk)
    """
    out = series.copy()
    for _ in range(d):
        out = out.diff().dropna()
    return out


def fit_arima(series, p=1, d=1, q=1, max_iter=200):
    """
    Ajusta um modelo ARIMA(p,d,q) via mínimos quadrados condicionais.

    Modelo, após diferenciar d vezes a série original (y_t -> w_t):
        w_t = c + phi_1*w_{t-1} + ... + phi_p*w_{t-p}
                + theta_1*e_{t-1} + ... + theta_q*e_{t-q} + e_t

    Como os termos de média móvel (e_{t-1}, ..., e_{t-q}) dependem de
    resíduos não observados, usamos um algoritmo iterativo (Hannan-Rissanen
    simplificado):
        1. Estima-se inicialmente um AR(p) puro por OLS para obter resíduos.
        2. Usa-se esses resíduos como proxy dos termos MA e reestima-se
           AR(p) + MA(q) por OLS, repetindo até convergir.

    Retorna um dicionário com os coeficientes, resíduos finais e a série
    diferenciada (necessária para reconstruir as previsões em nível).
    """
    w = difference(series, d=d).values
    n = len(w)

    # --- Passo 1: AR(p) puro por OLS para obter resíduos iniciais ---
    X = np.ones((n - p, 1))
    for lag in range(1, p + 1):
        X = np.hstack([X, w[p - lag: n - lag].reshape(-1, 1)])
    y = w[p:n]

    beta_ar, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta_ar

    # --- Passo 2: iterar incluindo termos MA(q) com os resíduos estimados ---
    e = np.zeros(n)
    e[p:n] = resid

    for _ in range(max_iter):
        X_full = np.ones((n - p, 1))
        for lag in range(1, p + 1):
            X_full = np.hstack([X_full, w[p - lag: n - lag].reshape(-1, 1)])
        for lag in range(1, q + 1):
            ma_col = np.zeros(n - p)
            valid = np.arange(p, n) - lag
            mask = valid >= 0
            ma_col[mask] = e[valid[mask]]
            X_full = np.hstack([X_full, ma_col.reshape(-1, 1)])

        beta_full, *_ = np.linalg.lstsq(X_full, y, rcond=None)
        new_resid = y - X_full @ beta_full
        new_e = np.zeros(n)
        new_e[p:n] = new_resid

        if np.allclose(new_e, e, atol=1e-8):
            e = new_e
            beta_full_final = beta_full
            break
        e = new_e
        beta_full_final = beta_full

    c = beta_full_final[0]
    phi = beta_full_final[1:1 + p]
    theta = beta_full_final[1 + p:1 + p + q]
    residuals = e[p:n]

    return {
        "c": c, "phi": phi, "theta": theta,
        "p": p, "d": d, "q": q,
        "residuals": residuals,
        "diff_series": w,
        "last_values": series.values[-max(p, d, 1):],
    }


def forecast_arima(model, n_diff_obs, steps=1):
    """
    Gera previsão de 1 passo à frente para a média condicional (em nível
    diferenciado). Para d=1, a previsão em nível é y_hat_t = y_{t-1} + w_hat_t.
    Usado aqui apenas para obter o resíduo do dia (a parte de média
    condicional do retorno), que alimenta o filtro GARCH.
    """
    c, phi, theta = model["c"], model["phi"], model["theta"]
    w = model["diff_series"]
    e = np.concatenate([np.zeros(len(w) - len(model["residuals"])), model["residuals"]])

    p, q = model["p"], model["q"]
    forecasts = []
    w_ext = list(w)
    e_ext = list(e)

    for _ in range(steps):
        ar_part = sum(phi[i] * w_ext[-(i + 1)] for i in range(p)) if p > 0 else 0.0
        ma_part = sum(theta[i] * e_ext[-(i + 1)] for i in range(q)) if q > 0 else 0.0
        w_hat = c + ar_part + ma_part
        forecasts.append(w_hat)
        w_ext.append(w_hat)
        e_ext.append(0.0)  # esperança do erro futuro é 0

    return np.array(forecasts)


# ============================================================
# 2. GARCH(1,1) via máxima verossimilhança
# ============================================================

def garch_neg_log_likelihood(params, resid):
    """
    Log-verossimilhança negativa do modelo GARCH(1,1):
        sigma2_t = omega + alpha * e_{t-1}^2 + beta * sigma2_{t-1}

    Assumindo erros condicionalmente normais:
        e_t | F_{t-1} ~ N(0, sigma2_t)

    log L = -0.5 * sum( log(2*pi) + log(sigma2_t) + e_t^2 / sigma2_t )
    """
    omega, alpha, beta = params
    
    # Penalidade suave para parâmetros fora da região estável para guiar o otimizador
    penalty = 0.0
    if omega <= 0:
        penalty += (1e-10 - omega) * 1e5
        omega = 1e-10
    if alpha < 0:
        penalty += (-alpha) * 1e5
        alpha = 0.0
    if beta < 0:
        penalty += (-beta) * 1e5
        beta = 0.0
    if (alpha + beta) >= 1.0:
        penalty += (alpha + beta - 0.999) * 1e5
        scale = 0.999 / (alpha + beta)
        alpha *= scale
        beta *= scale

    n = len(resid)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(resid)

    for t in range(1, n):
        sigma2[t] = omega + alpha * resid[t - 1] ** 2 + beta * sigma2[t - 1]

    sigma2 = np.maximum(sigma2, 1e-12)
    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + (resid ** 2) / sigma2)
    return -ll + penalty


def fit_garch(resid):
    """
    Estima omega, alpha, beta do GARCH(1,1) por máxima verossimilhança
    numérica utilizando o otimizador SLSQP com restrições lineares para
    garantir estacionariedade (alpha + beta < 1). Caso falhe, usa o L-BFGS-B
    como fallback.

    Retorna os parâmetros estimados e a série completa de variância
    condicional ajustada (sigma2_t) no período da amostra.
    """
    resid = np.asarray(resid)
    var0 = np.var(resid)

    # Chute inicial: GARCH "típico" de mercado financeiro
    x0 = [0.05 * var0, 0.08, 0.88]
    bounds = [(1e-10, var0), (1e-6, 0.45), (1e-6, 0.999)]
    
    # Restrição linear de estacionariedade: alpha + beta <= 0.999
    constraints = ({"type": "ineq", "fun": lambda x: 0.999 - (x[1] + x[2])},)

    res = optimize.minimize(
        garch_neg_log_likelihood, x0, args=(resid,),
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    
    # Fallback se o SLSQP não convergir
    if not res.success:
        res = optimize.minimize(
            garch_neg_log_likelihood, x0, args=(resid,),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 1000},
        )

    omega, alpha, beta = res.x
    n = len(resid)
    sigma2 = np.zeros(n)
    sigma2[0] = var0
    for t in range(1, n):
        sigma2[t] = omega + alpha * resid[t - 1] ** 2 + beta * sigma2[t - 1]

    return {
        "omega": omega, "alpha": alpha, "beta": beta,
        "sigma2": sigma2,
        "converged": res.success,
        "log_likelihood": -res.fun,
    }


def garch_forecast_next_sigma2(garch_params, last_resid, last_sigma2):
    """Projeta sigma2 do próximo período (1 passo à frente) dado o GARCH(1,1)."""
    omega, alpha, beta = garch_params["omega"], garch_params["alpha"], garch_params["beta"]
    return omega + alpha * last_resid ** 2 + beta * last_sigma2


# ============================================================
# 3. Otimização de Markowitz (média-variância clássica, SEM Black-Litterman)
# ============================================================

def markowitz_max_sharpe(mean_returns, cov_matrix, risk_free_rate=0.0,
                          allow_short=False, max_weight=1.0):
    """
    Resolve o problema clássico de Markowitz, encontrando os pesos do
    portfólio que maximizam o Índice de Sharpe:

        max_w  (w'mu - rf) / sqrt(w'Sigma w)
        s.a.   sum(w) = 1
               0 <= w_i <= max_weight   (sem posição vendida, se allow_short=False)

    mean_returns: vetor de retornos esperados (anualizados) por ativo
    cov_matrix:   matriz de covariância anualizada dos retornos
    max_weight:   peso máximo permitido para cada ativo (limite superior)
    """
    n = len(mean_returns)
    # Garante que o limite superior é matematicamente viável dado o número de ativos
    upper_bound = max(1.0 / n, max_weight)
    
    bounds = (-1.0, 1.0) if allow_short else (0.0, upper_bound)
    bounds = tuple(bounds for _ in range(n))

    def neg_sharpe(w):
        port_return = w @ mean_returns
        port_vol = np.sqrt(w @ cov_matrix @ w)
        if port_vol == 0:
            return 1e10
        return -(port_return - risk_free_rate) / port_vol

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    w0 = np.repeat(1.0 / n, n)

    result = optimize.minimize(
        neg_sharpe, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    weights = result.x
    port_return = weights @ mean_returns
    port_vol = np.sqrt(weights @ cov_matrix @ weights)
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else np.nan

    return {
        "weights": weights,
        "expected_return": port_return,
        "expected_vol": port_vol,
        "sharpe": sharpe,
        "success": result.success,
    }


def markowitz_min_variance(mean_returns, cov_matrix, allow_short=False):
    """Portfólio de mínima variância (vértice esquerdo da fronteira eficiente)."""
    n = len(mean_returns)
    bounds = (-1.0, 1.0) if allow_short else (0.0, 1.0)
    bounds = tuple(bounds for _ in range(n))

    def port_var(w):
        return w @ cov_matrix @ w

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    w0 = np.repeat(1.0 / n, n)

    result = optimize.minimize(
        port_var, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    weights = result.x
    return {
        "weights": weights,
        "expected_return": weights @ mean_returns,
        "expected_vol": np.sqrt(weights @ cov_matrix @ weights),
    }


def efficient_frontier(mean_returns, cov_matrix, n_points=40, allow_short=False):
    """
    Traça a fronteira eficiente de Markowitz, minimizando a variância para
    uma grade de retornos-alvo entre o portfólio de mínima variância e o
    ativo de maior retorno esperado individual.
    """
    n = len(mean_returns)
    bounds = (-1.0, 1.0) if allow_short else (0.0, 1.0)
    bounds = tuple(bounds for _ in range(n))

    min_var = markowitz_min_variance(mean_returns, cov_matrix, allow_short)
    target_returns = np.linspace(min_var["expected_return"], mean_returns.max(), n_points)

    frontier_vols = []
    frontier_rets = []
    for target in target_returns:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, target=target: w @ mean_returns - target},
        )
        w0 = np.repeat(1.0 / n, n)
        result = optimize.minimize(
            lambda w: w @ cov_matrix @ w, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-14},
        )
        if result.success:
            frontier_vols.append(np.sqrt(result.x @ cov_matrix @ result.x))
            frontier_rets.append(target)

    return np.array(frontier_vols), np.array(frontier_rets)


# ============================================================
# 4. Simulação de Monte Carlo (block bootstrap dos retornos)
# ============================================================

def monte_carlo_block_bootstrap(returns_series, n_sims=5000, horizon_days=252,
                                 block_size=20, initial_capital=1.0, seed=42):
    """
    Simula trajetórias futuras de capital via block bootstrap dos retornos
    históricos da estratégia (ou de um ativo/portfólio).

    Por que block bootstrap em vez de reamostragem i.i.d.?
        Retornos financeiros têm autocorrelação na volatilidade (volatility
        clustering - dias de alta volatilidade tendem a vir seguidos de
        outros dias de alta volatilidade). Reamostrar blocos contíguos de
        retornos (em vez de dias isolados) preserva parte dessa estrutura
        de dependência temporal, ao contrário do bootstrap i.i.d. clássico.

    Retorna a matriz de trajetórias simuladas (n_sims x horizon_days).
    """
    rng = np.random.default_rng(seed)
    returns_array = returns_series.dropna().values
    n_obs = len(returns_array)
    n_blocks_needed = int(np.ceil(horizon_days / block_size))

    simulations = np.zeros((n_sims, horizon_days))

    for sim in range(n_sims):
        path_returns = []
        for _ in range(n_blocks_needed):
            start_idx = rng.integers(0, n_obs - block_size)
            block = returns_array[start_idx: start_idx + block_size]
            path_returns.extend(block)
        path_returns = np.array(path_returns[:horizon_days])
        cum_path = initial_capital * np.cumprod(1 + path_returns)
        simulations[sim, :] = cum_path

    return simulations


def monte_carlo_summary(simulations, initial_capital=1.0):
    """Resumo estatístico das simulações de Monte Carlo no horizonte final."""
    final_values = simulations[:, -1]
    final_returns = final_values / initial_capital - 1

    return {
        "mean_final_return": np.mean(final_returns),
        "median_final_return": np.median(final_returns),
        "p5_final_return": np.percentile(final_returns, 5),
        "p95_final_return": np.percentile(final_returns, 95),
        "prob_loss": np.mean(final_returns < 0),
        "var_95_terminal": np.percentile(final_returns, 5),
        "cvar_95_terminal": final_returns[final_returns <= np.percentile(final_returns, 5)].mean(),
        "final_returns_dist": final_returns,
    }


# ============================================================
# 5. VaR e CVaR (histórico e paramétrico/Gaussiano)
# ============================================================

def historical_var_cvar(returns_series, confidence=0.95):
    """VaR e CVaR históricos (não-paramétricos), a partir do percentil empírico."""
    r = returns_series.dropna().values
    if len(r) == 0:
        return np.nan, np.nan
    var = np.percentile(r, (1 - confidence) * 100)
    cvar = r[r <= var].mean()
    return var, cvar


def parametric_var_cvar(returns_series, confidence=0.95):
    """
    VaR e CVaR paramétricos assumindo retornos ~ N(mu, sigma^2).

    VaR_param  = mu + sigma * z_alpha
    CVaR_param = mu - sigma * phi(z_alpha) / alpha
    onde z_alpha é o quantil da normal padrão e phi a densidade normal padrão.
    """
    mu = returns_series.mean()
    sigma = returns_series.std()
    alpha = 1 - confidence
    z = norm.ppf(alpha)

    var = mu + sigma * z
    cvar = mu - sigma * norm.pdf(z) / alpha
    return var, cvar
