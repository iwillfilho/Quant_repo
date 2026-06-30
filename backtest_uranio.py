# -*- coding: utf-8 -*-
"""
backtest_uranio.py
===================
Núcleo Quant - Liga de Mercado Financeiro - UFU - 2026.1
Projeto Final - Estratégia Quantitativa Completa: Mercado de Urânio

Fluxo do projeto: Tese -> Modelo -> Backtest -> Resultado

TESE
----
Ativos ligados ao urânio (ETFs e mineradoras) estão subprecificados frente à
pressão estrutural de demanda futura, sustentada por três vetores
geopolíticos independentes:
    1. Segurança energética: a guerra na Ucrânia acelerou a diversificação
       do Ocidente em relação ao urânio enriquecido russo.
    2. Compromisso multilateral (Declaração de Triplicar a Energia Nuclear,
       lançada na COP28/2023) para triplicar a capacidade nuclear até 2050.
    3. Demanda de energia elétrica por data centers de IA, com grandes
       empresas de tecnologia assinando contratos de compra de energia
       nuclear de longo prazo.
Do lado da oferta, o Cazaquistão (maior produtor mundial) anunciou cortes de
produção em 2025 e 2026, intensificando o desequilíbrio estrutural.

ATIVOS
------
- URA   (Global X Uranium ETF)        - ETF amplo de mineradoras de urânio
- NLR   (VanEck Uranium & Nuclear ETF)- ETF de urânio + energia nuclear
- CCJ   (Cameco Corporation)          - maior mineradora de urânio do mundo
- URNM  (Sprott Uranium Miners ETF)   - ETF concentrado em mineradoras puras
- SPY   (SPDR S&P 500 ETF)            - benchmark de mercado

HORIZONTE
---------
Médio prazo. Dados diários, mas sinais calculados sobre médias móveis de
3 meses (63 dias úteis) e 10 meses (210 dias úteis) - portanto a estratégia
gira posições com frequência de semanas a poucos meses, não diariamente.

PERÍODO
-------
2019-12-02 até 2026-06-26 (limitado pela data de início do URNM, o ativo
mais novo do universo, lançado em 03/12/2019). Isso fornece aproximadamente
6,5 anos de dados diários - acima do mínimo de 5 anos exigido.

Divisão temporal (sem look-ahead bias):
    In-sample (treino/calibração): 2019-12-02 a 2022-12-30  (~3 anos)
    Out-of-sample (teste):          2023-01-01 a 2026-06-26  (~3,5 anos)

MODELO
------
1. Sinal de tendência: cruzamento de médias móveis simples, SMA(63) x
   SMA(210) - "SMA 3x10" em meses.
2. Filtro estatístico de risco: para cada ativo, ajusta-se um modelo
   ARIMA(1,0,1) sobre os retornos para extrair a média condicional, e um
   GARCH(1,1) sobre os resíduos do ARIMA para obter a volatilidade
   condicional (sigma_t). O sinal de tendência só é executado quando a
   volatilidade condicional projetada não excede um múltiplo da volatilidade
   condicional média histórica (filtro de regime).
3. Alocação de capital entre os ativos com sinal ativo: otimização de
   Markowitz (média-variância, SEM Black-Litterman), maximizando o Índice
   de Sharpe da carteira, recalibrada trimestralmente com dados estritamente
   anteriores (sem look-ahead).
4. Gestão de risco: VaR e CVaR históricos e paramétricos da carteira, e
   simulação de Monte Carlo (block bootstrap) para projetar a distribuição
   de capital nos próximos 12 meses.

CUSTOS DE TRANSAÇÃO
--------------------
0,10% (10 bps) por operação, aplicado sempre que a posição em um ativo muda
de peso entre um rebalanceamento e outro.

ATENÇÃO A VIESES
-----------------
- Look-ahead bias: todos os sinais usam apenas informação até o fechamento
  do dia anterior (.shift(1)); os parâmetros de Markowitz em cada
  rebalanceamento usam apenas dados anteriores à data do rebalanceamento.
- Overfitting: os parâmetros do modelo (janelas de SMA, ordem do
  ARIMA-GARCH, threshold de volatilidade) foram fixados com base em
  convenção de mercado e na literatura, NÃO otimizados sobre o período de
  teste (out-of-sample).
- Custos: aplicados em toda mudança de posição.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, ".")
from quant_models import (
    fit_arima, fit_garch, garch_forecast_next_sigma2,
    markowitz_max_sharpe, markowitz_min_variance, efficient_frontier,
    monte_carlo_block_bootstrap, monte_carlo_summary,
    historical_var_cvar, parametric_var_cvar,
)

warnings.filterwarnings("ignore")
np.random.seed(42)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "URA": "#7F77DD", "NLR": "#378ADD", "CCJ": "#D85A30",
    "URNM": "#1D9E75", "SPY": "#888780", "Estrategia": "#3C3489",
    "Markowitz": "#993556",
}

ASSETS = ["URA", "NLR", "CCJ", "URNM"]
BENCHMARK = "SPY"
TICKERS = ASSETS + [BENCHMARK]

TRAIN_START = "2019-12-02"
TRAIN_END = "2022-12-30"
TEST_START = "2023-01-01"
TEST_END = "2026-06-26"

SMA_SHORT = 63      # ~3 meses úteis
SMA_LONG = 210       # ~10 meses úteis
VOL_THRESHOLD = 1.5  # filtro: bloqueia se sigma_t > 1.5x a sigma media historica
REBALANCE_FREQ_DAYS = 63  # Markowitz recalibrado a cada ~3 meses (trimestral)
TRANSACTION_COST = 0.0010  # 10 bps por operação
RISK_FREE_RATE = 0.045     # ~Treasury curto prazo medio do periodo (aprox.)
TRADING_DAYS = 252


# ======================================================================
# ETAPA 1 - CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ======================================================================

def load_data(path="historical_prices.csv"):
    """Carrega o histórico de preços ajustados (dados reais, baixados
    previamente do Yahoo Finance) e organiza no período de análise."""
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices = prices[TICKERS].sort_index()
    prices = prices.loc[TRAIN_START:TEST_END]
    prices = prices.ffill().dropna()
    return prices


# ======================================================================
# ETAPA 2 - MODELO: SINAIS DE MOMENTUM (SMA 3x10)
# ======================================================================

def compute_sma_signals(prices):
    """
    Sinal de tendência por cruzamento de médias móveis simples.
    Sinal = 1 (comprado) quando SMA_curta > SMA_longa, 0 caso contrário.
    Deslocado em 1 dia (.shift(1)) para evitar look-ahead bias: o sinal
    calculado com o fechamento de hoje só pode ser executado no fechamento
    de amanhã.
    """
    sma_signals = pd.DataFrame(index=prices.index)
    for ticker in ASSETS:
        sma_s = prices[ticker].rolling(SMA_SHORT).mean()
        sma_l = prices[ticker].rolling(SMA_LONG).mean()
        raw_signal = (sma_s > sma_l).astype(int)
        sma_signals[ticker] = raw_signal.shift(1).fillna(0)
    return sma_signals


# ======================================================================
# ETAPA 3 - MODELO: FILTRO ARIMA-GARCH(1,1) DE VOLATILIDADE CONDICIONAL
# ======================================================================

def compute_garch_vol_filter(returns, refit_every=126, min_obs=252):
    """
    Para cada ativo, ajusta de forma "rolling" (sem look-ahead) um modelo
    ARIMA(1,0,1)-GARCH(1,1):
        - ARIMA(1,0,1) sobre os retornos diários (%) para extrair a média
          condicional e gerar os resíduos.
        - GARCH(1,1) sobre os resíduos do ARIMA, estimando a variância
          condicional sigma2_t.

    O reajuste completo dos modelos é feito de forma prospectiva e recursiva
    fora da amostra (out-of-sample) a cada `refit_every` dias úteis, evitando
    qualquer look-ahead bias nos parâmetros ou na volatilidade estimada.

    Retorna um DataFrame com a volatilidade condicional diária estimada
    (sigma_t, em % ao dia) para cada ativo.
    """
    vol_filter = pd.DataFrame(index=returns.index, columns=ASSETS, dtype=float)

    for ticker in ASSETS:
        r = returns[ticker].dropna() * 100  # em % para estabilidade numérica
        idx = r.index
        n = len(r)

        sigma_values = np.full(n, np.nan)
        
        # O primeiro bloco [0, min_obs) é in-sample
        train_r = r.iloc[:min_obs]
        arima_model = fit_arima(train_r, p=1, d=0, q=1)
        garch_model = fit_garch(arima_model["residuals"])
        p_arima = arima_model["p"]
        
        # Salvar valores in-sample para o período inicial de treinamento
        sigma2_full = garch_model["sigma2"]
        sigma_values[p_arima:min_obs] = np.sqrt(sigma2_full)
        
        # Propagar de forma rolling prospectiva para o out-of-sample
        refit_points = list(range(min_obs, n, refit_every))
        if not refit_points or refit_points[-1] != n:
            refit_points.append(n)
            
        for i in range(len(refit_points) - 1):
            seg_start = refit_points[i]
            seg_end = refit_points[i+1]
            
            # Ajustar o modelo estritamente com dados passados até seg_start
            train_r = r.iloc[:seg_start]
            arima_model = fit_arima(train_r, p=1, d=0, q=1)
            garch_model = fit_garch(arima_model["residuals"])
            
            c = arima_model["c"]
            phi = arima_model["phi"][0] if len(arima_model["phi"]) > 0 else 0.0
            theta = arima_model["theta"][0] if len(arima_model["theta"]) > 0 else 0.0
            
            omega = garch_model["omega"]
            alpha = garch_model["alpha"]
            beta = garch_model["beta"]
            
            # Últimos valores in-sample no final do treino
            last_resid = arima_model["residuals"][-1]
            last_sigma2 = garch_model["sigma2"][-1]
            
            # Propagar recursivamente dia a dia sem vazar dados futuros
            for t in range(seg_start, seg_end):
                # Volatilidade de 1 passo à frente para o dia t
                sigma2_t = omega + alpha * (last_resid ** 2) + beta * last_sigma2
                sigma2_t = max(sigma2_t, 1e-12)
                sigma_values[t] = np.sqrt(sigma2_t)
                
                # Resíduo do dia t
                r_t = r.iloc[t]
                r_prev = r.iloc[t-1]
                resid_t = r_t - (c + phi * r_prev + theta * last_resid)
                
                # Atualizar estados
                last_resid = resid_t
                last_sigma2 = sigma2_t

        sigma_series = pd.Series(sigma_values, index=idx)
        sigma_series = sigma_series.ffill().bfill()
        vol_filter[ticker] = sigma_series / 100  # devolve para escala decimal

    return vol_filter


def compute_vol_signal(vol_filter, vol_threshold=VOL_THRESHOLD):
    """
    Sinal de filtro de volatilidade: 1 (libera operação) quando a
    volatilidade condicional do GARCH está abaixo de `vol_threshold` vezes
    a média expansiva (histórica até o momento) da própria volatilidade
    condicional. Usa expanding mean para não usar informação futura.
    """
    vol_signal = pd.DataFrame(index=vol_filter.index, columns=ASSETS, dtype=float)
    for ticker in ASSETS:
        hist_mean = vol_filter[ticker].expanding(min_periods=60).mean()
        sig = (vol_filter[ticker] <= vol_threshold * hist_mean).astype(float)
        sig = sig.shift(1).fillna(1.0)  # shift p/ evitar look-ahead
        vol_signal[ticker] = sig.astype(int)
    return vol_signal


# ======================================================================
# ETAPA 4 - ALOCAÇÃO: MARKOWITZ (MÉDIA-VARIÂNCIA, SEM BLACK-LITTERMAN)
# ======================================================================

def rolling_markowitz_weights(returns, active_signal, freq=REBALANCE_FREQ_DAYS,
                               lookback=252, max_weight=0.35):
    """
    A cada `freq` dias úteis, recalibra os pesos da carteira via otimização
    de Markowitz (maximização do Índice de Sharpe), usando APENAS os
    retornos dos últimos `lookback` dias ANTERIORES à data de
    rebalanceamento (sem look-ahead bias).

    Apenas os ativos com sinal de tendência ativo (SMA + filtro de
    volatilidade ambos positivos) entram na otimização; os demais recebem
    peso zero naquele rebalanceamento. Se nenhum ativo tiver sinal ativo, a
    carteira fica 100% em caixa (retorno zero) até o próximo
    rebalanceamento.
    """
    dates = returns.index
    weights_over_time = pd.DataFrame(0.0, index=dates, columns=ASSETS)

    rebalance_dates = dates[lookback::freq]
    if len(rebalance_dates) == 0 or rebalance_dates[-1] != dates[-1]:
        rebalance_dates = rebalance_dates.append(pd.DatetimeIndex([dates[-1]]))

    current_weights = pd.Series(0.0, index=ASSETS)

    for i in range(len(rebalance_dates) - 1):
        reb_date = rebalance_dates[i]
        next_reb_date = rebalance_dates[i+1]
        
        reb_loc = dates.get_loc(reb_date)
        next_reb_loc = dates.get_loc(next_reb_date)
        
        window = returns.iloc[reb_loc - lookback: reb_loc][ASSETS]  # estritamente passado

        active_today = active_signal.loc[reb_date]
        active_assets = [a for a in ASSETS if active_today[a] == 1]

        if len(active_assets) == 0:
            current_weights = pd.Series(0.0, index=ASSETS)
        elif len(active_assets) == 1:
            current_weights = pd.Series(0.0, index=ASSETS)
            current_weights[active_assets[0]] = 1.0
        else:
            sub_returns = window[active_assets]
            mean_ret = sub_returns.mean().values * TRADING_DAYS
            cov_mat = sub_returns.cov().values * TRADING_DAYS

            result = markowitz_max_sharpe(mean_ret, cov_mat,
                                           risk_free_rate=RISK_FREE_RATE,
                                           max_weight=max_weight)
            current_weights = pd.Series(0.0, index=ASSETS)
            for w, a in zip(result["weights"], active_assets):
                current_weights[a] = max(w, 0.0)
            if current_weights.sum() > 0:
                current_weights /= current_weights.sum()

        # Preenche os pesos de forma prospectiva do rebalanceamento atual até o próximo
        weights_over_time.iloc[reb_loc:next_reb_loc] = current_weights.values

    # Preenche o último segmento até o fim da série
    if len(rebalance_dates) > 0:
        last_reb_loc = dates.get_loc(rebalance_dates[-1])
        weights_over_time.iloc[last_reb_loc:] = current_weights.values

    return weights_over_time


# ======================================================================
# ETAPA 5 - BACKTEST: APLICAÇÃO DOS PESOS, CUSTOS E CÁLCULO DE RETORNOS
# ======================================================================

def run_backtest(returns, weights, transaction_cost=TRANSACTION_COST):
    """
    Calcula o retorno diário líquido da carteira:
        retorno_bruto_t = sum_i ( peso_{i,t-1} * retorno_{i,t} )
        custo_t         = sum_i ( |peso_{i,t} - peso_{i,t-1}| * custo_transacao )
        retorno_liq_t   = retorno_bruto_t - custo_t

    Os pesos usados em t são os pesos definidos no fechamento de t-1
    (shift(1)), de forma que o retorno do dia t reflete a carteira que já
    estava montada antes da abertura do pregão - sem look-ahead.
    """
    weights_shifted = weights.shift(1).fillna(0.0)
    gross_return = (weights_shifted[ASSETS] * returns[ASSETS]).sum(axis=1)

    turnover = weights.diff().abs().sum(axis=1).fillna(weights.iloc[0].abs().sum())
    costs = turnover * transaction_cost

    net_return = gross_return - costs
    return net_return, gross_return, turnover


# ======================================================================
# ETAPA 6 - MÉTRICAS DE DESEMPENHO
# ======================================================================

def performance_metrics(returns_series, risk_free_rate=RISK_FREE_RATE, weights=None):
    """Calcula as métricas obrigatórias do projeto: retorno total e
    anualizado, volatilidade anualizada, Sharpe, max drawdown e duração da
    recuperação, número de operações e taxa de acerto."""
    r = returns_series.dropna()
    cum = (1 + r).cumprod()
    total_return = cum.iloc[-1] - 1
    n_years = len(r) / TRADING_DAYS
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else np.nan

    vol = r.std() * np.sqrt(TRADING_DAYS)
    rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    sharpe = ((r - rf_daily).mean() / r.std()) * np.sqrt(TRADING_DAYS) if r.std() > 0 else np.nan

    # Cálculo correto da Downside Deviation para o Índice de Sortino
    excess_returns = r - rf_daily
    downside_returns = np.minimum(0, excess_returns)
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
    sortino = (excess_returns.mean() / downside_deviation) * np.sqrt(TRADING_DAYS) if downside_deviation > 0 else np.nan

    roll_max = cum.cummax()
    drawdown = cum / roll_max - 1
    max_dd = drawdown.min()

    end_dd = drawdown.idxmin()
    start_dd = cum.loc[:end_dd].idxmax()
    recovery_slice = cum.loc[end_dd:]
    recovered = recovery_slice[recovery_slice >= cum.loc[start_dd]]
    if len(recovered) > 0:
        # Medir do pico (início do drawdown) até a recuperação completa (padrão de mercado)
        recovery_days = (recovered.index[0] - start_dd).days
        recovery_str = f"{recovery_days} dias"
    else:
        recovery_str = f"{(cum.index[-1] - start_dd).days} dias (nao recuperado ate o fim da amostra)"

    active = r[r != 0]
    win_rate = (active > 0).mean() if len(active) > 0 else np.nan
    
    # Se os pesos forem fornecidos, calcula o número de dias com rebalanceamentos/operações reais
    if weights is not None:
        n_trades = int((weights.diff().abs().sum(axis=1) > 1e-5).sum())
    else:
        n_trades = int((r != 0).astype(int).diff().abs().sum() / 2) if len(r) > 1 else 0
        
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "Retorno total": total_return,
        "Retorno anualizado (CAGR)": cagr,
        "Volatilidade anualizada": vol,
        "Indice de Sharpe": sharpe,
        "Indice de Sortino": sortino,
        "Indice de Calmar": calmar,
        "Max drawdown": max_dd,
        "Duracao da recuperacao": recovery_str,
        "Numero aproximado de operacoes": n_trades,
        "Taxa de acerto (dias ativos)": win_rate,
    }


def fmt_metrics(metrics_dict):
    """Formata métricas para exibição (%, 2 casas decimais)."""
    pct_keys = {"Retorno total", "Retorno anualizado (CAGR)",
                "Volatilidade anualizada", "Max drawdown",
                "Taxa de acerto (dias ativos)"}
    out = {}
    for k, v in metrics_dict.items():
        if k in pct_keys and isinstance(v, (int, float)):
            out[k] = f"{v:.2%}"
        elif isinstance(v, (int, float)):
            out[k] = f"{v:.3f}" if not np.isnan(v) else "N/D"
        else:
            out[k] = v
    return out


# ======================================================================
# EXECUÇÃO PRINCIPAL
# ======================================================================

def main():
    print("=" * 70)
    print("PROJETO FINAL - ESTRATEGIA QUANTITATIVA: MERCADO DE URANIO")
    print("=" * 70)

    # ---------- 1. Dados ----------
    print("\n[1/7] Carregando dados historicos reais (2019-12-02 a 2026-06-26)...")
    prices = load_data("historical_prices.csv")
    returns = prices.pct_change().fillna(0)
    print(f"  Periodo carregado: {prices.index[0].date()} a {prices.index[-1].date()} "
          f"({len(prices)} pregoes)")

    # ---------- 2. Sinais SMA 3x10 ----------
    print("\n[2/7] Calculando sinais de momentum SMA(63) x SMA(210)...")
    sma_signals = compute_sma_signals(prices)

    # ---------- 3. Filtro ARIMA-GARCH(1,1) ----------
    print("\n[3/7] Ajustando ARIMA(1,0,1)-GARCH(1,1) por ativo (pode levar alguns minutos)...")
    vol_filter = compute_garch_vol_filter(returns)
    vol_signal = compute_vol_signal(vol_filter)

    combined_signal = (sma_signals[ASSETS].astype(int) & vol_signal[ASSETS].astype(int)).astype(int)
    print("  Sinais combinados (SMA + filtro de volatilidade) calculados.")

    # ---------- 4. Alocação de Markowitz (rolling, sem look-ahead) ----------
    print("\n[4/7] Otimizando pesos via Markowitz (media-variancia) a cada rebalanceamento trimestral...")
    weights = rolling_markowitz_weights(returns, combined_signal)

    # ---------- 5. Backtest ----------
    print("\n[5/7] Executando backtest com custos de transacao de 10 bps...")
    net_return, gross_return, turnover = run_backtest(returns, weights)

    train_mask = (net_return.index >= TRAIN_START) & (net_return.index <= TRAIN_END)
    test_mask = (net_return.index >= TEST_START) & (net_return.index <= TEST_END)

    strat_test = net_return[test_mask]
    strat_train = net_return[train_mask]
    bench_test = returns[BENCHMARK][test_mask]
    bench_train = returns[BENCHMARK][train_mask]
    bench_full = returns[BENCHMARK]

    # Buy & hold equal-weight do universo de urânio, para contexto
    bh_uranio = returns[ASSETS].mean(axis=1)
    bh_uranio_test = bh_uranio[test_mask]

    # ---------- 6. Métricas ----------
    print("\n[6/7] Calculando metricas de desempenho, VaR, CVaR e Monte Carlo...")

    metrics_strat_test = performance_metrics(strat_test, weights=weights.loc[strat_test.index])
    metrics_bench_test = performance_metrics(bench_test)
    metrics_bh_test = performance_metrics(bh_uranio_test)

    var_h, cvar_h = historical_var_cvar(strat_test, confidence=0.95)
    var_p, cvar_p = parametric_var_cvar(strat_test, confidence=0.95)
    var_h_b, cvar_h_b = historical_var_cvar(bench_test, confidence=0.95)

    metrics_strat_test["VaR 95% historico (diario)"] = var_h
    metrics_strat_test["CVaR 95% historico (diario)"] = cvar_h
    metrics_strat_test["VaR 95% parametrico (diario)"] = var_p
    metrics_strat_test["CVaR 95% parametrico (diario)"] = cvar_p

    metrics_bench_test["VaR 95% historico (diario)"] = var_h_b
    metrics_bench_test["CVaR 95% historico (diario)"] = cvar_h_b

    # Monte Carlo (block bootstrap) com base nos retornos OUT-OF-SAMPLE da estrategia
    mc_sims = monte_carlo_block_bootstrap(strat_test, n_sims=5000, horizon_days=252, block_size=20)
    mc_summary = monte_carlo_summary(mc_sims)

    mc_sims_bench = monte_carlo_block_bootstrap(bench_test, n_sims=5000, horizon_days=252, block_size=20)
    mc_summary_bench = monte_carlo_summary(mc_sims_bench)

    # Fronteira eficiente (ilustrativa) com dados out-of-sample completos
    mean_ret_full = returns[ASSETS][test_mask].mean().values * TRADING_DAYS
    cov_full = returns[ASSETS][test_mask].cov().values * TRADING_DAYS
    frontier_vols, frontier_rets = efficient_frontier(mean_ret_full, cov_full, n_points=30)
    max_sharpe_pf = markowitz_max_sharpe(mean_ret_full, cov_full, risk_free_rate=RISK_FREE_RATE, max_weight=0.35)
    min_var_pf = markowitz_min_variance(mean_ret_full, cov_full)

    # ---------- Tabela consolidada ----------
    df_metrics = pd.DataFrame({
        "Estrategia (out-of-sample)": fmt_metrics(metrics_strat_test),
        "Benchmark SPY (out-of-sample)": fmt_metrics(metrics_bench_test),
        "Buy&Hold Universo Uranio (out-of-sample)": fmt_metrics(metrics_bh_test),
    })

    print("\n" + "=" * 70)
    print("RESULTADOS - PERIODO OUT-OF-SAMPLE (2023-01-01 a 2026-06-26)")
    print("=" * 70)
    print(df_metrics.to_string())

    print("\nMarkowitz - Carteira de Sharpe Maximo (dados out-of-sample completos):")
    for a, w in zip(ASSETS, max_sharpe_pf["weights"]):
        print(f"  {a}: {w:.2%}")
    print(f"  Retorno esperado anualizado: {max_sharpe_pf['expected_return']:.2%}")
    print(f"  Volatilidade esperada anualizada: {max_sharpe_pf['expected_vol']:.2%}")
    print(f"  Sharpe esperado: {max_sharpe_pf['sharpe']:.2f}")

    print("\nSimulacao de Monte Carlo (block bootstrap, horizonte 12 meses, 5000 simulacoes):")
    print(f"  Estrategia - retorno medio simulado: {mc_summary['mean_final_return']:.2%}")
    print(f"  Estrategia - retorno mediano simulado: {mc_summary['median_final_return']:.2%}")
    print(f"  Estrategia - intervalo 90% (P5-P95): [{mc_summary['p5_final_return']:.2%}, {mc_summary['p95_final_return']:.2%}]")
    print(f"  Estrategia - probabilidade de perda em 12 meses: {mc_summary['prob_loss']:.2%}")
    print(f"  Estrategia - CVaR 95% terminal (12m): {mc_summary['cvar_95_terminal']:.2%}")

    # ---------- 7. Graficos ----------
    print("\n[7/7] Gerando graficos...")

    # 7.1 Equity curve completa (treino + teste) com marcacao do split
    fig, ax = plt.subplots(figsize=(12, 6))
    full_cum_strat = (1 + net_return).cumprod()
    full_cum_bench = (1 + bench_full).cumprod()
    ax.plot(full_cum_strat.index, full_cum_strat, label="Estrategia (SMA+ARIMA-GARCH+Markowitz)",
            color=COLORS["Estrategia"], linewidth=2)
    ax.plot(full_cum_bench.index, full_cum_bench, label="Benchmark SPY",
            color=COLORS["SPY"], linewidth=1.6)
    for a in ASSETS:
        bh = (1 + returns[a]).cumprod()
        ax.plot(bh.index, bh, label=f"Buy & Hold {a}", color=COLORS[a],
                linewidth=1, alpha=0.45, linestyle="--")
    ax.axvline(pd.Timestamp(TEST_START), color="black", linestyle=":", linewidth=1.2)
    ax.annotate("Inicio out-of-sample", xy=(pd.Timestamp(TEST_START), ax.get_ylim()[1] * 0.95),
                fontsize=9, rotation=90, va="top")
    ax.set_title("Curva de capital - Estrategia Quantitativa de Uranio vs. Benchmark e Buy&Hold")
    ax.set_xlabel("Data")
    ax.set_ylabel("Capital acumulado (base 1.0)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig("equity_curve.png")
    plt.close(fig)

    # 7.2 Drawdown (out-of-sample)
    fig, ax = plt.subplots(figsize=(12, 4))
    cum_s = (1 + strat_test).cumprod()
    dd_s = cum_s / cum_s.cummax() - 1
    cum_b = (1 + bench_test).cumprod()
    dd_b = cum_b / cum_b.cummax() - 1
    ax.fill_between(dd_s.index, dd_s, 0, color=COLORS["Estrategia"], alpha=0.35, label="Drawdown Estrategia")
    ax.plot(dd_b.index, dd_b, color=COLORS["SPY"], linewidth=1.3, label="Drawdown SPY")
    ax.set_title("Drawdown - Periodo Out-of-Sample (2023-2026)")
    ax.set_ylabel("Drawdown")
    ax.legend()
    fig.tight_layout()
    fig.savefig("drawdown_analysis.png")
    plt.close(fig)

    # 7.3 Distribuicao de retornos + VaR/CVaR
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(strat_test[strat_test != 0], bins=60, color=COLORS["Estrategia"], alpha=0.55,
            label="Estrategia (dias ativos)", density=True)
    ax.hist(bench_test, bins=60, color=COLORS["SPY"], alpha=0.4, label="SPY", density=True)
    ax.axvline(var_h, color=COLORS["Estrategia"], linestyle="--", linewidth=1.8,
               label=f"VaR 95% Estrategia: {var_h:.2%}")
    ax.axvline(cvar_h, color=COLORS["Markowitz"], linestyle=":", linewidth=1.8,
               label=f"CVaR 95% Estrategia: {cvar_h:.2%}")
    ax.set_title("Distribuicao de retornos diarios e risco de cauda (out-of-sample)")
    ax.set_xlabel("Retorno diario")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("return_distribution.png")
    plt.close(fig)

    # 7.4 Fronteira eficiente de Markowitz
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(frontier_vols, frontier_rets, color=COLORS["Markowitz"], linewidth=2, label="Fronteira eficiente")
    ax.scatter([max_sharpe_pf["expected_vol"]], [max_sharpe_pf["expected_return"]],
               color="#D85A30", s=90, zorder=5, label="Carteira de Sharpe maximo", marker="*")
    ax.scatter([min_var_pf["expected_vol"]], [min_var_pf["expected_return"]],
               color="#1D9E75", s=70, zorder=5, label="Minima variancia", marker="o")
    for a in ASSETS:
        a_ret = returns[a][test_mask].mean() * TRADING_DAYS
        a_vol = returns[a][test_mask].std() * np.sqrt(TRADING_DAYS)
        ax.scatter([a_vol], [a_ret], color=COLORS[a], s=55, label=a)
    ax.set_xlabel("Volatilidade anualizada")
    ax.set_ylabel("Retorno esperado anualizado")
    ax.set_title("Fronteira eficiente de Markowitz - Universo de Uranio (out-of-sample)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("efficient_frontier.png")
    plt.close(fig)

    # 7.5 Monte Carlo fan chart
    fig, ax = plt.subplots(figsize=(11, 6))
    horizon = np.arange(mc_sims.shape[1])
    p5 = np.percentile(mc_sims, 5, axis=0)
    p25 = np.percentile(mc_sims, 25, axis=0)
    p50 = np.percentile(mc_sims, 50, axis=0)
    p75 = np.percentile(mc_sims, 75, axis=0)
    p95 = np.percentile(mc_sims, 95, axis=0)
    ax.fill_between(horizon, p5, p95, color=COLORS["Estrategia"], alpha=0.15, label="Intervalo 90% (P5-P95)")
    ax.fill_between(horizon, p25, p75, color=COLORS["Estrategia"], alpha=0.3, label="Intervalo 50% (P25-P75)")
    ax.plot(horizon, p50, color=COLORS["Estrategia"], linewidth=2, label="Mediana simulada")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title("Simulacao de Monte Carlo (block bootstrap) - 12 meses, 5.000 trajetorias")
    ax.set_xlabel("Dias uteis a frente")
    ax.set_ylabel("Capital acumulado (base 1.0)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("monte_carlo_fanchart.png")
    plt.close(fig)

    # 7.6 GARCH: volatilidade condicional ao longo do tempo
    fig, axes = plt.subplots(len(ASSETS), 1, figsize=(12, 9), sharex=True)
    for ax, a in zip(axes, ASSETS):
        ax.plot(vol_filter.index, vol_filter[a] * np.sqrt(TRADING_DAYS), color=COLORS[a], linewidth=1)
        ax.set_ylabel(a, fontsize=9)
        ax.axvline(pd.Timestamp(TEST_START), color="black", linestyle=":", linewidth=0.8)
    axes[0].set_title("Volatilidade condicional ARIMA-GARCH(1,1) anualizada, por ativo")
    axes[-1].set_xlabel("Data")
    fig.tight_layout()
    fig.savefig("garch_volatility.png")
    plt.close(fig)

    print("  Graficos salvos: equity_curve.png, drawdown_analysis.png, return_distribution.png,")
    print("                   efficient_frontier.png, monte_carlo_fanchart.png, garch_volatility.png")

    # ---------- Salvar métricas e pesos ----------
    df_metrics.to_csv("metrics_summary.csv", encoding="utf-8-sig")
    weights.to_csv("markowitz_weights_over_time.csv", encoding="utf-8-sig")

    mc_table = pd.DataFrame({
        "Estrategia": mc_summary, "Benchmark SPY": mc_summary_bench
    }).drop(["final_returns_dist"])
    mc_table.to_csv("monte_carlo_summary.csv", encoding="utf-8-sig")

    print("\nArquivos salvos: metrics_summary.csv, markowitz_weights_over_time.csv, monte_carlo_summary.csv")
    print("\nBacktest concluido com sucesso.")

    return {
        "metrics_strat_test": metrics_strat_test,
        "metrics_bench_test": metrics_bench_test,
        "metrics_bh_test": metrics_bh_test,
        "max_sharpe_pf": max_sharpe_pf,
        "min_var_pf": min_var_pf,
        "mc_summary": mc_summary,
        "mc_summary_bench": mc_summary_bench,
    }


if __name__ == "__main__":
    results = main()
