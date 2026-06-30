# Projeto Final — Estratégia Quantitativa: Mercado de Urânio

**Núcleo Quant · Liga de Mercado Financeiro · UFU · 2026.1**

Fluxo do projeto: **Tese → Modelo → Backtest → Resultado**

---

## 1. Tese

Ativos ligados ao urânio (ETFs setoriais e mineradoras) estão estruturalmente
subprecificados frente à pressão de demanda futura, sustentada por três
vetores geopolíticos independentes:

1. **Segurança energética** — a guerra na Ucrânia acelerou a diversificação
   do Ocidente em relação ao urânio enriquecido russo, levando o preço spot
   de US$ 43/lb (jan/2022) a US$ 100/lb (jan/2024).
2. **Compromisso multilateral de expansão nuclear** — a Declaração para
   Triplicar a Energia Nuclear (COP28, dez/2023) já reúne mais de 30 países
   comprometidos com a triplicação da capacidade nuclear instalada até 2050.
3. **Demanda de energia por data centers de IA** — grandes empresas de
   tecnologia (Microsoft, Google, Amazon) assinaram contratos de compra de
   energia nuclear de longo prazo; o BloombergNEF projeta a demanda de
   energia de data centers nos EUA quase triplicando até 2035.

Do lado da oferta, o Cazaquistão (maior produtor mundial, ~40% da produção
global) anunciou cortes de produção em 2025 e 2026 por restrições de
insumos (ácido sulfúrico) e decisões estratégicas, intensificando o
desequilíbrio estrutural entre oferta e demanda.

**Mecanismo da ineficiência:** o setor ficou mais de uma década fora do
radar de investidores institucionais após o acidente de Fukushima (2011),
gerando *underinvestment* crônico em capacidade produtiva. A correção desse
viés (reabertura de minas, requalificação de cadeias de suprimento) é
estruturalmente lenta, criando uma janela de mispricing que a estratégia
tenta capturar via momentum de preço, com controle estatístico de risco.

## 2. Universo de ativos

| Ticker | Descrição | Início real do histórico |
|---|---|---|
| URA | Global X Uranium ETF | 04/11/2010 |
| NLR | VanEck Uranium & Nuclear ETF | 13/08/2007 |
| CCJ | Cameco Corporation (maior mineradora do mundo) | ~1996 (NYSE) |
| URNM | Sprott Uranium Miners ETF | 03/12/2019 |
| SPY | SPDR S&P 500 ETF (benchmark) | — |

O universo de **dados utilizados neste projeto** começa em **02/12/2019**,
alinhado pela disponibilidade do ativo mais recente (URNM), conforme
solicitado. Isso fornece **1.650 pregões** (~6,5 anos) de dados diários
reais e ajustados, baixados do Yahoo Finance — acima do mínimo de 5 anos
exigido pelo projeto.

## 3. Divisão temporal (sem look-ahead bias)

| Período | Intervalo | Uso |
|---|---|---|
| **In-sample** (treino/calibração) | 2019-12-02 a 2022-12-30 (~3 anos) | Ajuste dos parâmetros do modelo |
| **Out-of-sample** (teste) | 2023-01-01 a 2026-06-26 (~3,5 anos) | Avaliação final, nunca usado para calibrar |

Todas as métricas de desempenho reportadas na Seção 6 referem-se
exclusivamente ao período **out-of-sample**.

## 4. Modelo

### 4.1 Sinal de tendência — SMA 3×10

Cruzamento de médias móveis simples sobre o preço de fechamento:

- SMA curta: 63 pregões (~3 meses)
- SMA longa: 210 pregões (~10 meses)
- Sinal = 1 (comprado) quando SMA(63) > SMA(210); 0 caso contrário
- Sinal deslocado em 1 dia (`shift(1)`) para evitar look-ahead bias

### 4.2 Filtro de risco — ARIMA(1,0,1)-GARCH(1,1)

Para cada ativo, ajustamos (de forma *rolling*, sem look-ahead, reajustado
a cada ~6 meses usando apenas dados passados):

1. **ARIMA(1,0,1)** sobre os retornos diários, para extrair a média
   condicional e gerar os resíduos do modelo.
2. **GARCH(1,1)** sobre os resíduos do ARIMA, estimando a variância
   condicional `σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}`.

Ambos os modelos foram **implementados do zero** em `quant_models.py`
(mínimos quadrados condicionais para o ARIMA; máxima verossimilhança
numérica via `scipy.optimize` para o GARCH), já que os pacotes `statsmodels`
e `arch` não estavam disponíveis no ambiente de execução. Isso tem a
vantagem adicional de tornar cada cálculo absolutamente transparente.

O sinal de tendência só é executado quando a volatilidade condicional do
GARCH não excede 1,5× a média histórica expansiva da própria volatilidade
condicional — um filtro de regime que bloqueia novas entradas em períodos
de estresse extremo (ex: pânico do COVID em março/2020, choque da invasão
da Ucrânia em fevereiro/2022).

### 4.3 Alocação de capital — Markowitz (média-variância, sem Black-Litterman)

A cada rebalanceamento trimestral (63 pregões), entre os ativos com sinal
combinado ativo (SMA **e** filtro de volatilidade simultaneamente positivos),
otimizamos os pesos da carteira maximizando o Índice de Sharpe esperado:

```
max_w   (w'μ − rf) / sqrt(w'Σw)
s.a.    Σw = 1,  0 ≤ w_i ≤ 1   (long-only)
```

`μ` e `Σ` (retorno esperado e matriz de covariância) são estimados
exclusivamente com a janela de 252 dias **anteriores** à data de
rebalanceamento — nunca com dados futuros. Implementado via
`scipy.optimize.minimize` (SLSQP). **Não foi usado Black-Litterman**,
conforme solicitado — apenas a otimização média-variância clássica.

### 4.4 Custos de transação

0,10% (10 bps) aplicados sobre toda variação de peso (turnover) entre um
rebalanceamento e o seguinte.

## 5. Vieses controlados

- **Look-ahead bias:** todos os sinais (SMA, GARCH, Markowitz) usam
  exclusivamente informação disponível até o fechamento do dia/período
  anterior à execução.
- **Overfitting:** os parâmetros (janelas de SMA, ordem ARIMA/GARCH,
  threshold de volatilidade, frequência de rebalanceamento) foram fixados
  por convenção de mercado/literatura — não foram otimizados sobre o
  período de teste.
- **Custos:** incluídos em toda mudança de posição.

## 6. Resultados (out-of-sample, 2023-01-01 a 2026-06-26)

| Métrica | Estratégia | Benchmark SPY | Buy & Hold Urânio (equal-weight) |
|---|---|---|---|
| Retorno total | **127,30%** | 99,36% | 170,61% |
| Retorno anualizado (CAGR) | **26,75%** | 22,04% | 33,29% |
| Volatilidade anualizada | 40,11% | 15,17% | 39,41% |
| Índice de Sharpe | **0,68** | **1,10** | 0,81 |
| Índice de Sortino | 0,93 | 1,50 | 1,28 |
| Índice de Calmar | 0,61 | 1,18 | 0,89 |
| Max drawdown | −43,74% | −18,76% | −37,52% |
| Duração da recuperação do MDD | 312 dias | 79 dias | 62 dias |
| Número aproximado de operações | 4 | 1 | 0 |
| Taxa de acerto (dias ativos) | 53,47% | 56,88% | 52,92% |
| VaR 95% histórico (diário) | −3,90% | −1,40% | — |
| CVaR 95% histórico (diário) | −5,80% | −2,10% | — |
| VaR 95% paramétrico (diário) | −4,00% | — | — |
| CVaR 95% paramétrico (diário) | −5,10% | — | — |

### Markowitz — carteira de Sharpe máximo (dados out-of-sample completos)

| Ativo | Peso ótimo |
|---|---|
| URA | 0,00% |
| NLR | 0,00% |
| CCJ | **100,00%** |
| URNM | 0,00% |

Retorno esperado anualizado: 55,10% · Volatilidade esperada: 46,66% ·
Sharpe esperado: 1,08.

A otimização converge para uma **solução de canto** (100% em CCJ) porque a
Cameco isoladamente domina em retorno médio e Sharpe individual o resto do
universo no período — um resultado correto matematicamente, mas que expõe
a fragilidade de Markowitz com poucos ativos altamente correlacionados (ver
Seção 8, Limitações).

### Simulação de Monte Carlo (block bootstrap, horizonte de 12 meses, 5.000 trajetórias)

| | Estratégia | Benchmark SPY |
|---|---|---|
| Retorno médio simulado | 33,29% | 23,45% |
| Retorno mediano simulado | 25,07% | 22,59% |
| Intervalo 90% (P5–P95) | [−32,14%, 126,76%] | [−2,18%, 50,33%] |
| Probabilidade de perda em 12 meses | **27,40%** | 6,46% |
| CVaR 95% terminal (12 meses) | −40,85% | −7,87% |

## 7. Interpretação e conclusão

**A tese é confirmada em retorno absoluto, mas refutada em retorno ajustado
ao risco.** A estratégia supera o S&P 500 em retorno total (127,3% vs.
99,4%) e em CAGR (26,75% vs. 22,04%) no período out-of-sample, validando o
vetor direcional da tese geopolítica: o ciclo de alta do urânio impulsionado
por segurança energética, compromissos de triplicação nuclear e demanda de
IA é real e mensurável nos preços.

Entretanto, o **Índice de Sharpe da estratégia (0,68) fica abaixo do
benchmark (1,10)** — a volatilidade do setor de urânio (40% ao ano, contra
15% do S&P 500) consome todo o prêmio de retorno em termos ajustados ao
risco. O mesmo padrão se repete no Sortino e no Calmar. Isso é uma
conclusão honesta e relevante: **o urânio entrega retorno, mas não retorno
eficiente** — um investidor que pondera retorno por unidade de risco
estaria, neste período, melhor alocado no S&P 500 puro do que na estratégia
quantitativa de urânio, apesar do retorno nominal superior.

O filtro ARIMA-GARCH cumpriu seu papel: a volatilidade condicional disparou
visivelmente em março/2020 (pânico do COVID) e fevereiro/2022 (invasão da
Ucrânia), bloqueando corretamente novas entradas nesses períodos de
estresse — mas não foi suficiente para evitar o grande drawdown de 2024-2025
(−43,74%), que ocorreu durante um regime de volatilidade moderada e
reflete principalmente a reversão natural do ciclo de alta do setor.

A simulação de Monte Carlo reforça essa leitura assimétrica de risco: a
probabilidade de perda em 12 meses para a estratégia (27,4%) é mais de
quatro vezes a do benchmark (6,5%), e o CVaR terminal de 12 meses
(−40,85%) é cerca de cinco vezes pior que o do SPY (−7,87%).

## 8. Limitações

- **Universo restrito e concentrado.** Apenas 4 ativos de urânio; a
  otimização de Markowitz convergiu para 100% em CCJ, expondo o portfólio
  ao risco idiossincrático de uma única empresa em vez de diversificar
  dentro do setor.
- **Correlação intrasetorial elevada.** URA, NLR, CCJ e URNM compartilham
  exposição quase total ao mesmo fator de risco (preço do urânio e
  sentimento do setor nuclear), o que limita o ganho real de diversificação
  da otimização de Markowitz — a "fronteira eficiente" do projeto é, na
  prática, quase uma linha entre dois pontos.
- **Período atípico.** O out-of-sample (2023–2026) inclui o boom de IA/data
  centers, um catalisador de demanda que não tem precedente histórico
  direto — o resultado pode não ser generalizável para outros ciclos do
  setor.
- **ARIMA-GARCH com reajuste semestral.** O modelo de volatilidade não é
  recalibrado diariamente por custo computacional; em transições rápidas de
  regime, a resposta do filtro pode ter defasagem de alguns dias.
- **Liquidez não modelada.** NLR e URNM têm volume diário menor que URA e
  CCJ; o custo de 10 bps assumido pode ser otimista para esses ativos em
  rebalanceamentos de maior porte.
- **CCJ sem dividendos/ajustes de eventos corporativos tratados à parte** —
  os preços usados já são ajustados pelo provedor de dados (Yahoo Finance),
  mas qualquer inconsistência de ajuste retroativo afeta igualmente
  benchmark e estratégia.

## 9. Como executar

```bash
pip install -r requirements.txt
python backtest_uranio.py
```

O script:

1. Carrega `historical_prices.csv` (dados reais, dez/2019–jun/2026);
2. Calcula os sinais SMA 3×10;
3. Ajusta ARIMA(1,0,1)-GARCH(1,1) por ativo (módulo `quant_models.py`);
4. Otimiza os pesos via Markowitz a cada rebalanceamento trimestral;
5. Roda o backtest com custos de transação;
6. Calcula todas as métricas, VaR, CVaR e Monte Carlo;
7. Gera os gráficos e salva os CSVs de resultado.

## 10. Arquivos do projeto

| Arquivo | Conteúdo |
|---|---|
| `backtest_uranio.py` | Script principal: dados, sinais, backtest, métricas, gráficos |
| `quant_models.py` | Implementações de ARIMA, GARCH(1,1), Markowitz, Monte Carlo, VaR/CVaR |
| `historical_prices.csv` | Dados reais de preços ajustados (URA, NLR, CCJ, URNM, SPY) |
| `metrics_summary.csv` | Tabela de métricas de desempenho out-of-sample |
| `monte_carlo_summary.csv` | Resumo estatístico da simulação de Monte Carlo |
| `markowitz_weights_over_time.csv` | Série histórica dos pesos ótimos de Markowitz |
| `equity_curve.png` | Curva de capital: estratégia vs. benchmark vs. buy & hold |
| `drawdown_analysis.png` | Análise de drawdown out-of-sample |
| `return_distribution.png` | Distribuição de retornos diários com VaR/CVaR |
| `efficient_frontier.png` | Fronteira eficiente de Markowitz do universo |
| `monte_carlo_fanchart.png` | Fan chart da simulação de Monte Carlo (12 meses) |
| `garch_volatility.png` | Volatilidade condicional ARIMA-GARCH por ativo ao longo do tempo |
| `requirements.txt` | Dependências do projeto |

## 11. Próximos passos (não exigidos pelo enunciado, mas honestos)

Caso o projeto continuasse: testar pesos por paridade de risco em vez de
Sharpe máximo (mitigaria a solução de canto em CCJ); ampliar o universo
para mineradoras individuais (Kazatomprom via ADR, Denison Mines, NexGen);
e recalibrar o GARCH em frequência mensal para reagir mais rápido a
mudanças de regime.
