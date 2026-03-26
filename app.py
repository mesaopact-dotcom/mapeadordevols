import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import datetime
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# =============================================================================
# AAI SENIOR - MOTOR DE CÁLCULO FINANCEIRO (BLACK-SCHOLES)
# =============================================================================

class BlackScholesEngine:
    """
    Motor de cálculo para Volatilidade Implícita e Gregas.
    Baseado em Black-Scholes (1973).
    """
    
    @staticmethod
    def call_price(S, K, T, r, sigma):
        if T <= 0: return max(0, S - K)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S, K, T, r, sigma):
        if T <= 0: return max(0, K - S)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def implied_volatility(market_price, S, K, T, r, option_type='call'):
        if market_price <= 0.01 or T <= 0:
            return 0.0
        
        def objective_function(sigma):
            if option_type.lower() == 'call':
                return BlackScholesEngine.call_price(S, K, T, r, sigma) - market_price
            else:
                return BlackScholesEngine.put_price(S, K, T, r, sigma) - market_price

        try:
            return brentq(objective_function, 1e-6, 5.0)
        except (ValueError, RuntimeError):
            return 0.0

    @staticmethod
    def calculate_greeks(S, K, T, r, sigma, option_type='call'):
        if T <= 0 or sigma <= 0:
            return {"Delta": 0, "Gamma": 0, "Vega": 0, "Theta": 0}
            
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        pdf_d1 = norm.pdf(d1)
        
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
            theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) 
                     - r * K * np.exp(-r * T) * norm.cdf(d2))
        else:
            delta = norm.cdf(d1) - 1
            theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) 
                     + r * K * np.exp(-r * T) * norm.cdf(-d2))
            
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * pdf_d1
        
        return {
            "Delta": delta,
            "Gamma": gamma,
            "Vega": vega / 100,
            "Theta": theta / 365
        }

# =============================================================================
# INTERFACE STREAMLIT - DASHBOARD DO ASSESSOR SÉNIOR
# =============================================================================

def main():
    st.set_page_config(page_title="AAI Senior - Volatility Smile Mapper", layout="wide")
    
    st.sidebar.title("💎 AAI SENIOR")
    st.sidebar.markdown("**Mapeador de Skew & Smile**")
    st.sidebar.markdown("---")
    
    # 1. Parâmetros de Entrada
    ticker_input = st.sidebar.text_input("Ticker Ativo Objeto (B3)", "PETR4.SA").upper()
    risk_free_rate = st.sidebar.slider("Taxa Selic (% a.a.)", 0.0, 20.0, 10.75) / 100
    
    st.title("📊 Mapeador de Volatilidade Implícita e Smile")
    st.caption("Visão Estratégica para Assessoria de Alta Performance")

    try:
        asset = yf.Ticker(ticker_input)
        spot_price = asset.history(period="1d")['Close'].iloc[-1]
        
        # Obter datas de vencimento disponíveis
        expirations = asset.options
        if not expirations:
            st.error("Nenhuma opção encontrada para este ticker no momento.")
            return

        selected_expiry = st.sidebar.selectbox("Vencimento da Opção", expirations)
        
        # 2. Processamento da Grade de Opções
        opt_chain = asset.option_chain(selected_expiry)
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        # Calcular tempo para o vencimento
        expiry_dt = datetime.datetime.strptime(selected_expiry, '%Y-%m-%d')
        T = (expiry_dt - datetime.datetime.now()).days / 365
        
        st.subheader(f"Análise de {ticker_input} - Spot: R$ {spot_price:.2f}")
        
        # Interface de abas
        tab1, tab2, tab3 = st.tabs(["😊 Smile de Volatilidade", "📋 Grade de IV & Gregas", "📈 Payoff de Estratégia"])

        with tab1:
            # Processar IV para a grade
            calls_subset = calls[(calls['strike'] > spot_price * 0.7) & (calls['strike'] < spot_price * 1.3)].copy()
            calls_subset['IV_Calculada'] = calls_subset.apply(
                lambda row: BlackScholesEngine.implied_volatility(row['lastPrice'], spot_price, row['strike'], T, risk_free_rate, 'call'), axis=1
            )
            
            # Filtro para remover IVs zeradas ou anormais (outliers de baixa liquidez)
            calls_plot = calls_subset[calls_subset['IV_Calculada'] > 0.05]

            fig_smile = px.line(calls_plot, x='strike', y='IV_Calculada', 
                                title=f"Smile de Volatilidade (Calls) - Vencimento {selected_expiry}",
                                labels={'strike': 'Strike (R$)', 'IV_Calculada': 'Volatilidade Implícita'},
                                markers=True, template="plotly_dark")
            fig_smile.add_vline(x=spot_price, line_dash="dash", line_color="cyan", annotation_text="At-the-money")
            st.plotly_chart(fig_smile, use_container_width=True)
            
            st.info("""
            **Análise do AAI Senior:** O "Smile" acima mostra como o mercado precifica o risco. 
            Se a curva estiver inclinada para a esquerda (*Skew de Baixa*), as Puts estão mais caras, indicando medo de queda.
            """)

        with tab2:
            st.markdown("### Grade Detalhada de Opções (ITM/ATM/OTM)")
            # Preparar dataframe para exibição
            display_df = calls_subset[['contractSymbol', 'strike', 'lastPrice', 'IV_Calculada']].copy()
            display_df['IV %'] = (display_df['IV_Calculada'] * 100).round(2)
            
            # Adicionar Gregas
            display_df['Delta'] = display_df.apply(
                lambda row: BlackScholesEngine.calculate_greeks(spot_price, row['strike'], T, risk_free_rate, row['IV_Calculada'], 'call')['Delta'], axis=1
            ).round(3)
            
            display_df['Vega (1%)'] = display_df.apply(
                lambda row: BlackScholesEngine.calculate_greeks(spot_price, row['strike'], T, risk_free_rate, row['IV_Calculada'], 'call')['Vega'], axis=1
            ).round(4)

            st.dataframe(display_df.drop(columns=['IV_Calculada']), use_container_width=True)

        with tab3:
            st.markdown("### Simulador de Operação")
            col_opt1, col_opt2 = st.columns(2)
            
            with col_opt1:
                sel_strike = st.selectbox("Escolha o Strike para simular", display_df['strike'].tolist())
                opt_data = display_df[display_df['strike'] == sel_strike].iloc[0]
                price_sim = opt_data['lastPrice']
                iv_sim = opt_data['IV_Calculada']
                
            with col_opt2:
                st.metric("Prêmio da Opção", f"R$ {price_sim:.2f}")
                st.metric("Delta da Operação", f"{opt_data['Delta']}")

            # Gráfico de Payoff dinâmico
            s_range = np.linspace(spot_price * 0.8, spot_price * 1.2, 100)
            payoff = np.maximum(s_range - sel_strike, 0) - price_sim
            
            fig_payoff = go.Figure()
            fig_payoff.add_trace(go.Scatter(x=s_range, y=payoff, name="Resultado no Vencimento", fill='tozeroy'))
            fig_payoff.add_hline(y=0, line_dash="dash", line_color="white")
            fig_payoff.update_layout(title=f"Payoff de Compra de Call: Strike {sel_strike}", template="plotly_dark")
            st.plotly_chart(fig_payoff, use_container_width=True)

        # 6. Disclaimer Regulatório (MANDATÓRIO)
        st.sidebar.markdown("---")
        st.sidebar.warning("""
        **Compliance CVM 178:** Dados históricos/públicos. 
        Não é recomendação. 
        Verifique suitability.
        """)

    except Exception as e:
        st.error(f"Erro na conexão com os dados da B3: {str(e)}")
        st.info("Dica: Use tickers como PETR4.SA, VALE3.SA ou ITUB4.SA")

if __name__ == "__main__":
    main()