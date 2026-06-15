import streamlit as st
import numpy as np
from motor_fisico import MotorRefletor
import matplotlib.pyplot as plt

# Configurando o Matplotlib para casar com a sua paleta de cores
plt.rcParams.update({
    'figure.facecolor': '#12060f',     # --fundo
    'axes.facecolor': '#21091d',       # --fundo-2
    'text.color': '#fff4df',           # --texto
    'axes.labelcolor': '#fff4df',      # --texto
    'xtick.color': '#d5bfd0',          # --suave
    'ytick.color': '#d5bfd0',          # --suave
    'grid.color': '#6d173a',           # --vinho (substituindo o rgba por uma cor da sua paleta)
    'grid.alpha': 0.3                  # Controle de opacidade nativo do Matplotlib
})

# Configuração da página para modo Wide (estilo dashboard)
st.set_page_config(layout="wide", page_title="Antenas Refletoras Lab")

st.title("Antenas Refletoras Lab — Plataforma Virtual de Antenas Refletoras")
st.markdown("---")

# Criando o menu lateral para parâmetros globais da antena
st.sidebar.header("Parâmetros Estruturais")
D = st.sidebar.slider("Diâmetro do Prato (m)", 0.5, 5.0, 2.4, 0.1)
f_over_d = st.sidebar.slider("Relação F/D (Focal/Diâmetro)", 0.25, 0.6, 0.4, 0.05)
freq_ghz = st.sidebar.slider("Frequência de Operação (GHz)", 1.0, 12.0, 5.0, 0.5)

# Inicializa o motor com os dados do menu lateral
motor = MotorRefletor(D, f_over_d, freq_ghz)

# Criando a estrutura de Três Abas da nossa plataforma
aba1, aba2, aba3 = st.tabs([
    "Aba 1: Sandbox de Varredura Focal", 
    "Aba 2: Processamento Multi-feed & Rastreamento", 
    "Aba 3: Simulador Orbitário LEO"
])

# ========================================================
# CONTEÚDO DA ABA 1: SANDBOX DE VARREDURA
# ========================================================
with aba1:
    st.subheader("Modelagem do Refletor e Deslocamento de Feixe (Beam Steering)")
    st.write("Mova o alimentador para fora do foco para inclinar o feixe eletromagnético eletronicamente.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("#### Controles Locais")
        delta_x = st.slider("Deslocamento do Alimentador Δx (m)", -0.3, 0.3, 0.0, 0.02)
        q = st.slider("Diretividade do Alimentador (q)", 1.0, 5.0, 2.5, 0.5)
        
    with col2:
        # Executa o cálculo físico da Aba 1
        dados = motor.calcular_sistema(delta_x, q)
        
        # Renderiza as métricas em caixas bonitas (Cards)
        m1, m2, m3 = st.columns(3)
        m1.metric("Distância Focal (F)", f"{dados['f']:.2f} m")
        m2.metric("Ganho Máximo Estimado", f"{dados['ganho']:.1f} dBi")
        m3.metric("Largura de Feixe (HPBW)", f"{dados['hpbw']:.2f}°")
        
        # Gerando os gráficos com Matplotlib
        fig = plt.figure(figsize=(14, 5))
        
        # Subplot 1: Geometria
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot_surface(dados["X"], dados["Y"], dados["Z"], cmap='viridis', alpha=0.6)
        ax1.scatter([delta_x], [0], [dados["f"]], color='red', s=100, label='Alimentador')
        ax1.set_title("Posicionamento Físico do Alimentador")
        ax1.legend()
        
        # Subplot 2: Diagrama de Irradiação
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(dados["theta_deg"], dados["pattern_dB"], color='dodgerblue', linewidth=2)
        ax2.axhline(-3, color='gray', linestyle=':')
        ax2.set_title("Diagrama de Irradiação (Efeito do Deslocamento)")
        ax2.set_ylabel("Ganho Relativo (dB)")
        ax2.set_xlabel("Ângulo (Graus)")
        ax2.grid(True, linestyle="--")
        ax2.set_ylim([-40, 5])
        
        st.pyplot(fig)

# ========================================================
# CONTEÚDO DA ABA 2: MULTI-FEED
# ========================================================
with aba2:
    st.subheader("Distribuição de Energia no Plano Focal (Região de Recepção)")
    st.write("Simule uma onda chegando do espaço com um ângulo inclinado e veja o foco de energia se mover no plano dos alimentadores.")
    
    col1_aba2, col2_aba2 = st.columns([1, 3])
    
    with col1_aba2:
        st.markdown("#### Satélite Transmissor")
        theta_inc = st.slider("Ângulo de Incidência do Sinal (Graus)", -10.0, 10.0, 0.0, 0.5)
        st.markdown("---")
        st.markdown("""
        **Conceito de Engenharia:**
        Quando o sinal chega em $0^\circ$ (perfeitamente alinhado), a energia se concentra exatamente no centro $(0,0)$. 
        
        Conforme o satélite se move, o ponto de máxima energia sai do centro. Colocando uma matriz de cornetas ali, podemos rastrear o sinal digitalmente!
        """)
        
    with col2_aba2:
        with st.spinner("Computando campos focais..."):
            # Calcula a matriz do plano focal
            XF, YF, intensidade = motor.calcular_campo_focal(theta_inc)
            
            # Plotagem do Mapa de Calor (Contourf)
            fig_focal, ax_focal = plt.subplots(figsize=(8, 5))
            grafico_mapa = ax_focal.contourf(XF, YF, intensidade, 50, cmap='inferno')
            fig_focal.colorbar(grafico_mapa, ax=ax_focal, label="Intensidade Normalizada de Energia")
            
            # Desenha marcações simulando cornetas (Feed Array) no plano
            ax_focal.scatter([0, -0.08, 0.08, 0, 0], [0, 0, 0, -0.08, 0.08], color='cyan', edgecolors='white', s=120, label='Canais de Alimentação', marker='o')
            
            ax_focal.set_title(f"Plano Focal da Parábola (Sinal vindo de {theta_inc}°)")
            ax_focal.set_xlabel("Deslocamento Horizontal Xf (m)")
            ax_focal.set_ylabel("Deslocamento Vertical Yf (m)")
            ax_focal.grid(True, color='gray', linestyle=':', alpha=0.5)
            ax_focal.legend(loc="upper right")
            
            st.pyplot(fig_focal)
# ========================================================
# CONTEÚDO DA ABA 3: SIMULADOR LEO
# ========================================================
with aba3:
    st.subheader("🌌 Cenário Dinâmico: Rastreamento de Satélites de Órbita Baixa (LEO)")
    st.write("Avance o tempo da simulação para assistir ao satélite cruzando o céu e veja a antena refletora ajustando eletronicamente o seu canal focal.")

    # CORREÇÃO DE ESCOPO: Calculamos tudo ANTES de dividir as colunas visuais
    tempo = st.slider("Instante da Órbita (Tempo t)", 0, 100, 50, 1, key="slider_tempo_leo")
    ang_sat, dx_otimo, perda = motor.simular_passagem_satelite(tempo)

    col1_aba3, col2_aba3 = st.columns([1, 3])

    with col1_aba3:
        st.markdown("---")
        st.markdown("#### Status do Rastreamento")
        st.metric("Elevação do Satélite", f"{ang_sat:.1f}°")
        st.metric("Ajuste Focal Necessário (Δx)", f"{dx_otimo*100:.1f} cm")
        st.metric("Atenuação por Varredura", f"{perda:.2f} dB")

    with col2_aba3:
        # Agora dx_otimo está visível e perfeitamente acessível aqui dentro
        XF, YF, intensidade = motor.calcular_campo_focal(ang_sat)
        
        fig_leo, (ax_ceu, ax_foco_din) = plt.subplots(1, 2, figsize=(14, 5.5))
        
        # Gráfico Esquerdo: O satélite no céu
        ax_ceu.axhline(0, color='green', linestyle='-', linewidth=2, label='Horizonte (UFC Quixadá)')
        ax_ceu.scatter([ang_sat], [10], color='gold', s=200, marker='*', label='Satélite LEO')
        ax_ceu.plot([0, ang_sat], [0, 10], color='yellow', linestyle='--', alpha=0.6)
        ax_ceu.set_xlim([-15, 15])
        ax_ceu.set_ylim([-2, 15])
        ax_ceu.set_title("Visão Espacial (Passagem Orbital)")
        ax_ceu.set_xlabel("Ângulo no Céu (Graus)")
        ax_ceu.get_yaxis().set_visible(False)
        ax_ceu.legend(loc="upper right")
        ax_ceu.grid(True, alpha=0.3)

        # Gráfico Direito: O mapa de calor focal dinâmico
        grafico_mapa = ax_foco_din.contourf(XF, YF, intensidade, 40, cmap='inferno')
        
        # Desenha a matriz de cornetas (Feed Array)
        pos_cornetas_x = [0, -0.08, 0.08, 0, 0]
        pos_cornetas_y = [0, 0, 0, -0.08, 0.08]
        ax_foco_din.scatter(pos_cornetas_x, pos_cornetas_y, color='cyan', edgecolors='white', s=150, label='Matriz de Feixes')
        
        # Identifica e destaca a corneta mais próxima do spot de energia
        list_distancias = [float(np.abs(cx - dx_otimo)) for cx in pos_cornetas_x]
        corneta_ativa = int(np.argmin(list_distancias))
        
        ax_foco_din.scatter([pos_cornetas_x[corneta_ativa]], [pos_cornetas_y[corneta_ativa]], 
                            color='lime', edgecolors='black', s=200, label='Canal Ativo (Rastreando!)', marker='s')

        ax_foco_din.set_title("Ajuste Digital do Plano Focal")
        ax_foco_din.set_xlabel("Xf (m)")
        ax_foco_din.set_ylabel("Yf (m)")
        ax_foco_din.legend(loc="upper right")
        
        st.pyplot(fig_leo)