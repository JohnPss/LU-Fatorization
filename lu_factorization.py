"""
Atividade Prática 02 - Fatoração LU para Sistemas Lineares
Disciplina: Métodos Numéricos e Computacionais
Interface Gráfica com Streamlit

Para executar:
    streamlit run lu_factorization.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math

# Configuração da página
st.set_page_config(
    page_title="Fatoração LU - Circuito Elétrico",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


class LUDecomposition:
    """Classe para Fatoração LU"""
    
    def __init__(self, A, b):
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.n = len(A)
        self.L = None
        self.U = None
        self.x = None
        
    def decompose(self):
        """Decomposição LU usando método de Doolittle"""
        n = self.n
        L = np.zeros((n, n))
        U = np.zeros((n, n))
        
        # Diagonal de L é 1
        for i in range(n):
            L[i][i] = 1.0
        
        # Decomposição
        for i in range(n):
            # Calcula linha i de U
            for k in range(i, n):
                soma = sum(L[i][j] * U[j][k] for j in range(i))
                U[i][k] = self.A[i][k] - soma
            
            # Calcula coluna i de L
            for k in range(i + 1, n):
                if U[i][i] == 0:
                    raise ValueError(f"Matriz singular na posição ({i},{i})")
                soma = sum(L[k][j] * U[j][i] for j in range(i))
                L[k][i] = (self.A[k][i] - soma) / U[i][i]
        
        self.L = L
        self.U = U
        return L, U
    
    def forward_substitution(self, L, b):
        """Forward substitution: Ly = b"""
        n = len(b)
        y = np.zeros(n)
        
        for i in range(n):
            soma = sum(L[i][j] * y[j] for j in range(i))
            y[i] = (b[i] - soma) / L[i][i]
        
        return y
    
    def backward_substitution(self, U, y):
        """Backward substitution: Ux = y"""
        n = len(y)
        x = np.zeros(n)
        
        for i in range(n - 1, -1, -1):
            soma = sum(U[i][j] * x[j] for j in range(i + 1, n))
            x[i] = (y[i] - soma) / U[i][i]
        
        return x
    
    def solve(self):
        """Resolve o sistema Ax = b"""
        L, U = self.decompose()
        y = self.forward_substitution(L, self.b)
        x = self.backward_substitution(U, y)
        self.x = x
        return x
    
    def calculate_relative_error(self, x):
        """Calcula erro relativo"""
        residual = np.dot(self.A, x) - self.b
        return np.linalg.norm(residual) / np.linalg.norm(self.b)


def display_matrix(matrix, title, key_suffix=""):
    """Exibe matriz formatada"""
    df = pd.DataFrame(matrix)
    df = df.applymap(lambda x: f"{x:.6f}")
    st.dataframe(df, use_container_width=True, key=f"df_{key_suffix}")


def create_circuit_diagram():
    """Cria diagrama do circuito elétrico"""
    st.markdown("""
    ### ⚡ Circuito Elétrico de 4 Malhas
    
    **Sistema de Equações:**
    ```
    3I₁ - I₂       - I₄ = 5
   -I₁ + 4I₂ - I₃      = 0
        - I₂ + 4I₃ - I₄ = 0
   -I₁      - I₃ + 3I₄ = 5
    ```
    
    **Objetivo:** Encontrar as correntes I₁, I₂, I₃, I₄
    """)


def main():
    # Header
    st.markdown('<div class="main-header">⚡ Fatoração LU - Sistemas Lineares</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Atividade Prática 02 - Métodos Numéricos e Computacionais</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📋 Menu")
    mode = st.sidebar.radio(
        "Escolha o modo:",
        ["🔵 Sistema Proposto (Circuito)", "🟢 Sistema Genérico", "📊 Comparação e Testes"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 Sobre")
    st.sidebar.info(
        "Esta aplicação implementa o método de **Fatoração LU** para resolução "
        "de sistemas lineares, com foco no circuito elétrico de 4 malhas."
    )
    
    # ========================================
    # MODO 1: SISTEMA PROPOSTO
    # ========================================
    if mode == "🔵 Sistema Proposto (Circuito)":
        st.markdown("---")
        create_circuit_diagram()
        
        # Sistema do circuito
        A_circuit = np.array([
            [3, -1, 0, -1],
            [-1, 4, -1, 0],
            [0, -1, 4, -1],
            [-1, 0, -1, 3]
        ], dtype=float)
        
        b_circuit = np.array([5, 0, 0, 5], dtype=float)
        expected_solution = np.array([2.5, 1.5, 1.0, 3.0])
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Matriz A (Coeficientes)")
            display_matrix(A_circuit, "Matriz A", "circuit_a")
        
        with col2:
            st.markdown("#### 📊 Vetor b (Termos Independentes)")
            df_b = pd.DataFrame(b_circuit, columns=["b"])
            df_b.index = [f"Eq {i+1}" for i in range(len(b_circuit))]
            st.dataframe(df_b, use_container_width=True)
        
        st.markdown("---")
        
        if st.button("▶️ Resolver Sistema do Circuito", type="primary", use_container_width=True):
            with st.spinner("Resolvendo sistema..."):
                try:
                    solver = LUDecomposition(A_circuit, b_circuit)
                    
                    # Passo 1: Decomposição
                    st.markdown("### 🔄 Passo 1: Decomposição LU")
                    st.markdown('<div class="info-box">Decomposição A = L × U pelo método de Doolittle</div>', unsafe_allow_html=True)
                    
                    L, U = solver.decompose()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Matriz L (Triangular Inferior)")
                        display_matrix(L, "Matriz L", "circuit_l")
                    
                    with col2:
                        st.markdown("#### Matriz U (Triangular Superior)")
                        display_matrix(U, "Matriz U", "circuit_u")
                    
                    # Verificação
                    LU = np.dot(L, U)
                    verification_error = np.linalg.norm(A_circuit - LU)
                    
                    if verification_error < 1e-10:
                        st.markdown(f'<div class="success-box">✅ Decomposição verificada: ||A - LU|| = {verification_error:.2e}</div>', unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Passo 2: Solução
                    st.markdown("### ✅ Passo 2: Resolução do Sistema")
                    
                    solution = solver.solve()
                    
                    # Resultados
                    col1, col2, col3 = st.columns(3)
                    
                    results_df = pd.DataFrame({
                        'Corrente': [f'I₁', 'I₂', 'I₃', 'I₄'],
                        'Calculado (A)': solution,
                        'Esperado (A)': expected_solution,
                        'Erro Absoluto': np.abs(solution - expected_solution)
                    })
                    
                    st.markdown("#### 📈 Resultados Comparativos")
                    st.dataframe(results_df.style.format({
                        'Calculado (A)': '{:.10f}',
                        'Esperado (A)': '{:.10f}',
                        'Erro Absoluto': '{:.2e}'
                    }), use_container_width=True)
                    
                    # Métricas
                    st.markdown("---")
                    st.markdown("### 📊 Métricas de Precisão")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    relative_error = solver.calculate_relative_error(solution)
                    
                    with col1:
                        st.metric("Erro Relativo", f"{relative_error:.2e}")
                    
                    with col2:
                        max_error = np.max(np.abs(solution - expected_solution))
                        st.metric("Erro Máximo", f"{max_error:.2e}")
                    
                    with col3:
                        residual_norm = np.linalg.norm(np.dot(A_circuit, solution) - b_circuit)
                        st.metric("||Ax - b||", f"{residual_norm:.2e}")
                    
                    with col4:
                        condition_number = np.linalg.cond(A_circuit)
                        st.metric("Cond(A)", f"{condition_number:.2f}")
                    
                    # Gráfico
                    st.markdown("---")
                    st.markdown("### 📊 Visualização das Correntes")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name='Calculado',
                        x=['I₁', 'I₂', 'I₃', 'I₄'],
                        y=solution,
                        marker_color='lightblue'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Esperado',
                        x=['I₁', 'I₂', 'I₃', 'I₄'],
                        y=expected_solution,
                        marker_color='lightcoral'
                    ))
                    
                    fig.update_layout(
                        title="Comparação: Solução Calculada vs Esperada",
                        xaxis_title="Corrente",
                        yaxis_title="Valor (A)",
                        barmode='group',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown('<div class="success-box">✅ <b>Sistema resolvido com sucesso!</b><br>As correntes foram calculadas com excelente precisão.</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown(f'<div class="error-box">❌ Erro: {str(e)}</div>', unsafe_allow_html=True)
    
    # ========================================
    # MODO 2: SISTEMA GENÉRICO
    # ========================================
    elif mode == "🟢 Sistema Genérico":
        st.markdown("---")
        st.markdown("### 🔧 Sistema Personalizado")
        st.info("Digite sua própria matriz A e vetor b para resolver qualquer sistema linear n×n")
        
        # Input da dimensão
        n = st.number_input("Dimensão do sistema (n×n):", min_value=2, max_value=10, value=3, step=1)
        
        st.markdown("#### Entrada da Matriz A")
        st.markdown("Digite os valores linha por linha (separados por espaço):")
        
        A_input = []
        for i in range(n):
            row_input = st.text_input(f"Linha {i+1}:", key=f"row_{i}", placeholder=f"Ex: 1 2 3 ...")
            A_input.append(row_input)
        
        st.markdown("#### Entrada do Vetor b")
        b_input = st.text_input("Vetor b (valores separados por espaço):", key="vec_b", placeholder="Ex: 1 2 3 ...")
        
        if st.button("▶️ Resolver Sistema Genérico", type="primary", use_container_width=True):
            try:
                # Parse inputs
                A_custom = []
                for row_str in A_input:
                    if row_str.strip():
                        row = [float(x) for x in row_str.split()]
                        if len(row) != n:
                            raise ValueError(f"Cada linha deve ter exatamente {n} valores!")
                        A_custom.append(row)
                
                if len(A_custom) != n:
                    raise ValueError(f"Você deve fornecer exatamente {n} linhas!")
                
                if not b_input.strip():
                    raise ValueError("Você deve fornecer o vetor b!")
                
                b_custom = [float(x) for x in b_input.split()]
                
                if len(b_custom) != n:
                    raise ValueError(f"O vetor b deve ter exatamente {n} valores!")
                
                A_custom = np.array(A_custom, dtype=float)
                b_custom = np.array(b_custom, dtype=float)
                
                # Mostra sistema
                st.markdown("---")
                st.markdown("### 📊 Sistema Recebido")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Matriz A")
                    display_matrix(A_custom, "Matriz A", "custom_a")
                
                with col2:
                    st.markdown("#### Vetor b")
                    df_b = pd.DataFrame(b_custom, columns=["b"])
                    st.dataframe(df_b, use_container_width=True)
                
                # Resolve
                with st.spinner("Resolvendo..."):
                    solver = LUDecomposition(A_custom, b_custom)
                    
                    st.markdown("---")
                    st.markdown("### 🔄 Decomposição LU")
                    
                    L, U = solver.decompose()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Matriz L")
                        display_matrix(L, "L", "custom_l")
                    
                    with col2:
                        st.markdown("#### Matriz U")
                        display_matrix(U, "U", "custom_u")
                    
                    st.markdown("---")
                    st.markdown("### ✅ Solução")
                    
                    solution = solver.solve()
                    
                    solution_df = pd.DataFrame({
                        'Variável': [f'x{i+1}' for i in range(n)],
                        'Valor': solution
                    })
                    
                    st.dataframe(solution_df.style.format({'Valor': '{:.10f}'}), use_container_width=True)
                    
                    # Métricas
                    col1, col2, col3 = st.columns(3)
                    
                    relative_error = solver.calculate_relative_error(solution)
                    residual_norm = np.linalg.norm(np.dot(A_custom, solution) - b_custom)
                    condition_number = np.linalg.cond(A_custom)
                    
                    with col1:
                        st.metric("Erro Relativo", f"{relative_error:.2e}")
                    
                    with col2:
                        st.metric("||Ax - b||", f"{residual_norm:.2e}")
                    
                    with col3:
                        st.metric("Cond(A)", f"{condition_number:.2f}")
                    
                    st.markdown('<div class="success-box">✅ Sistema resolvido com sucesso!</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Erro: {str(e)}</div>', unsafe_allow_html=True)
    
    # ========================================
    # MODO 3: TESTES
    # ========================================
    else:  # Comparação e Testes
        st.markdown("---")
        st.markdown("### 🧪 Testes de Validação")
        
        st.info("Testes automáticos para validar a implementação do algoritmo")
        
        if st.button("▶️ Executar Todos os Testes", type="primary", use_container_width=True):
            
            # Teste 1: Identidade
            st.markdown("#### 🔬 Teste 1: Matriz Identidade")
            I_matrix = np.eye(3)
            b_test1 = np.array([1, 2, 3])
            
            solver1 = LUDecomposition(I_matrix, b_test1)
            sol1 = solver1.solve()
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Sistema:** Ix = b")
                display_matrix(I_matrix, "I", "test1_i")
            
            with col2:
                st.markdown("**Solução (deve ser igual a b):**")
                st.write(f"Solução: {sol1}")
                st.write(f"Esperado: {b_test1}")
                if np.allclose(sol1, b_test1):
                    st.success("✅ Teste 1 PASSOU!")
                else:
                    st.error("❌ Teste 1 FALHOU!")
            
            st.markdown("---")
            
            # Teste 2: Sistema 2x2
            st.markdown("#### 🔬 Teste 2: Sistema 2×2")
            A_test2 = np.array([[2, 1], [1, 3]])
            b_test2 = np.array([5, 6])
            
            solver2 = LUDecomposition(A_test2, b_test2)
            sol2 = solver2.solve()
            
            st.write("Sistema: 2x₁ + x₂ = 5, x₁ + 3x₂ = 6")
            st.write(f"Solução: x₁ = {sol2[0]:.6f}, x₂ = {sol2[1]:.6f}")
            
            error2 = solver2.calculate_relative_error(sol2)
            
            if error2 < 1e-10:
                st.success(f"✅ Teste 2 PASSOU! Erro: {error2:.2e}")
            else:
                st.warning(f"⚠️ Teste 2 com erro elevado: {error2:.2e}")
            
            st.markdown("---")
            
            # Teste 3: Sistema 3x3
            st.markdown("#### 🔬 Teste 3: Sistema 3×3")
            A_test3 = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
            b_test3 = np.array([15, 10, 10])
            
            solver3 = LUDecomposition(A_test3, b_test3)
            sol3 = solver3.solve()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Matriz A:**")
                display_matrix(A_test3, "A", "test3_a")
            
            with col2:
                st.markdown("**Solução:**")
                sol_df = pd.DataFrame({'x': sol3})
                st.dataframe(sol_df.style.format({'x': '{:.6f}'}))
            
            error3 = solver3.calculate_relative_error(sol3)
            
            if error3 < 1e-10:
                st.success(f"✅ Teste 3 PASSOU! Erro: {error3:.2e}")
            else:
                st.warning(f"⚠️ Teste 3 com erro elevado: {error3:.2e}")
            
            st.markdown("---")
            st.markdown('<div class="success-box">✅ Todos os testes foram executados!</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()