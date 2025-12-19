# Fatoração LU - Sistemas Lineares  

Este projeto implementa o método de Fatoração LU para resolução de sistemas lineares, com uma interface gráfica interativa desenvolvida em Streamlit. O foco principal é a resolução de um circuito elétrico de 4 malhas, mas também permite resolver sistemas genéricos.

## Pré-requisitos

- Python 3.8 ou superior instalado.

## Instalação e Execução

### Linux / macOS

1.  **Abra o terminal** na pasta do projeto.

2.  **Crie um ambiente virtual** (recomendado):
    ```bash
    python3 -m venv venv
    ```

3.  **Ative o ambiente virtual**:
    ```bash
    source venv/bin/activate
    ```

4.  **Instale as dependências**:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Execute a aplicação**:
    ```bash
    streamlit run lu_factorization.py
    ```

### Windows

1.  **Abra o Prompt de Comando (cmd) ou PowerShell** na pasta do projeto.

2.  **Crie um ambiente virtual** (recomendado):
    ```cmd
    python -m venv venv
    ```

3.  **Ative o ambiente virtual**:
    -   No **Command Prompt (cmd)**:
        ```cmd
        venv\Scripts\activate
        ```
    -   No **PowerShell**:
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
        *(Se houver erro de permissão no PowerShell, execute `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` antes)*

4.  **Instale as dependências**:
    ```cmd
    pip install -r requirements.txt
    ```

5.  **Execute a aplicação**:
    ```cmd
    streamlit run lu_factorization.py
    ```

---

## Tutorial de Uso Geral

Ao iniciar a aplicação, você verá um menu lateral com três modos de operação:

### 1. 🔵 Sistema Proposto (Circuito)
Este modo resolve o problema específico do circuito elétrico de 4 malhas proposto na atividade.
-   Visualize o diagrama do circuito e o sistema de equações.
-   Clique em **"▶️ Resolver Sistema do Circuito"**.
-   O sistema exibirá:
    -   A decomposição LU (Matrizes L e U).
    -   As correntes calculadas (I₁, I₂, I₃, I₄).
    -   Comparação com os valores esperados e métricas de erro.

### 2. 🟢 Sistema Genérico
Permite resolver qualquer sistema linear quadrado (n×n).
-   Defina a dimensão do sistema (ex: 3 para um sistema 3×3).
-   Insira os valores da **Matriz A** linha por linha (valores separados por espaço).
-   Insira os valores do **Vetor b** (separados por espaço).
-   Clique em **"▶️ Resolver Sistema Genérico"** para ver a solução passo a passo.

### 3. 📊 Comparação e Testes
Executa uma bateria de testes automáticos para validar a implementação.
-   Clique em **"▶️ Executar Todos os Testes"**.
-   O sistema verificará casos básicos (Matriz Identidade, sistemas 2x2 e 3x3) e informará se a implementação está correta.
