import streamlit as st
import json
import requests

# Titulo da aplicação
st.title('Modelo de Predição de Salário')

# Input do Usuário
st.write("Quantos meses o profissional está na empresa?")
tempo_na_empresa = st.slider("Meses", min_value=1,max_value=120, value=60, step=1)

st.write("Qual o nivel do profissional na empresa?")
nivel_na_empresa = st.slider("Nivel", min_value=1,max_value=10, value=5, step=1)

# Praparar os dados para enviar para a API
input_features = {
    'tempo_na_empresa': tempo_na_empresa,
    'nivel_na_empresa': nivel_na_empresa
}

# Botão de Predição
if st.button('Estimar Salário'):
    # Enviar os dados para a API
    response = requests.post(url = 'http://127.0.0.1:8000/predict', data=json.dumps(input_features))

    res_json = json.loads(response.text)
    salario_em_reais = round(res_json['salario_em_reais'], 2)
    st.subheader(f'O salário estimado é de R$ {salario_em_reais}')