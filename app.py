import streamlit as st
from agent.multi_agent import run_multi_agent
from utils.log_loader import load_logs

st.title("📊 대출 로그 AI 분석기")

logs = load_logs()

selected_log = st.selectbox("로그 선택", range(len(logs)))

if st.button("분석"):
    result = run_multi_agent(logs[selected_log])
    st.write(result)