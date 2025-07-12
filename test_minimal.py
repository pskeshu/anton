import streamlit as st

st.title("🔬 Minimal Test")
st.write("Hello World")

if st.button("Test"):
    st.write("Button clicked!")

st.write("✅ If you see this, basic Streamlit works")