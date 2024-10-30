import streamlit as st

# Estilo de fondo
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background:
radial-gradient(black 15%, transparent 16%) 0 0,
radial-gradient(black 15%, transparent 16%) 8px 8px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 0 1px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 8px 9px;
background-color:#282828;
background-size:16px 16px;
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


# Título de la aplicación
st.title("Modelos de Regresión")

# Información adicional
st.header("Materia: Aprendizaje Máquina I")
st.subheader("Desarrollado por: Javier Horacio Pérez Ricárdez")
st.subheader("Catedrático: Félix Orlando Martínez")

# Agregar contenido adicional según sea necesario
st.write("Esta aplicación permite explorar diferentes modelos de regresión.")
