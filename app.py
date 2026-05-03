import streamlit as st
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import os
import pandas as pd
import sqlite3

# ════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ════════════════════════════════════════════════════════

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

df = pd.read_csv("train.csv")


# ════════════════════════════════════════════════════════
# 2. RECURSOS CACHEADOS (se inicializan una sola vez)
# ════════════════════════════════════════════════════════

@st.cache_resource
def get_db_connection():
    """Crea base SQLite en memoria con los datos. Solo se ejecuta una vez."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    df_copy = df.copy()
    df_copy.columns = [c.replace(" ", "_").replace("-", "_") for c in df_copy.columns]
    df_copy.to_sql("ventas", conn, index=False, if_exists="replace")
    return conn


@st.cache_resource
def get_chroma_collection():
    """Carga la base vectorial ChromaDB. Solo se ejecuta una vez."""
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    cliente_db = chromadb.PersistentClient(path="./mi_base_vectorial")
    return cliente_db.get_or_create_collection(
        name="mis_documentos",
        embedding_function=embedding_fn
    )


# ════════════════════════════════════════════════════════
# 3. ESQUEMA DE LA TABLA (para el prompt SQL)
# ════════════════════════════════════════════════════════

ESQUEMA = """
Tabla: ventas
Columnas:
- Row_ID (int)
- Order_ID, Order_Date, Ship_Date (texto)
- Ship_Mode: 'Second Class', 'Standard Class', 'First Class', 'Same Day'
- Customer_ID, Customer_Name (texto)
- Segment: 'Consumer', 'Corporate', 'Home Office'
- Country, City, State, Postal_Code (texto)
- Region: 'South', 'West', 'Central', 'East'
- Product_ID, Product_Name (texto)
- Category: 'Furniture', 'Office Supplies', 'Technology'
- Sub_Category (texto)
- Sales (float): monto de la venta en dólares
"""


# ════════════════════════════════════════════════════════
# 4. FUNCIONES AUXILIARES (cada una hace UNA cosa)
# ════════════════════════════════════════════════════════

def llamar_llm(prompt: str, system: str = None) -> str:
    """Llama a LLaMA 3 con un prompt simple."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    return response.choices[0].message.content


def es_pregunta_de_datos(pregunta: str) -> bool:
    """Decide si la pregunta es sobre datos o conversacional."""
    clasificacion = llamar_llm(
        f"Clasifica esta pregunta. Responde SOLO una palabra:\n"
        f"- DATOS si pregunta sobre ventas, productos, clientes, recomendaciones de negocio\n"
        f"- CHAT si es saludo, conversación o pregunta personal\n\n"
        f"Pregunta: {pregunta}"
    )
    return "DATOS" in clasificacion.upper()


def generar_sql(pregunta: str) -> str:
    prompt = f"""Eres un experto en SQL para análisis de ventas retail.

TABLA: ventas
{ESQUEMA}

Convierte la pregunta en SQL válido para sqlite3 siguiendo estos PATRONES:

PATRÓN 1 — "dónde / qué región / qué estado vendió más":
SELECT Region, SUM(Sales) as Total_Ventas, COUNT(*) as Ordenes
FROM ventas GROUP BY Region ORDER BY Total_Ventas DESC;

PATRÓN 2 — "qué categoría / producto vendió más":
SELECT Category, SUM(Sales) as Total_Ventas
FROM ventas GROUP BY Category ORDER BY Total_Ventas DESC;

PATRÓN 3 — "quién es el mejor cliente / quién compró más":
SELECT Customer_Name, SUM(Sales) as Total_Gastado, COUNT(*) as Compras
FROM ventas GROUP BY Customer_Name ORDER BY Total_Gastado DESC LIMIT 10;

PATRÓN 4 — "cómo aumentar ventas / vender más / crecer":
SELECT Category, Region, SUM(Sales) as Total_Ventas, COUNT(*) as Ordenes
FROM ventas GROUP BY Category, Region ORDER BY Total_Ventas DESC LIMIT 12;

PATRÓN 5 — pregunta sobre un cliente específico:
SELECT Customer_Name, Product_Name, Category, Sales
FROM ventas WHERE LOWER(Customer_Name) LIKE '%nombre%' LIMIT 50;

PATRÓN 6 — pregunta sobre un estado/región/categoría específica:
SELECT * FROM ventas WHERE LOWER(State) LIKE '%texas%' LIMIT 50;

REGLAS ESTRICTAS:
- Usa SOLO las columnas del esquema
- NUNCA agrupes por más de 2 dimensiones
- SIEMPRE usa LIMIT cuando agrupes (máximo 20 filas)
- Para nombres usa LIKE '%texto%' con LOWER()
- Responde SOLO con el SQL, sin explicaciones, sin markdown

PREGUNTA: {pregunta}
SQL:"""

    sql = llamar_llm(prompt).strip()
    return sql.replace("```sql", "").replace("```", "").strip()






def ejecutar_sql(sql: str) -> str:
    conn = get_db_connection()
    try:
        resultado = pd.read_sql_query(sql, conn)
        if resultado.empty:
            return "VACÍO"
        # limitar a 50 filas máximo para no exceder tokens
        if len(resultado) > 50:
            resultado = resultado.head(50)
        return resultado.to_string(index=False)
    except Exception as e:
        return f"ERROR_SQL: {e}"




def buscar_semantico(pregunta: str) -> str:
    """Fallback: busca con ChromaDB cuando el SQL no encuentra nada."""
    coleccion = get_chroma_collection()
    resultados = coleccion.query(query_texts=[pregunta], n_results=5)
    return "\n".join([f"- {f}" for f in resultados["documents"][0]])


def interpretar_resultados(pregunta: str, sql: str, datos: str) -> str:
    prompt = f"""Eres un analista de datos senior especializado en retail.

    # REGLAS DE PRECISIÓN (no negociables)
    - TODO número que cites debe aparecer LITERALMENTE en los RESULTADOS de abajo
    - Formato monetario: siempre $X,XXX.XX (con coma de miles y dos decimales)
    - Si un dato no está en los resultados → di "no está en los datos consultados"
    - NUNCA estimes, redondees a millones, ni sumes mentalmente
    - NUNCA inventes columnas o filtros que no estén en el SQL ejecutado

    # ESTRUCTURA DE RESPUESTA
    **1. Respuesta directa** (1-2 oraciones)
    Responde la pregunta literal con el número exacto.

    **2. Contexto del análisis**
    Si el SQL aplicó filtros o agrupaciones, explícalo en una línea: "Para esto agrupé por X / filtré por Y porque..."

    **3. Hallazgos clave** (solo si aportan)
    2-3 bullets con los datos más relevantes — todos citados de los resultados.

    **4. Recomendaciones accionables** (solo si la pregunta lo pide)
    Máximo 3 recomendaciones. Cada una debe:
    - Empezar con un verbo (Aumentar, Investigar, Reducir, Probar)
    - Estar atada a un dato real de los resultados
    - Ser específica, no genérica

    # QUÉ EVITAR
    - Frases como "se podría considerar", "tal vez", "es posible que"
    - Recomendaciones genéricas tipo "mejorar marketing" sin datos detrás
    - Repetir la pregunta del usuario
    - Saludar o despedirte
    - Decir "espero que esto ayude"

    # IDIOMA Y EXTENSIÓN
    - Español neutro, profesional pero conversacional
    - Máximo 250 palabras
    - Sin emojis salvo en bullets si aporta claridad

    ---
    PREGUNTA: {pregunta}

    SQL EJECUTADO:
    {sql}

    RESULTADOS (única fuente de verdad):
    {datos}
    ---

    Respuesta:"""
    return llamar_llm(prompt)


# ════════════════════════════════════════════════════════
# 5. ORQUESTADOR PRINCIPAL
# ════════════════════════════════════════════════════════

def rag_responder(pregunta: str) -> str:
    try:
        # si es chat, responder directo
        if not es_pregunta_de_datos(pregunta):
            return llamar_llm(
                pregunta,
                system="Eres un asistente de análisis de ventas retail amigable. Responde en español de forma breve y natural."
            )

        # si es de datos: SQL → ejecutar → interpretar
        sql = generar_sql(pregunta)
        datos = ejecutar_sql(sql)

        # si SQL no devuelve nada o falla, usar búsqueda semántica
        if datos in ("VACÍO",) or datos.startswith("ERROR_SQL"):
            datos = "Búsqueda semántica relevante:\n" + buscar_semantico(pregunta)

        return interpretar_resultados(pregunta, sql, datos)

    except Exception as e:
        if "429" in str(e):
            return "⚠️ Límite alcanzado. Intenta en unos minutos."
        return f"Error: {str(e)}"


# ════════════════════════════════════════════════════════
# 6. INTERFAZ STREAMLIT
# ════════════════════════════════════════════════════════

st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="centered")
st.title("🤖 RAG Assistant")
st.caption("Hazle preguntas a tus datos — powered by LLaMA 3 + Text-to-SQL + ChromaDB")
st.divider()

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.write(msg["contenido"])

pregunta = st.chat_input("Escribe tu pregunta aquí...")

if pregunta:
    with st.chat_message("user"):
        st.write(pregunta)
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})

    with st.chat_message("assistant"):
        with st.spinner("Analizando datos..."):
            respuesta = rag_responder(pregunta)
        st.write(respuesta)
    st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta})