# 🤖 RAG Sales Assistant

Asistente conversacional que responde preguntas en lenguaje natural sobre datos de ventas usando Text-to-SQL y RAG.

## Stack técnico

- **LLaMA 3.3 70B** (vía Groq) — generación de SQL e interpretación
- **ChromaDB** — base vectorial para búsqueda semántica
- **Text-to-SQL** — convierte lenguaje natural en consultas SQL
- **Streamlit** — interfaz web

## Cómo funciona

1. El usuario hace una pregunta en español
2. LLaMA clasifica si es conversacional o sobre datos
3. Si es sobre datos: genera SQL → lo ejecuta sobre el CSV → interpreta resultados
4. Si SQL falla: usa búsqueda semántica de ChromaDB como fallback

## Setup local

Instala dependencias:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

Crea un archivo `.env` con tu API key de Groq:
\`\`\`
GROQ_API_KEY=tu_key_aqui
\`\`\`

Corre la app:
\`\`\`bash
streamlit run app.py
\`\`\`

## Dataset

Superstore Sales Dataset de Kaggle (9,800 filas, 18 columnas).

## Autor

Jhonatan Portilla