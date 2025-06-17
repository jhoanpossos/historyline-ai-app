# -*- coding: utf-8 -*-
# historyline_app.py
# Author: Jhoan (with Gemini's collaboration)
# Project: HistoryLine AI for TII Abu Dhabi CrowdLabel Challenge
# Version: 2.19 (API Key & SpaCy model handling refactor)
# Contextualization (new feature)

import streamlit as st
import os
import re
import time
import numpy as np
import spacy
# Importaciones necesarias para la descarga de spaCy
from spacy.util import is_package
from spacy.cli import download
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from dotenv import load_dotenv # Import load_dotenv
import requests
from PIL import Image
import io
import json
import urllib3
import pandas as pd
import datetime
from typing import Union # Added Union import

# ==============================================================================
# Seguridad: Deshabilitar advertencias de SSL para desarrollo.
# ¡IMPORTANTE! NUNCA USAR EN PRODUCCIÓN. Representa un riesgo de seguridad.
# Asegúrate de que las URLs de tus APIs usen HTTPS y tengan certificados válidos.
# ==============================================================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==============================================================================
# 0. LANGUAGE DICTIONARY AND LOCALIZED TEXTS
# ==============================================================================

LANG_DICT = {
    "es": {
        "page_title": "HistoryLine IA",
        "loading_models": "Cargando modelos de IA (esto solo debería pasar una vez por sesión)...",
        "models_loaded": "✅ Modelos y configuraciones cargados.",
        "error_loading_models": "Error cargando modelos de NLP o NLTK:",
        "warning_no_google_api_key": "ADVERTENCIA: No se encontró la GOOGLE_API_KEY. El chat y la funcionalidad VQA no funcionarán.",
        "warning_no_tii_api_key": "ADVERTENCIA: La TII_API_KEY no está configurada. La funcionalidad VQA del TII no funcionará.",
        "app_cannot_continue": "La aplicación no puede continuar sin los modelos base. Revisa la configuración y las variables de entorno.",
        "my_chats_header": "Mis Chats",
        "get_tii_vqa_button": "🤖 Obtener Tarea VQA del TII",
        "vqa_warning_no_google_api": "No se puede obtener tarea VQA: GOOGLE_API_KEY no configurada.",
        "vqa_warning_no_tii_api": "No se puede obtener tarea VQA: TII_API_KEY no configurada.",
        "vqa_spinner_text": "Obteniendo tarea VQA de la API del TII y procesando con Gemini Vision...",
        "vqa_toast_success": "¡Nueva tarea VQA añadida y analizada!",
        "vqa_error_valid_task": "No se pudo obtener una tarea VQA válida. Consulta la consola para más detalles.",
        "vqa_toast_no_task": "No hay tareas VQA disponibles en este momento desde la API del TII.",
        "vqa_toast_invalid_format": "La tarea VQA obtenida no tiene el formato esperado (revisa 'content.image.url' o 'task.text').",
        "new_chat_button": "➕ Nuevo Chat",
        "chat_title_prefix": "🤖 HistoryLine:",
        "chat_input_placeholder": "Escribe tu mensaje...",
        "thinking_spinner_text": "Pensando...",
        "reply_button_tooltip": "Responder a este mensaje",
        "responding_to": "Respondiendo a:",
        "historyline_header": "HistoryLine ✨",
        "update_historyline_button": "🔄 Actualizar Vista HistoryLine",
        "historyline_updated_toast": "Vista HistoryLine actualizada. El análisis se realiza por mensaje.",
        "empty_chat_caption": "Escribe mensajes para iniciar el análisis HistoryLine.",
        "topic_prefix": "Tópico",
        "topic_unknown_title": "Título Desconocido",
        "revisiting_topic": "(Retomando Tópico",
        "pending_title": "Título Pendiente",
        "prompt_summary_prefix": "   -> \"",
        "subtopic_prefix": "  Nuevo Subtópico",
        "detail_prefix": "    -> \"",
        "error_gemini_api": "Lo siento, tuve un problema al contactar la API de Gemini.",
        "error_gemini_vqa": "Error al procesar la VQA después de varios reintentos.",
        "error_gemini_title": "Tópico (Error IA)",
        "error_api_timeout": "Error: Tiempo de espera agotado al descargar imagen/procesar VQA.",
        "error_request_exception": "Error al descargar la imagen VQA/contactar Gemini Vision.",
        "error_api_generic": "Error en la API de Gemini Vision al procesar la VQA.",
        "error_max_retries_vqa_task": "Se excedió el número máximo de reintentos para obtener tarea VQA.",
        "error_max_retries_vqa_gemini": "Se excedió el número máximo de reintentos para VQA con Gemini.",
        "error_max_retries_gemini_chat": "Se excedió el número máximo de reintentos para generar respuesta.",
        "error_max_retries_gemini_title": "Se excedió el número máximo de reintentos para generar el título.",
        "chat_file_loaded": "'chat_ia.txt' cargado como 'Chat de Archivo (ia)'.",
        "generating_initial_analysis": "Generando análisis inicial para ",
        "initial_analysis_completed": "✅ Análisis inicial de ",
        "no_chat_file_found": "No se encontró 'chat_ia.txt'. Creando chat de ejemplo.",
        "new_main_topic_detected": "🚨 NUEVO TÓPICO PRINCIPAL:",
        "new_main_topic_unrelated_detected": "🚨 NUEVO TÓPICO PRINCIPAL (No relacionado):",
        "continuation_of_detail": "➡️ Continuación de DETALLE",
        "new_sub_subtopic": "↪️ Nuevo SUB-SUBTÓPICO",
        "new_subtopic": "➡️ Nuevo SUBTÓPICO",
        "revisiting_topic_log": "🔄 RETOMANDO TÓPICO:",
        "vqa_specific_title_generated": "✨ VQA detectada. Specific title generated:",
        "language_selector_label": "Select Language",
        "export_replies_button": "💾 Exportar Respuestas a CSV",
        "replies_csv_filename": "respuestas_historyline.csv",
        "preparing_response_toast": "Preparando su respuesta...",
        "updating_history_toast": "Actualizando vista del historial..."
    },
    "en": {
        "page_title": "HistoryLine AI",
        "loading_models": "Loading AI models (this should only happen once per session)...",
        "models_loaded": "✅ Models and configurations loaded.",
        "error_loading_models": "Error loading NLP or NLTK models:",
        "warning_no_google_api_key": "WARNING: GOOGLE_API_KEY not found. Chat and VQA functionality will not work.",
        "warning_no_tii_api_key": "WARNING: TII_API_KEY is not configured. TII VQA functionality will not work.",
        "app_cannot_continue": "The application cannot continue without base models. Check configuration and environment variables.",
        "my_chats_header": "My Chats",
        "get_tii_vqa_button": "🤖 Get TII VQA Task",
        "vqa_warning_no_google_api": "Cannot get VQA task: GOOGLE_API_KEY not configured.",
        "vqa_warning_no_tii_api": "Cannot get VQA task: TII_API_KEY not configured.",
        "vqa_spinner_text": "Getting VQA task from TII API and processing with Gemini Vision...",
        "vqa_toast_success": "New VQA task added and analyzed!",
        "vqa_error_valid_task": "Could not get a valid VQA task. Check console for details.",
        "vqa_toast_no_task": "No VQA tasks available at the moment from TII API.",
        "vqa_toast_invalid_format": "The VQA task obtained does not have the expected format (check 'content.image.url' or 'task.text').",
        "new_chat_button": "➕ New Chat",
        "chat_title_prefix": "🤖 HistoryLine:",
        "chat_input_placeholder": "Type your message...",
        "thinking_spinner_text": "Thinking...",
        "reply_button_tooltip": "Reply to this message",
        "responding_to": "Replying to:",
        "historyline_header": "HistoryLine ✨",
        "update_historyline_button": "🔄 Update HistoryLine View",
        "historyline_updated_toast": "HistoryLine view updated. Analysis is performed per message.",
        "empty_chat_caption": "Type messages to start HistoryLine analysis.",
        "topic_prefix": "Topic",
        "topic_unknown_title": "Unknown Title",
        "revisiting_topic": "(Revisiting Topic",
        "pending_title": "Pending Title",
        "prompt_summary_prefix": "   -> \"",
        "subtopic_prefix": "  New Subtopic",
        "detail_prefix": "    -> \"",
        "error_gemini_api": "Sorry, I had a problem contacting the Gemini API.",
        "error_gemini_vqa": "Error processing VQA after several retries.",
        "error_gemini_title": "Topic (AI Error)",
        "error_api_timeout": "Error: Timeout when downloading image/processing VQA.",
        "error_request_exception": "Error downloading VQA image/contacting Gemini Vision.",
        "error_api_generic": "Error in Gemini Vision API when processing VQA.",
        "error_max_retries_vqa_task": "Maximum number of retries exceeded for VQA task.",
        "error_max_retries_vqa_gemini": "Maximum number of retries exceeded for VQA with Gemini.",
        "error_max_retries_gemini_chat": "Maximum number of retries exceeded for generating response.",
        "error_max_retries_gemini_title": "Maximum number of retries exceeded for generating title.",
        "chat_file_loaded": "'chat_ia.txt' loaded as 'Chat File (ia)'.",
        "generating_initial_analysis": "Generating initial analysis for ",
        "initial_analysis_completed": "✅ Initial analysis of ",
        "no_chat_file_found": "Could not find 'chat_ia.txt'. Creating example chat.",
        "new_main_topic_detected": "🚨 NEW MAIN TOPIC:",
        "new_main_topic_unrelated_detected": "🚨 NEW MAIN TOPIC (Unrelated):",
        "continuation_of_detail": "➡️ Continuation of DETAIL",
        "new_sub_subtopic": "↪️ New SUB-SUBTOPIC",
        "new_subtopic": "➡️ New SUBTOPIC",
        "revisiting_topic_log": "🔄 REVISITING TOPIC:",
        "vqa_specific_title_generated": "✨ VQA detected. Specific title generated:",
        "language_selector_label": "Select Language",
        "export_replies_button": "💾 Export Replies to CSV",
        "replies_csv_filename": "historyline_replies.csv",
        "preparing_response_toast": "Preparing your reply...",
        "updating_history_toast": "Updating history view..."
    }
}

# ==============================================================================
# 1. CONFIGURATION AND MODEL LOADING (STREAMLIT CACHED)
# ==============================================================================
# Initialize language in st.session_state before any use of T
if "language" not in st.session_state:
    st.session_state.language = "es" # Default language
T = LANG_DICT[st.session_state.language] # Access to localized text dictionary

st.set_page_config(layout="wide", page_title=T["page_title"])

@st.cache_resource
def load_models_and_config():
    """
    Loads all models and configurations only once.
    Streamlit will cache these objects for optimal performance.
    """
    print(f"[{time.strftime('%H:%M:%S')}] {T['loading_models']}")
    
    # Cargar variables de entorno desde .env
    # Se recomienda usar os.environ.get() para acceder a ellas de forma segura.
    load_dotenv()

    # --- Calibrated Parameters ---
    config = {
        "THRESHOLD_DETAIL": 0.75,
        "THRESHOLD_SUBTOPIC": 0.55,
        "THRESHOLD_NEW_TOPIC": 0.40,
        "REVISIT_THRESHOLD": 0.35,
        "MIN_PROMPT_WEIGHT": 0.60,
        "MAX_PROMPT_WEIGHT": 0.95,
        # Parameters for general Exponential Backoff
        "INITIAL_RETRY_DELAY": 2,
        "MAX_RETRY_ATTEMPTS": 5,
        "RETRY_MULTIPLIER": 2,
        "TII_BASE_URL": "https://crowdlabel.tii.ae/api/2025.2/tasks/pick", # Endpoint corrected as per Sultan
        "REQUEST_TIMEOUT": 10, # Timeout for HTTP requests
        # New parameters for contextualization
        "CONTEXT_HIERARCHY_MAX_ITEMS": 5, # Max number of recent hierarchy items to include
        "CONTEXT_CHAT_MAX_EXCHANGES": 2, # Max number of recent user/assistant exchanges to include
        "CONTEXT_MAX_CHARS": 1000 # Max character limit for the combined context string
    }
    
    # --- NLP model loading ---
    try:
        config["nlp"] = spacy.load("es_core_news_sm") # We use the Spanish model
    except Exception as e:
        st.error(f"{T['error_loading_models']} {e}")
        return None

    # The SentenceTransformer model is multilingual, no need to change
    config["embedding_model"] = SentenceTransformer('all-MiniLM-L6-v2')

    # --- Gemini client configuration (from Streamlit Secrets or environment variables) ---
    # Prioriza variables de entorno, luego Streamlit Secrets
    google_api_key = os.environ.get("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    config["TII_API_KEY"] = os.environ.get("TII_API_KEY") or st.secrets.get("TII_API_KEY") # Prioritize env var for TII_API_KEY

    if google_api_key:
        genai.configure(api_key=google_api_key)
        config["gemini_model"] = genai.GenerativeModel('gemini-1.5-flash-latest')
    else:
        config["gemini_model"] = None
        st.error(T["warning_no_google_api_key"], icon="🚨")
    
    if not config["TII_API_KEY"]:
        st.error(T["warning_no_tii_api_key"], icon="🚨")

    print(f"[{time.strftime('%H:%M:%S')}] {T['models_loaded']}")
    return config

CONFIG = load_models_and_config()
if not CONFIG:
    st.error(T["app_cannot_continue"])
    st.stop()

# ==============================================================================
# 2. SESSION STATE (The App's Live Memory)
# ==============================================================================
def get_initial_analysis_state():
    return {
        "main_topic_memory": {},
        "last_n_emb": None,
        "last_nn_emb": None,
        "last_nnn_emb": None,
        "current_n_id": 0,
        "current_nn_id": 0,
        "current_nnn_id": 0,
        "hierarchy": [],
        "topic_titles": {}
    }

if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_key" not in st.session_state:
    st.session_state.current_chat_key = "Nuevo Chat 1"
    st.session_state.chats["Nuevo Chat 1"] = {"messages": [], "analysis_state": get_initial_analysis_state()}
    st.session_state.chats["Nuevo Chat 1"]["replies_data"] = [] # New list to store reply data
if "scroll_to_index" not in st.session_state:
    st.session_state.scroll_to_index = None

# Temporary variable to store the index of the message being replied to
if 'referenced_message_index' not in st.session_state:
    st.session_state.referenced_message_index = None

# Initialize last VQA call timestamp for rate limiting
if 'last_vqa_time' not in st.session_state:
    st.session_state.last_vqa_time = 0.0

# ==============================================================================
# 3. ANALYSIS AND API LOGIC (The App's "Backend") - Auxiliary Functions
# ==============================================================================

# New function to get stopwords based on language
@st.cache_resource
def get_localized_stopwords_set(lang_code: str):
    try:
        if lang_code == "es":
            nltk.data.find('corpora/stopwords') 
            return set(stopwords.words('spanish'))
        elif lang_code == "en":
            nltk.data.find('corpora/stopwords') 
            return set(stopwords.words('english'))
        else:
            return set() 
    except LookupError:
        # Esto se intentará solo si no se encuentra localmente.
        # En despliegue, es mejor que las stopwords estén pre-descargadas (ver setup.sh o Dockerfile)
        nltk.download('stopwords', quiet=True)
        return get_localized_stopwords_set(lang_code) 

def preprocess_text(text: str):
    stopwords_set_localized = get_localized_stopwords_set(st.session_state.language)
    doc = CONFIG['nlp'](text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stopwords_set_localized]
    return ' '.join(tokens)

def get_embedding(text: str):
    return CONFIG['embedding_model'].encode(text, convert_to_tensor=True)

def cosine_similarity(emb1, emb2) -> float:
    if emb1 is None or emb2 is None: return -1.0
    return util.cos_sim(emb1.cpu(), emb2.cpu()).item()

def calculate_dynamic_weight(similarity: float) -> float:
    clamped_similarity = max(0, min(1, similarity))
    return CONFIG['MIN_PROMPT_WEIGHT'] + (clamped_similarity * (CONFIG['MAX_PROMPT_WEIGHT'] - CONFIG['MIN_PROMPT_WEIGHT']))

def generate_title_with_gemini(text_to_title: str):
    """Generates a short title using Gemini with exponential backoff."""
    if not CONFIG.get("gemini_model"): return T["error_gemini_title"]
    
    if st.session_state.language == "es":
        system_prompt = "Eres un editor experto. Basado en el siguiente texto, que inicia un tema, crea un título corto y conciso (máximo 6 palabras). Responde únicamente con el título."
    else: 
        system_prompt = "You are an expert editor. Based on the following text, which starts a topic, create a short and concise title (maximum 6 words). Respond only with the title."
    
    retries = 0
    delay = CONFIG['INITIAL_RETRY_DELAY']
    while retries < CONFIG['MAX_RETRY_ATTEMPTS']:
        try:
            response = CONFIG['gemini_model'].generate_content([system_prompt, text_to_title])
            return response.text.strip().replace('"', '')
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] ⚠️ {T['error_max_retries_gemini_title']} (attempt {retries + 1}/{CONFIG['MAX_RETRY_ATTEMPTS']}): {e}")
            retries += 1
            if retries < CONFIG['MAX_RETRY_ATTEMPTS']:
                print(f"[{time.strftime('%H:%M:%S')}] {T['error_max_retries_gemini_title']} {delay:.2f} seconds before retrying...")
                time.sleep(delay)
                delay *= CONFIG['RETRY_MULTIPLIER']
            else:
                print(f"[{time.strftime('%H:%M:%S')}] ❌ {T['error_max_retries_gemini_title']}")
                break
    return T["error_gemini_title"]

def submit_label_to_tii(task_id: str, answer: str):
    """Submits the generated answer to the TII CrowdLabel API."""
    url = "https://crowdlabel.tii.ae/api/2025.2/tasks/submit"
    headers = {
        "x-api-key": CONFIG["TII_API_KEY"],
        "Content-Type": "application/json"
    }
    data = {
        "task_id": task_id,
        "answer": answer,
        "metadata": {
            "source": "HistoryLine_AI"
        }
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=CONFIG["REQUEST_TIMEOUT"], verify=False)
        response.raise_for_status()
        print(f"[SUBMIT] ✅ Task {task_id} submitted successfully.")
    except Exception as e:
        print(f"[SUBMIT] ❌ Failed to submit task {task_id}: {e}")

def log_vqa_analysis(task_id: str, question: str, gemini_answer: str, current_chat_key: str):
    """
    Logs VQA task details and HistoryLine hierarchical analysis to a local JSONL file.
    """
    log_file_path = "vqa_historyline_log.jsonl"
    
    # Get the current hierarchical analysis from session state
    analysis_state = st.session_state.chats[current_chat_key]["analysis_state"]
    hierarchy_data = analysis_state["hierarchy"]
    topic_titles = analysis_state["topic_titles"]

    hierarchical_tags = []
    for item in hierarchy_data:
        # Simplification of hierarchy for logging
        simplified_tag = {
            "exchange_index": item.get('index'),
            "level": item.get('level'),
            "topic_id": item.get('topic_id'),
            "title_or_snippet": item.get('vqa_gemini_title') or item.get('title')
        }
        if not simplified_tag["title_or_snippet"]:
            # If no specific title, use a snippet of the text
            text_content = item.get('text', '')
            if isinstance(text_content, str) and text_content.startswith("Pregunta VQA: "):
                snippet = text_content[len("Pregunta VQA: "):].split(" Respuesta de la imagen:")[0].strip()
            else:
                snippet = text_content
            simplified_tag["title_or_snippet"] = snippet[:60] + "..." if len(snippet) > 60 else snippet
            
        hierarchical_tags.append(simplified_tag)

    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "task_id": task_id,
        "question": question,
        "gemini_answer": gemini_answer,
        "hierarchical_tags": hierarchical_tags
    }

    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + "\n")
        print(f"[LOG] ✅ VQA analysis for task {task_id} logged to {log_file_path}")
    except Exception as e:
        print(f"[LOG] ❌ Failed to log VQA analysis for task {task_id}: {e}")

# --- NEW CONTEXTUALIZATION FUNCTION ---
def get_contextual_prompt_prefix(current_chat_key: str) -> str:
    """
    Generates a concise contextual prefix for the Gemini prompt based on
    the current chat's hierarchy and recent messages.
    """
    active_chat = st.session_state.chats[current_chat_key]
    analysis_state = active_chat["analysis_state"]
    hierarchy = analysis_state["hierarchy"]
    messages = active_chat["messages"]
    
    context_lines = []
    
    # Add recent hierarchy items
    # Iterate in reverse to get the most recent relevant items
    num_hierarchy_items = 0
    # Exclude the very last entry in hierarchy if it's not fully processed for this turn
    history_to_sample_hierarchy = hierarchy[:-1] if hierarchy else [] 

    for item in reversed(history_to_sample_hierarchy):
        if num_hierarchy_items >= CONFIG['CONTEXT_HIERARCHY_MAX_ITEMS']:
            break

        level = item['level']
        topic_id = item['topic_id']
        title = item.get('title') or item.get('vqa_gemini_title')
        text_snippet = ""
        
        # Get a snippet if no specific title exists
        if not title:
            original_text = item.get('text', '')
            if isinstance(original_text, dict) and original_text.get('type') == 'vqa':
                text_snippet = original_text['question']
            elif isinstance(original_text, str) and original_text.startswith("Pregunta VQA: "):
                text_snippet = original_text[len("Pregunta VQA: "):].split(" Respuesta de la imagen:")[0].strip()
            else:
                text_snippet = original_text
            text_snippet = text_snippet if len(text_snippet) < 60 else text_snippet[:57] + "..."
            title = text_snippet # Use snippet as title for context if no actual title

        line_to_add = ""
        if level == 'n':
            line_to_add = f"- {T['topic_prefix']} {topic_id}: {title}"
        elif level == 'n (revisit)':
            line_to_add = f"- {T['revisiting_topic']} {topic_id}): {title}"
        elif level == 'n.n':
            line_to_add = f"  - {T['subtopic_prefix']}: {title}"
        elif level == 'n.n.n':
            line_to_add = f"    - {T['detail_prefix']}{title}\""
        
        if line_to_add:
            context_lines.append(line_to_add)
            num_hierarchy_items += 1

    # Reverse the hierarchy context lines so they appear chronologically (older first)
    context_lines.reverse()

    # Add recent chat exchanges
    # We want the last X full exchanges (user + assistant) excluding the current turn.
    # So, we look at messages up to the second to last message (index len-2)
    chat_exchanges_to_include = []
    num_exchanges_added = 0
    # messages list: [user0, assistant0, user1, assistant1, ..., userN-1, assistantN-1, userN (current prompt)]
    # We want up to assistantN-1, so iterate backwards from len(messages) - 2
    
    # Ensure we don't go out of bounds and messages are paired
    start_index = len(messages) - 2 # Start from the second to last message (likely an assistant response)
    
    for i in range(start_index, -1, -1):
        if num_exchanges_added >= CONFIG['CONTEXT_CHAT_MAX_EXCHANGES']:
            break
        
        # We need a pair: assistant response followed by user message
        if messages[i]['role'] == 'assistant' and (i - 1) >= 0 and messages[i-1]['role'] == 'user':
            user_msg = messages[i-1]['content']
            assistant_msg = messages[i]['content']

            user_text_snippet = ""
            if isinstance(user_msg, dict) and user_msg.get('type') == 'vqa':
                user_text_snippet = f"VQA: {user_msg['question'][:70]}{'...' if len(user_msg['question']) > 70 else ''}"
            else:
                user_text_snippet = user_msg[:70] + "..." if len(user_msg) > 70 else user_msg

            assistant_text_snippet = assistant_msg[:70] + "..." if len(assistant_msg) > 70 else assistant_msg

            chat_exchanges_to_include.append(f"  Usuario: \"{user_text_snippet}\"\n  Asistente: \"{assistant_text_snippet}\"")
            num_exchanges_added += 1
            # Decrement i by an additional 1 to skip the user message in the next iteration
            # as it's already processed as part of this exchange. The for loop's decrement handles one.
            # No explicit `i -= 1` needed here as `range` handles iteration.
            
    chat_exchanges_to_include.reverse() # Order from oldest to newest

    # Combine all parts into a single string
    full_context_parts = []
    
    if context_lines:
        full_context_parts.append(f"{T['topic_prefix']}:")
        full_context_parts.extend(context_lines)
    
    if chat_exchanges_to_include:
        if full_context_parts: 
            full_context_parts.append("\n--- Historial de conversación reciente ---")
        else: 
            full_context_parts.append("Historial de conversación reciente:")
        full_context_parts.extend(chat_exchanges_to_include)

    # Final string combining all context
    final_context_string = "\n".join(full_context_parts)
    
    # Trim context to max characters as specified in CONFIG
    if len(final_context_string) > CONFIG['CONTEXT_MAX_CHARS']:
        final_context_string = final_context_string[:CONFIG['CONTEXT_MAX_CHARS'] - len("\n... (contexto truncado)")] + "\n... (contexto truncado)"

    return final_context_string if final_context_string else ""


def get_gemini_response(prompt_content: Union[str, dict], referenced_message_content: Union[str, dict, None] = None, current_chat_key: str = None):
    """
    Gets a response from Gemini for a text prompt (or VQA).
    Implements exponential backoff for Gemini calls.
    Now includes a contextual prefix based on chat history.
    """
    gemini_model = CONFIG.get("gemini_model")
    if not gemini_model:
        return T["error_gemini_api"]

    # Determine if it's a VQA task or a regular text chat
    is_vqa_prompt = isinstance(prompt_content, dict) and prompt_content.get('type') == 'vqa'

    # Get contextual prefix for the LLM based on HistoryLine analysis
    # This will be used for both text and VQA prompts
    context_prefix = get_contextual_prompt_prefix(current_chat_key)

    # Set up system instructions for the LLM
    # VQA specific system prompt - ALWAYS in English and explicitly requests English response
    system_instructions_vqa_english = "You are an expert vision assistant that answers questions about images. Your goal is to provide a concise and relevant answer based on the image and the provided conversation context. **Respond strictly in English.**"
    
    if st.session_state.language == "es":
        system_instructions_general = "Eres HistoryLine AI, un asistente de conversación experto en mantener el hilo de la conversación y proporcionar respuestas precisas y contextualizadas."
    else:
        system_instructions_general = "You are HistoryLine AI, an expert conversational assistant that maintains the conversation flow and provides accurate, contextualized answers."

    final_prompt_parts = []

    if is_vqa_prompt:
        final_prompt_parts.append(system_instructions_vqa_english) # Use the English VQA prompt
        if context_prefix:
            # Context for VQA can be mixed language, but the instruction to respond in English is key.
            final_prompt_parts.append(f"\n--- CONTEXT --- Context for the image question from conversation history:\n{context_prefix}\n--- END CONTEXT ---")
        
        # The original question from the task, can be in any language
        final_prompt_parts.append(f"\nQuestion about the image: {prompt_content['question']}")
        
        # Call get_vqa_response_with_gemini which handles image download and vision model call
        # The 'question' parameter for get_vqa_response_with_gemini is now the combined text prompt
        combined_text_prompt_for_vqa = "\n".join(final_prompt_parts)
        
        return get_vqa_response_with_gemini(
            prompt_content['image_url'],
            combined_text_prompt_for_vqa, # Pass the combined text prompt for VQA
            prompt_content.get('task_id'),
            current_chat_key
        )
    else:
        # This path is for regular text chat.
        final_prompt_parts.append(system_instructions_general)
        
        if context_prefix:
            if st.session_state.language == "es":
                final_prompt_parts.append(f"\n--- CONTEXTO DE LA CONVERSACIÓN ---\n{context_prefix}\n--- FIN CONTEXTO ---")
            else:
                final_prompt_parts.append(f"\n--- CONVERSATION CONTEXT ---\n{context_prefix}\n--- END CONTEXT ---")

        if referenced_message_content:
            if isinstance(referenced_message_content, dict) and referenced_message_content.get('type') == 'vqa':
                referenced_text_snippet = f"el mensaje VQA anterior con la pregunta: '{referenced_message_content['question'][:70]}{'...' if len(referenced_message_content['question']) > 70 else ''}'"
            else:
                referenced_text_snippet = f"su mensaje anterior: '{referenced_message_content[:70]}{'...' if len(referenced_message_content) > 70 else ''}'"
            
            if st.session_state.language == "es":
                final_prompt_parts.append(f"\nEl usuario está respondiendo a {referenced_text_snippet}.")
                final_prompt_parts.append(f"Mensaje actual del usuario: '{prompt_content}'")
            else:
                final_prompt_parts.append(f"\nUser is responding to {referenced_text_snippet}.")
                final_prompt_parts.append(f"User's current message: '{prompt_content}'")
        else:
            if st.session_state.language == "es":
                final_prompt_parts.append(f"\nMensaje actual del usuario: '{prompt_content}'")
            else:
                final_prompt_parts.append(f"\nUser's current message: '{prompt_content}'")

        # The actual input to generate_content for text prompts
        gemini_input_for_text = [str(part) for part in final_prompt_parts]

    retries = 0
    delay = CONFIG['INITIAL_RETRY_DELAY']
    while retries < CONFIG['MAX_RETRY_ATTEMPTS']:
        try:
            # For text-based prompts, use gemini_input_for_text
            response = gemini_model.generate_content(gemini_input_for_text)
            return response.text
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] ⚠️ {T['error_max_retries_gemini_chat']} (attempt {retries + 1}/{CONFIG['MAX_RETRY_ATTEMPTS']}): {e}")
            retries += 1
            if retries < CONFIG['MAX_RETRY_ATTEMPTS']:
                print(f"[{time.strftime('%H:%M:%S')}] {delay:.2f} seconds before retrying...")
                time.sleep(delay)
                delay *= CONFIG['RETRY_MULTIPLIER']
            else:
                print(f"[{time.strftime('%H:%M:%S')}] ❌ {T['error_max_retries_gemini_chat']}")
                break
    return T["error_gemini_api"]


def fetch_tii_vqa_task():
    """Gets a VQA task from the TII API with exponential backoff and timeout."""
    if not CONFIG.get("TII_API_KEY"):
        st.error(T["warning_no_tii_api_key"])
        return None

    headers = {'x-api-key': CONFIG['TII_API_KEY']}
    print(f"[{time.strftime('%H:%M:%S')}] {T['vqa_spinner_text'].split('...')[0]} {CONFIG['TII_BASE_URL']}")
    
    retries = 0
    delay = CONFIG['INITIAL_RETRY_DELAY']
    while retries < CONFIG['MAX_RETRY_ATTEMPTS']:
        try:
            # TII Production endpoint (2025.2) for picking tasks, with category='vqa'
            response = requests.get(CONFIG['TII_BASE_URL'], headers=headers, params={"category": "vqa"}, timeout=CONFIG['REQUEST_TIMEOUT'], verify=False) 
            response.raise_for_status() # Raises HTTPError for 4xx/5xx
            tasks = response.json()
            
            print(f"[{time.strftime('%H:%M:%S')}] Full response from TII API:\n{json.dumps(tasks, indent=4)}")

            if tasks and len(tasks) > 0:
                if ('content' in tasks[0] and 'image' in tasks[0]['content'] and 'url' in tasks[0]['content']['image'] and
                    'task' in tasks[0] and 'text' in tasks[0]['task']):
                    print(f"[{time.strftime('%H:%M:%S')}] ✅ {T['vqa_toast_success'].split('!')[0]}.")
                    return tasks[0]
                else:
                    st.toast(T["vqa_toast_invalid_format"])
                    print(f"[{time.strftime('%H:%M:%S')}] {T['vqa_toast_invalid_format']}:\n{json.dumps(tasks[0], indent=4)}")
                    return None
            else:
                st.toast(T["vqa_toast_no_task"])
                return None
        except requests.exceptions.Timeout:
            print(f"[{time.strftime('%H:%M:%S')}] ⚠️ {T['error_api_timeout']} (attempt {retries + 1}).")
        except requests.exceptions.RequestException as e:
            print(f"[{time.strftime('%H:%M:%S')}] ⚠️ {T['error_request_exception']} (attempt {retries + 1}): {e}")
        except json.JSONDecodeError:
            print(f"[{time.strftime('%H:%M:%S')}] ⚠️ {T['error_api_generic']} (attempt {retries + 1}).")
            
        retries += 1
        if retries < CONFIG['MAX_RETRY_ATTEMPTS']:
            print(f"[{time.strftime('%H:%M:%S')}] Waiting {delay:.2f} seconds before retrying...")
            time.sleep(delay)
            delay *= CONFIG['RETRY_MULTIPLIER']
        else:
            print(f"[{time.strftime('%H:%M:%S')}] ❌ {T['error_max_retries_vqa_task']}")
            break
    return None

def get_vqa_response_with_gemini(image_url: str, question: str, task_id: str = None, current_chat_key: str = None):
    """Downloads an image and gets a response from Gemini Vision with exponential backoff and timeout."""
    gemini_model = CONFIG.get("gemini_model")
    if not gemini_model: return T["error_gemini_vqa"]
    
    print(f"[{time.strftime('%H:%M:%S')}] {T['vqa_spinner_text'].split('...')[0]} {image_url}...")
    retries = 0
    delay = CONFIG['INITIAL_RETRY_DELAY']
    while retries < CONFIG['MAX_RETRY_ATTEMPTS']:
        try:
            response_image = requests.get(image_url, stream=True, timeout=CONFIG['REQUEST_TIMEOUT'], verify=False)
            response_image.raise_for_status()
            image = Image.open(io.BytesIO(response_image.content))
            
            # The 'question' parameter here is already the combined text prompt from get_gemini_response
            # It now contains the VQA system prompt, context, and original question.
            response_gemini = gemini_model.generate_content([image, question]) 
            gemini_answer = response_gemini.text
            
            # --- Fallback response for Gemini failure or empty response ---
            if not gemini_answer or gemini_answer.strip() == "":
                gemini_answer = "No answer available at this time."

            # --- Trim Gemini response to max 512 characters ---
            gemini_answer = gemini_answer[:512]

            print(f"[{time.strftime('%H:%M:%S')}] ✅ {T['vqa_toast_success'].split('!')[0]}.")

            # --- SUBMISSION STEP ---
            if task_id:
                # Label is sent to the TII API using production endpoint "2025.2" and category='vqa' implicit in task type
                submit_label_to_tii(task_id, gemini_answer)
                # Log the VQA analysis after successful submission
                if current_chat_key:
                    log_vqa_analysis(task_id, question, gemini_answer, current_chat_key)
            # --- END SUBMISSION STEP ---

            return gemini_answer
        except requests.exceptions.Timeout:
            st.error(f"{T['error_api_timeout']} (attempt {retries + 1}).")
            # If timeout, and it's the last retry, return fallback
            if retries == CONFIG['MAX_RETRY_ATTEMPTS'] -1:
                return "No answer available at this time."
        except requests.exceptions.RequestException as e:
            st.error(f"{T['error_request_exception']} (attempt {retries + 1}): {e}")
            # If request exception, and it's the last retry, return fallback
            if retries == CONFIG['MAX_RETRY_ATTEMPTS'] -1:
                return "No answer available at this time."
        except Exception as e:
            st.error(f"{T['error_api_generic']} (attempt {retries + 1}): {e}")
            # If generic error, and it's the last retry, return fallback
            if retries == CONFIG['MAX_RETRY_ATTEMPTS'] -1:
                return "No answer available at this time."
        
        retries += 1
        if retries < CONFIG['MAX_RETRY_ATTEMPTS']:
            print(f"[{time.strftime('%H:%M:%S')}] Waiting {delay:.2f} seconds before retrying...")
            time.sleep(delay)
            delay *= CONFIG['RETRY_MULTIPLIER']
        else:
            print(f"[{time.strftime('%H:%M:%S')}] ❌ {T['error_max_retries_vqa_gemini']}")
            break
    return "No answer available at this time." # Return fallback if all retries fail and no specific error returned it earlier

def send_vqa_auto():
    """
    Automatically fetches a VQA task, gets Gemini's answer, submits it to TII,
    and logs the analysis. This is triggered by various user interactions.
    This function executes in the background without changing the active chat tab.
    """
    # Prevent multiple rapid API calls: ensure at least a 5-second delay
    if time.time() - st.session_state.last_vqa_time < 5: # 5-second delay
        print(f"[{time.strftime('%H:%M:%S')}] ⚠️ Skipping automatic VQA task due to rate limit.")
        return

    if not CONFIG.get("gemini_model"):
        print(f"[{time.strftime('%H:%M:%S')}] ❌ Cannot perform automatic VQA: GOOGLE_API_KEY not configured.")
        return
    elif not CONFIG.get("TII_API_KEY"):
        print(f"[{time.strftime('%H:%M:%S')}] ❌ Cannot perform automatic VQA: TII_API_KEY not configured.")
        return

    # Fetch task without displaying a spinner to the user immediately
    task = fetch_tii_vqa_task()
    st.session_state.last_vqa_time = time.time() # Update timestamp after this fetch attempt

    # If fetch_tii_vqa_task() returns None, stop the execution immediately.
    # No spinner is shown from here, and no toast is displayed if there's no valid task.
    if task is None:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ fetch_tii_vqa_task returned None. Stopping auto VQA.")
        return

    # If a task is successfully retrieved, then proceed with Gemini processing
    # The spinner associated with T['vqa_spinner_text'] is typically in the frontend function
    # that calls send_vqa_auto (like the "Get TII VQA Task" button).
    # For background execution, we will simply proceed.
    
    # Extract details from the fetched task
    image_url_from_task = task['content']['image']['url']
    question_from_task = task['task']['text']

    vqa_prompt_content = {
        "type": "vqa",
        "image_url": image_url_from_task,
        "question": question_from_task,
        "task_id": task.get('id')
    }
    
    # Get Gemini's answer, which will also handle submission and logging
    gemini_answer = get_gemini_response(vqa_prompt_content, current_chat_key=st.session_state.current_chat_key)
    
    # Append the VQA exchange to the dedicated VQA chat, but do NOT switch to it.
    vqa_chat_key = f"{T['topic_prefix']} VQA TII"
    if vqa_chat_key not in st.session_state.chats:
        st.session_state.chats[vqa_chat_key] = {
            "messages": [],
            "analysis_state": get_initial_analysis_state(),
            "replies_data": [] # Initialize replies_data for new VQA chats
        }
    
    # Check if the last message in the VQA chat is the same VQA task to avoid duplicates on quick reruns
    last_message_content = st.session_state.chats[vqa_chat_key]['messages'][-1]['content'] if st.session_state.chats[vqa_chat_key]['messages'] else None
    
    is_duplicate_vqa = False
    if isinstance(last_message_content, dict) and last_message_content.get('type') == 'vqa' and last_message_content.get('task_id') == task.get('id'):
        is_duplicate_vqa = True

    if not is_duplicate_vqa:
        st.session_state.chats[vqa_chat_key]['messages'].append({"role": "user", "content": vqa_prompt_content})
        st.session_state.chats[vqa_chat_key]['messages'].append({"role": "assistant", "content": gemini_answer})
        exchange_index = (len(st.session_state.chats[vqa_chat_key]['messages']) // 2) - 1
        process_single_exchange_for_hierarchy(vqa_prompt_content, gemini_answer, exchange_index)
        
        # Do NOT set st.session_state.current_chat_key here
        # Do NOT display st.toast here
        print(f"[{time.strftime('%H:%M:%S')}] ✅ Automatic VQA task {task.get('id')} processed in background.")

    # Rerun is handled by the calling button/interaction, no need for an explicit st.rerun() here.


# --- INCREMENTAL HIERARCHICAL ANALYSIS LOGIC (PAIR BY PAIR INGESTION) ---
def process_single_exchange_for_hierarchy(user_message_content, assistant_response_content, exchange_index: int):
    """
    Processes a single pair (user prompt, assistant response)
    and updates the topic hierarchy state in real-time.
    """
    active_chat_state = st.session_state.chats[st.session_state.current_chat_key]["analysis_state"]

    # Extract text for embeddings and display
    is_vqa_exchange = isinstance(user_message_content, dict) and user_message_content.get('type') == 'vqa'
    if is_vqa_exchange:
        prompt_text_for_embedding = user_message_content['question']
        doc_text_prompt_for_display = f"Pregunta VQA: {user_message_content['question']}"
    else:
        prompt_text_for_embedding = user_message_content
        doc_text_prompt_for_display = user_message_content
    
    response_text_for_embedding = assistant_response_content
    # CORRECTION: Use assistant_response_content correctly
    doc_text_response_for_display = assistant_response_content


    # Combine for the full document text (to generate titles if necessary)
    doc_text = f"{doc_text_prompt_for_display} {doc_text_response_for_display}"

    # Preprocess and get embeddings
    prompt_clean = preprocess_text(prompt_text_for_embedding)
    response_clean = preprocess_text(response_text_for_embedding)
    
    emb_prompt = get_embedding(prompt_clean)
    emb_response = get_embedding(response_clean)
    
    sim_pr = cosine_similarity(emb_prompt, emb_response)
    prompt_weight = calculate_dynamic_weight(sim_pr)
    emb = (prompt_weight * emb_prompt) + ((1.0 - prompt_weight) * emb_response)
    
    # Retrieve current memory state (now directly updated in the dictionary!)
    main_topic_memory = active_chat_state["main_topic_memory"]
    last_n_emb = active_chat_state["last_n_emb"]
    last_nn_emb = active_chat_state["last_nn_emb"]
    last_nnn_emb = active_chat_state["last_nnn_emb"]
    current_n_id = active_chat_state["current_n_id"]
    current_nn_id = active_chat_state["current_nn_id"]
    current_nnn_id = active_chat_state["current_nnn_id"]
    topic_titles = active_chat_state["topic_titles"]

    current_level_info = {} # To store the result of this exchange

    # Change detection logic (identical to the original)
    if exchange_index == 0: # First exchange of the chat
        current_n_id = 1
        current_nn_id = 1
        current_nnn_id = 1
        main_topic_memory[current_n_id] = emb
        last_n_emb, last_nn_emb, last_nnn_emb = emb, emb, emb
        
        # For the first topic, we always generate a title.
        title = generate_title_with_gemini(doc_text)
        topic_titles[current_n_id] = title
        
        current_level_info = {'index': exchange_index, 'level': 'n', 'topic_id': current_n_id, 'text': doc_text, 'title': title}
        print(f"[{time.strftime('%H:%M:%S')}] {T['new_main_topic_detected']} {current_n_id}. {title}")
    else:
        sim_nnn = cosine_similarity(emb, last_nnn_emb)
        sim_nn = cosine_similarity(emb, last_nn_emb)
        sim_n = cosine_similarity(emb, last_n_emb)
        
        if sim_nnn > CONFIG['THRESHOLD_DETAIL']:
            current_level_info = {'index': exchange_index, 'level': 'n.n.n', 'topic_id': f"{current_n_id}.{current_nn_id}.{current_nnn_id}", 'text': doc_text}
            last_nnn_emb = emb
            print(f"[{time.strftime('%H:%M:%S')}] {T['continuation_of_detail']} (Topic {current_n_id}.{current_nn_id}.{current_nnn_id})")
        elif sim_nn > CONFIG['THRESHOLD_SUBTOPIC']:
            current_nnn_id += 1
            current_level_info = {'index': exchange_index, 'level': 'n.n.n', 'topic_id': f"{current_n_id}.{current_nn_id}.{current_nnn_id}", 'text': doc_text}
            last_nnn_emb = emb
            print(f"[{time.strftime('%H:%M:%S')}] {T['new_sub_subtopic']} (Topic {current_n_id}.{current_nn_id}.{current_nnn_id})")
        elif sim_n > CONFIG['THRESHOLD_NEW_TOPIC']:
            current_nn_id += 1
            current_nnn_id = 1
            current_level_info = {'index': exchange_index, 'level': 'n.n', 'topic_id': f"{current_n_id}.{current_nn_id}", 'text': doc_text}
            last_nn_emb = emb
            last_nnn_emb = emb
            print(f"[{time.strftime('%H:%M:%S')}] {T['new_subtopic']} (Topic {current_n_id}.{current_nn_id})")
        else:
            best_match_id, max_similarity = None, -1.0
            for topic_id, topic_emb in main_topic_memory.items():
                sim = cosine_similarity(emb, topic_emb)
                if sim > max_similarity: max_similarity, best_match_id = sim, topic_id
            
            if max_similarity > CONFIG['REVISIT_THRESHOLD']:
                current_n_id = best_match_id
                current_nn_id = 1
                current_nnn_id = 1
                last_n_emb = main_topic_memory[current_n_id]
                last_nn_emb, last_nnn_emb = last_n_emb, last_n_emb
                title_revisit = topic_titles.get(current_n_id, f"{T['topic_prefix']} {current_n_id} ({T['pending_title']})")
                current_level_info = {'index': exchange_index, 'level': 'n (revisit)', 'topic_id': current_n_id, 'text': doc_text, 'title_revisit': title_revisit}
                print(f"[{time.strftime('%H:%M:%S')}] {T['revisiting_topic_log']} {current_n_id}. {title_revisit}")
            else:
                current_n_id = len(main_topic_memory) + 1
                current_nn_id = 1
                current_nnn_id = 1
                main_topic_memory[current_n_id] = emb
                last_n_emb = emb
                last_nn_emb = emb
                last_nnn_emb = emb
                
                title = generate_title_with_gemini(doc_text)
                topic_titles[current_n_id] = title
                
                current_level_info = {'index': exchange_index, 'level': 'n', 'topic_id': current_n_id, 'text': doc_text, 'title': title}
                print(f"[{time.strftime('%H:%M:%S')}] {T['new_main_topic_unrelated_detected']} {current_n_id}. {title}")
    
    # === New: Generate title for VQA regardless of hierarchy level ===
    if is_vqa_exchange:
        vqa_title = generate_title_with_gemini(doc_text)
        current_level_info['vqa_gemini_title'] = vqa_title
        print(f"[{time.strftime('%H:%M:%S')}] {T['vqa_specific_title_generated']} {vqa_title}")

    # Update session state with new values
    active_chat_state["main_topic_memory"] = main_topic_memory
    active_chat_state["last_n_emb"] = last_n_emb
    active_chat_state["last_nn_emb"] = last_nn_emb
    active_chat_state["last_nnn_emb"] = last_nnn_emb
    active_chat_state["current_n_id"] = current_n_id
    active_chat_state["current_nn_id"] = current_nn_id
    active_chat_state["current_nnn_id"] = current_nnn_id
    active_chat_state["hierarchy"].append(current_level_info)
    active_chat_state["topic_titles"] = topic_titles


# ==============================================================================
# 4. INITIAL CHAT LOADING FROM FILE (EXAMPLE)
# ==============================================================================
if "initial_load_done" not in st.session_state:
    try:
        with open("chat_ia.txt", 'r', encoding='utf-8') as f: content = f.read()
        
        parts = re.split(r'(Dijiste:|ChatGPT dijo:)', content)
        messages_from_file = []
        if len(parts) > 1:
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts) and parts[i+1].strip():
                    role = "user" if parts[i].strip() == "Dijiste:" else "assistant"
                    messages_from_file.append({"role": role, "content": parts[i+1].strip()})
        
        chat_file_key = "Chat de Archivo (ia)"
        st.session_state.chats[chat_file_key] = {
            "messages": messages_from_file,
            "analysis_state": get_initial_analysis_state(),
            "replies_data": [] # Initialize for file chats as well
        }
        
        temp_current_chat_key = st.session_state.current_chat_key
        st.session_state.current_chat_key = chat_file_key

        print(f"[{time.strftime('%H:%M:%S')}] {T['generating_initial_analysis']} '{chat_file_key}' (simulating history load)...")
        
        current_analysis_state = st.session_state.chats[chat_file_key]["analysis_state"]
        current_analysis_state["main_topic_memory"] = {}
        current_analysis_state["last_n_emb"] = None
        current_analysis_state["last_nn_emb"] = None
        current_analysis_state["last_nnn_emb"] = None
        current_analysis_state["current_n_id"] = 0
        current_analysis_state["current_nn_id"] = 0
        current_analysis_state["current_nnn_id"] = 0
        current_analysis_state["hierarchy"] = []
        current_analysis_state["topic_titles"] = {}
        
        for i in range(0, len(messages_from_file), 2):
            user_msg_content = messages_from_file[i]['content']
            assistant_msg_content = messages_from_file[i+1]['content'] if (i+1) < len(messages_from_file) else ""
            process_single_exchange_for_hierarchy(user_msg_content, assistant_msg_content, i // 2)
        
        print(f"[{time.strftime('%H:%M:%S')}] {T['initial_analysis_completed']} '{chat_file_key}' completed.")
        
        st.session_state.current_chat_key = temp_current_chat_key

        if not st.session_state.chats["Nuevo Chat 1"]["messages"]:
            st.session_state.current_chat_key = chat_file_key
            
    except FileNotFoundError:
        print(f"[{time.strftime('%H:%M:%S')}] {T['no_chat_file_found']}")
        example_messages = [
            {"role": "user", "content": "¿Me podrías explicar qué es la computación cuántica de una manera sencilla?"},
            {"role": "assistant", "content": "Claro. Imagina que las computadoras normales usan bits que pueden ser 0 o 1. Las computadoras cuánticas usan 'qubits' que pueden ser 0, 1 o una combinación de ambos al mismo tiempo. Esto se llama superposición. También tienen 'entrelazamiento', donde dos qubits están conectados sin importar la distancia, y 'interferencia' para explorar muchas posibilidades a la vez. Todo esto les permite resolver problemas que las computadoras clásicas no pueden, como descubrir nuevos medicamentos o materiales, o romper cifrados."},
            {"role": "user", "content": "Aquí hay un mensaje de VQA para que HistoryLine lo procese:", "type": "vqa", "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Quantum_computer_IBM_Q_System_One.jpg/640px-Quantum_computer_IBM_Q_System_One.jpg", "question": "¿Qué tipo de dispositivo se muestra en esta imagen y cuál es su propósito principal?"},
            {"role": "assistant", "content": "La imagen muestra un dispositivo de computación cuántica, como el IBM Q System One. Su propósito principal es realizar cálculos complejos y resolver problemas que están más allá de las capacidades de las computadoras clásicas, utilizando principios de la mecánica cuántica como la superposición y el entrelazamiento."},
            {"role": "user", "content": "¿Y qué tan cerca estamos de tener computadoras cuánticas que realmente puedan hacer esto a gran escala?"},
            {"role": "assistant", "content": "Estamos en una fase de desarrollo muy activa. Ya existen computadoras cuánticas funcionales, pero la mayoría son 'NISQ' (Noisy Intermediate-Scale Quantum) devices. Son pequeñas, propensas a errores y aún no superan a las supercomputadoras clásicas en todas las tareas. Pero la investigación avanza rápidamente, con empresas como IBM, Google y varias universidades invirtiendo fuertemente. Se espera que en la próxima década veamos avances significativos hacia computadoras cuánticas tolerantes a fallos que puedan abordar problemas comerciales y científicos realmente complejos."},
            {"role": "user", "content": "Entiendo. ¿Y qué papel juega la criptografía en todo esto? ¿Las computadoras cuánticas la pondrán en peligro?"},
            {"role": "assistant", "content": "Esa es una excelente pregunta y un área crítica de investigación. Algunos algoritmos de cifrado actuales, como RSA, que se basan en la dificultad de factorizar números grandes, podrían ser vulnerables a algoritmos cuánticos como el algoritmo de Shor. Por eso, se está desarrollando la 'criptografía post-cuántica' (PQC), que son algoritmos de cifrado diseñados para ser seguros incluso frente a computadoras cuánticas potente. Muchas organizaciones ya están investigando y preparándose para migrar a estos nuevos estándares de seguridad. Es un campo en constante evolución."},
            {"role": "user", "content": "Entonces, ¿la computación cuántica no solo es una promesa de avance sino también un desafío para la seguridad digital?"},
            {"role": "assistant", "content": "Exactamente. Es una moneda de dos caras. Por un lado, abre puertas a soluciones revolucionarias en ciencia, medicina y optimización. Por otro, presenta un desafío significativo para la seguridad de la información tal como la conocemos hoy. La clave está en la investigación y el desarrollo proactivo de medidas de seguridad que puedan coexistir con estas nuevas capacidades. La colaboración entre gobiernos, la academia y la industria es fundamental para asegurar una transición segura a la era post-cuántica."}
        ]
        chat_example_key = "Chat de Ejemplo (cuántico)"
        st.session_state.chats[chat_example_key] = {
            "messages": [],
            "analysis_state": get_initial_analysis_state(),
            "replies_data": [] # Initialize for the example chat
        }

        temp_current_chat_key = st.session_state.current_chat_key
        st.session_state.current_chat_key = chat_example_key

        for i in range(0, len(example_messages), 2):
            user_msg = example_messages[i]['content']
            assistant_msg = example_messages[i+1]['content'] if (i+1) < len(example_messages) else ""
            process_single_exchange_for_hierarchy(user_msg, assistant_msg, i // 2)
        print(f"[{time.strftime('%H:%M:%S')}] {T['initial_analysis_completed']} '{chat_example_key}' completed.")

        st.session_state.current_chat_key = temp_current_chat_key

        if not st.session_state.chats["Nuevo Chat 1"]["messages"]:
            st.session_state.current_chat_key = chat_example_key
    st.session_state.initial_load_done = True


# ==============================================================================
# 5. 3-PANEL INTERFACE RENDERING
# ==============================================================================
left_col, main_col, right_col = st.columns([0.2, 0.5, 0.3])

# --- Left Panel ---
with left_col:
    st.header(T["my_chats_header"])

    # Language Selector
    selected_language = st.selectbox(
        label=T["language_selector_label"],
        options=["es", "en"],
        format_func=lambda x: "Español" if x == "es" else "English",
        index=0 if st.session_state.language == "es" else 1,
        key="language_selector"
    )
    if selected_language != st.session_state.language:
        st.session_state.language = selected_language
        st.rerun() # Rerun to apply language changes to the entire UI

    # --- BUTTON COMMENTED OUT ---
    # if st.button(T["get_tii_vqa_button"], use_container_width=True):
    #       send_vqa_auto()
    #       st.rerun()

    st.divider()
    for chat_key in list(st.session_state.chats.keys()):
        display_chat_name = chat_key
        if st.session_state.language == "en":
            if chat_key == "Chat de Archivo (ia)": display_chat_name = "File Chat (ia)"
            elif chat_key == "Chat de Ejemplo (cuántico)": display_chat_name = "Example Chat (quantum)"
            # Handle VQA chat key translation
            elif chat_key.startswith("Tópico VQA TII"): display_chat_name = "TII VQA Task"
            elif chat_key.startswith("Nuevo Chat"): display_chat_name = "New Chat " + chat_key.split(" ")[2]


        if st.button(display_chat_name, use_container_width=True, type=("primary" if st.session_state.current_chat_key == chat_key else "secondary")):
            st.session_state.current_chat_key = chat_key
            st.session_state.scroll_to_index = None
            st.rerun()
    if st.button(T["new_chat_button"], use_container_width=True, key="new_chat_button"):
        key = f"{T['new_chat_button'].split(' ')[1]} Chat {len(st.session_state.chats) + 1}" # Use translated text
        st.session_state.chats[key] = {
            "messages": [],
            "analysis_state": get_initial_analysis_state(),
            "replies_data": [] # Initialize replies_data for new chats
        }
        st.session_state.scroll_to_index = None
        st.rerun()

# --- Main Panel ---
with main_col:
    st.title(f"{T['chat_title_prefix']} {st.session_state.current_chat_key}")
    active_chat = st.session_state.chats[st.session_state.current_chat_key]

    chat_container = st.container(height=600, border=False)
    with chat_container:
        for i, message in enumerate(active_chat["messages"]):
            message_container = st.container()
            with message_container:
                st.markdown(f"<div id='msg-{i}'></div>", unsafe_allow_html=True)
                with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
                    if isinstance(message['content'], dict) and message['content'].get('type') == 'vqa':
                        st.image(message['content']['image_url'], caption=message['content']['question'])
                        content_to_display = message['content']['question']
                    else:
                        content_to_display = message['content']

                    msg_col, btn_col = st.columns([0.9, 0.1])
                    with msg_col:
                        st.markdown(content_to_display)
                    with btn_col:
                        # When clicking reply, save the index of the referenced message
                        if st.button("↩️", key=f"reply_{st.session_state.current_chat_key}_{i}", help=T["reply_button_tooltip"]):
                            st.session_state.referenced_message = content_to_display # Content to display in the info box
                            st.session_state.referenced_message_index = i # Exact index of the message being replied to
                            
                            send_vqa_auto() # Trigger automatic VQA task on reply button click
                            st.toast(T["preparing_response_toast"], icon="💬") # Show "preparing reply" toast
                            st.rerun()

    if 'referenced_message' in st.session_state and st.session_state.referenced_message_index is not None:
        st.info(f"{T['responding_to']} \"{st.session_state.referenced_message[:70]}{'...' if len(st.session_state.referenced_message) > 70 else ''}\"")

    if prompt := st.chat_input(T["chat_input_placeholder"]):
        referenced_context = st.session_state.pop('referenced_message', None)
        referenced_index = st.session_state.pop('referenced_message_index', None) # Retrieve the saved index

        active_chat["messages"].append({"role": "user", "content": prompt})

        with st.spinner(T["thinking_spinner_text"]):
            # Pass current_chat_key to get_gemini_response for context generation
            response_text = get_gemini_response(prompt, referenced_context, current_chat_key=st.session_state.current_chat_key)
            active_chat["messages"].append({"role": "assistant", "content": response_text})

        # === PROCESS THIS NEW EXCHANGE FOR THE HIERARCHY ===
        exchange_index = (len(active_chat['messages']) // 2) - 1
        process_single_exchange_for_hierarchy(prompt, response_text, exchange_index)

        # === SAVE THE REPLY RELATIONSHIP IF IT EXISTS ===
        if referenced_index is not None:
            # The message that replies (the user's prompt we just added) is the second to last message.
            replying_message_index = len(active_chat['messages']) - 2

            # Get the full content of the message that was replied to
            replied_to_full_content = active_chat['messages'][referenced_index]['content']
            if isinstance(replied_to_full_content, dict) and replied_to_full_content.get('type') == 'vqa':
                replied_to_text = replied_to_full_content['question'] # Use the VQA question
            else:
                replied_to_text = replied_to_full_content # Use the normal text

            active_chat["replies_data"].append({
                'replying_message_index': replying_message_index,
                'replied_to_message_index': referenced_index,
                'replying_user_prompt': prompt,
                'replied_to_content': replied_to_text
            })
            print(f"[{time.strftime('%H:%M:%S')}] Reply recorded: {replying_message_index} -> {referenced_index}")

        st.rerun()

# --- Right Panel ---
with right_col:
    st.header(T["historyline_header"])

    # Button to export replies to CSV
    replies_df = pd.DataFrame(st.session_state.chats[st.session_state.current_chat_key]["replies_data"])
    if not replies_df.empty:
        csv = replies_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=T["export_replies_button"],
            data=csv,
            file_name=T["replies_csv_filename"],
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info(T["empty_chat_caption"]) # No replies, use this generic message for now

    st.divider()

    analysis_state = st.session_state.chats[st.session_state.current_chat_key]["analysis_state"]
    hierarchy = analysis_state['hierarchy']
    topic_titles = analysis_state['topic_titles']

    if hierarchy:
        rendered_main_topics = set()
        processed_revisits_toc = set()

        for item in hierarchy:
            main_topic_id = int(str(item['topic_id']).split('.')[0])
            level = item['level']

            anchor_index = item['index'] * 2

            if level == 'n':
                if main_topic_id not in rendered_main_topics:
                    title = topic_titles.get(main_topic_id, f"{T['topic_prefix']} {main_topic_id} ({T['topic_unknown_title']})")
                    if st.button(f"**{main_topic_id}. {title}**", key=f"toc_{st.session_state.current_chat_key}_{item['index']}_topic_n", use_container_width=True):
                        send_vqa_auto() # Trigger automatic VQA task on TOC button click
                        st.session_state.scroll_to_index = anchor_index
                        st.toast(T["updating_history_toast"], icon="📚") # Show "updating history" toast
                        st.rerun() # Rerun to apply scroll and potential VQA task addition
                    rendered_main_topics.add(main_topic_id)
            elif level == 'n (revisit)':
                revisit_identifier = (item['index'], item['topic_id'])
                if revisit_identifier not in processed_revisits_toc:
                    title_revisit = topic_titles.get(main_topic_id, f"{T['topic_prefix']} {main_topic_id} ({T['pending_title']})")
                    if st.button(f"{T['revisiting_topic']} {main_topic_id}: {title_revisit})", key=f"toc_{st.session_state.current_chat_key}_{item['index']}_revisit_topic", use_container_width=True):
                        send_vqa_auto() # Trigger automatic VQA task on TOC button click
                        st.session_state.scroll_to_index = anchor_index
                        st.toast(T["updating_history_toast"], icon="📚") # Show "updating history" toast
                        st.rerun() # Rerun to apply scroll and potential VQA task addition
                    processed_revisits_toc.add(revisit_identifier)
                
                display_text = item.get('vqa_gemini_title') # Prioritize VQA title if it exists
                if not display_text:
                    prompt_text_content = item['text']
                    if isinstance(prompt_text_content, str) and prompt_text_content.startswith("Pregunta VQA: "):
                        display_text = prompt_text_content[len("Pregunta VQA: "):].split(" Respuesta de la imagen:")[0].strip()
                    else:
                        display_text = prompt_text_content
                display_text = display_text if len(display_text) < 60 else display_text[:57] + "..."
                if st.button(f"{T['prompt_summary_prefix']}{display_text}\"", key=f"toc_detail_{st.session_state.current_chat_key}_{item['index']}_revisit_prompt", use_container_width=True):
                    send_vqa_auto() # Trigger automatic VQA task on TOC button click
                    st.session_state.scroll_to_index = anchor_index
                    st.toast(T["updating_history_toast"], icon="📚") # Show "updating history" toast
                    st.rerun() # Rerun to apply scroll and potential VQA task addition

            elif level == 'n.n':
                display_text = item.get('vqa_gemini_title')
                if not display_text:
                    prompt_text_content = item['text']
                    if isinstance(prompt_text_content, str) and prompt_text_content.startswith("Pregunta VQA: "):
                        display_text = prompt_text_content[len("Pregunta VQA: "):].split(" Respuesta de la imagen:")[0].strip()
                    else:
                        display_text = prompt_text_content
                display_text = display_text if len(display_text) < 60 else display_text[:57] + "..."
                if st.button(f"  {main_topic_id}.{str(item['topic_id']).split('.')[1]} {display_text}", key=f"toc_sub_{st.session_state.current_chat_key}_{item['index']}", use_container_width=True):
                    send_vqa_auto() # Trigger automatic VQA task on TOC button click
                    st.session_state.scroll_to_index = anchor_index
                    st.toast(T["updating_history_toast"], icon="📚") # Show "updating history" toast
                    st.rerun() # Rerun to apply scroll and potential VQA task addition
            elif level == 'n.n.n':
                display_text = item.get('vqa_gemini_title')
                if not display_text:
                    prompt_text_content = item['text']
                    if isinstance(prompt_text_content, str) and prompt_text_content.startswith("Pregunta VQA: "):
                        display_text = prompt_text_content[len("Pregunta VQA: "):].split(" Respuesta de la imagen:")[0].strip()
                    else:
                        display_text = prompt_text_content
                display_text = display_text if len(display_text) < 60 else display_text[:57] + "..."
                if st.button(f"{T['detail_prefix']}{display_text}\"", key=f"toc_subsub_{st.session_state.current_chat_key}_{item['index']}", use_container_width=True):
                    send_vqa_auto() # Trigger automatic VQA task on TOC button click
                    st.session_state.scroll_to_index = anchor_index
                    st.toast(T["updating_history_toast"], icon="📚") # Show "updating history" toast
                    st.rerun() # Rerun to apply scroll and potential VQA task addition
    else:
        st.caption(T["empty_chat_caption"])

# --- JavaScript Component for Scroll ---
if st.session_state.scroll_to_index is not None:
    index_to_scroll = st.session_state.pop('scroll_to_index')
    st.markdown(
        f"""
        <script>
            var el = window.parent.document.getElementById('msg-{index_to_scroll}');
            if (el) {{
                el.scrollIntoView({{behavior: 'smooth', block: 'center'}});
            }}
        </script>
        """,
        unsafe_allow_html=True, # ¡CRUCIAL! Permite la inyección de HTML/JS
    )