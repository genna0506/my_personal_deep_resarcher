#!/usr/bin/env python3
"""
Agente di ricerca locale: Brave Search + Qwen3 via Ollama.

Flusso:
  1. Prende una domanda dal terminale.
  2. Interroga Brave Search API (gratuita) per i risultati web.
  3. Passa i risultati come contesto a qwen3:4b via Ollama (locale, gratis).
  4. Stampa un'analisi strutturata con fonti citate [n].

Ottimizzato per MacBook Air M1 / 8GB: contesto breve, num_ctx ridotto,
"thinking" disabilitato per risparmiare RAM e tempo.
"""

import os
import sys
import json
import textwrap

import requests

try:
    from dotenv import load_dotenv
    # Carica il .env che sta ACCANTO a questo script, non nella cartella corrente:
    # così l'agente funziona anche se lanciato da una directory diversa (es. ~).
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass  # python-dotenv è opzionale: si può usare anche la variabile d'ambiente

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
TAVILY_URL = "https://api.tavily.com/search"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

# Tuning per 8GB di RAM: pochi risultati, snippet corti, finestra di contesto piccola.
NUM_RESULTS = int(os.getenv("RESEARCH_NUM_RESULTS", "5"))
SNIPPET_CHARS = 320          # taglio di ogni snippet per non gonfiare il contesto
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))  # KV cache contenuta
TEMPERATURE = 0.3            # poca fantasia: vogliamo aderenza alle fonti

# Lettura del CONTENUTO COMPLETO delle pagine (non solo lo snippet di Tavily).
# Su 8GB va usato con POCHE fonti (3-4) perche' il testo integrale e' molto lungo:
# ogni pagina viene comunque troncata a CONTENT_CHARS per non saturare la RAM.
FULL_CONTENT = os.getenv("RESEARCH_FULL_CONTENT", "0").strip() in {"1", "true", "yes"}
CONTENT_CHARS = int(os.getenv("RESEARCH_CONTENT_CHARS", "2500"))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def die(msg: str, code: int = 1):
    print(f"\n❌ {msg}\n", file=sys.stderr)
    sys.exit(code)


def check_ollama():
    """Verifica che il server Ollama sia attivo e che il modello sia presente."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        die(
            "Ollama non è in esecuzione.\n"
            "   Avvialo in un altro terminale con:\n"
            '     OLLAMA_FLASH_ATTENTION="1" OLLAMA_KV_CACHE_TYPE="q8_0" ollama serve\n'
            "   (oppure: brew services start ollama)"
        )
    except requests.exceptions.RequestException as e:
        die(f"Errore nel contattare Ollama su {OLLAMA_URL}: {e}")

    models = [m["name"] for m in r.json().get("models", [])]
    if not any(m.split(":")[0] == MODEL.split(":")[0] for m in models):
        die(
            f"Il modello '{MODEL}' non risulta installato.\n"
            f"   Scaricalo con:  ollama pull {MODEL}\n"
            f"   Modelli presenti: {', '.join(models) or '(nessuno)'}"
        )


def tavily_search(query: str, count: int = NUM_RESULTS):
    """Cerca su Tavily e restituisce una lista di dict {title, url, description}."""
    if not TAVILY_API_KEY:
        die(
            "Manca la TAVILY_API_KEY.\n"
            "   Ottieni una chiave gratuita (1000 ricerche/mese) su https://app.tavily.com\n"
            "   poi esportala:   export TAVILY_API_KEY='tvly-...'\n"
            "   oppure mettila in un file .env (vedi README)."
        )

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": count,
        "include_answer": False,
        "search_depth": "basic",
        "include_raw_content": FULL_CONTENT,  # testo completo delle pagine se attivo
    }
    try:
        r = requests.post(TAVILY_URL, json=payload, timeout=15)
    except requests.exceptions.RequestException as e:
        die(f"Errore di rete su Tavily: {e}")

    if r.status_code == 401:
        die("Tavily: chiave API non valida (401). Controlla la TAVILY_API_KEY.")
    if r.status_code == 429:
        die("Tavily: limite di richieste raggiunto (429). Il piano gratuito "
            "consente 1000 ricerche/mese. Riprova domani o aggiorna il piano.")
    if r.status_code != 200:
        die(f"Tavily ha risposto con HTTP {r.status_code}: {r.text[:200]}")

    results = r.json().get("results", [])
    out = []
    for item in results[:count]:
        # Con FULL_CONTENT usa il testo integrale; se Tavily non riesce a estrarlo
        # (raw_content nullo) ripiega sullo snippet breve, cosi' non si perde la fonte.
        if FULL_CONTENT:
            text = (item.get("raw_content") or item.get("content") or "").strip()
        else:
            text = (item.get("content") or "").strip()
        out.append({
            "title": item.get("title", "").strip(),
            "url": item.get("url", "").strip(),
            "description": text,
        })
    return out


def build_context(results):
    """Costruisce il blocco di contesto numerato da dare al modello."""
    limit = CONTENT_CHARS if FULL_CONTENT else SNIPPET_CHARS
    lines = []
    for i, res in enumerate(results, 1):
        snippet = res["description"][:limit]
        lines.append(f"[{i}] {res['title']}\n    {snippet}\n    URL: {res['url']}")
    return "\n".join(lines)


# "/no_think" è lo switch soft di Qwen3 per disattivare il ragionamento esplicito:
# fondamentale su 8GB per non sprecare tempo/token. Il parametro API "think:false"
# non è affidabile su tutte le versioni di Ollama, quindi usiamo questo + strip.
SYSTEM_PROMPT = (
    "/no_think\n"
    "Sei un assistente di ricerca. Rispondi SEMPRE nella lingua della domanda. "
    "Usa esclusivamente le fonti numerate fornite: non inventare fatti. "
    "Cita ogni affermazione con il numero della fonte tra parentesi quadre, es. [1]. "
    "Se le fonti non bastano a rispondere, dillo esplicitamente. Sii conciso e fattuale."
)

USER_TEMPLATE = """Domanda: {question}

Fonti disponibili:
{context}

Produci un'analisi strutturata in italiano con questo formato:

## Risposta sintetica
(2-3 frasi che rispondono direttamente, con citazioni [n])

## Approfondimento
(punti chiave, ognuno con la sua citazione [n])

## Limiti / cose da verificare
(eventuali lacune nelle fonti)

## Fonti
(elenco numerato: [n] Titolo — URL)
"""


def clean_stream(acc: str) -> str:
    """Restituisce la porzione di testo accumulato sicura da stampare, rimuovendo
    il blocco <think>...</think> che Qwen3 può emettere all'inizio della risposta.

    Pensata per lo streaming: viene richiamata a ogni chunk sul testo cumulativo e
    cresce in modo monotòno, così si stampa solo il delta. Finché il ragionamento
    non è chiuso (o un eventuale tag iniziale è incompleto) restituisce "".
    """
    s = acc.lstrip()
    if s.startswith("<think>"):
        end = acc.find("</think>")
        if end == -1:
            return ""  # ancora dentro il ragionamento: non stampare nulla
        return acc[end + len("</think>"):].lstrip()
    if s and "<think>".startswith(s):
        return ""  # potrebbe diventare un tag <think>: aspetta altri chunk
    return acc


def ask_model(question: str, context: str):
    """Invia la richiesta a Ollama in streaming e stampa la risposta a schermo."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                question=question, context=context)},
        ],
        "stream": True,
        "options": {
            "num_ctx": NUM_CTX,
            "temperature": TEMPERATURE,
        },
    }

    try:
        with requests.post(f"{OLLAMA_URL}/api/chat", json=payload,
                           stream=True, timeout=300) as r:
            r.raise_for_status()
            print()
            acc, emitted = "", 0
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    acc += piece
                    safe = clean_stream(acc)
                    if len(safe) > emitted:
                        print(safe[emitted:], end="", flush=True)
                        emitted = len(safe)
                if chunk.get("done"):
                    break
            print("\n")
    except requests.exceptions.ConnectionError:
        die("Connessione a Ollama persa durante la generazione. È ancora avviato?")
    except requests.exceptions.RequestException as e:
        die(f"Errore durante la generazione: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def research(question: str):
    print(f"\n🔎 Cerco: {question}")
    results = tavily_search(question)
    if not results:
        die("Nessun risultato da Brave per questa query. Prova a riformulare.")
    print(f"   Trovate {len(results)} fonti. Genero l'analisi con {MODEL}...")
    context = build_context(results)
    ask_model(question, context)


def main():
    check_ollama()
    args = sys.argv[1:]

    if args:
        # Modalità una-tantum:  python main.py "la mia domanda"
        research(" ".join(args))
        return

    # Modalità interattiva
    print(textwrap.dedent(f"""\
        🧠 Agente di ricerca locale ({MODEL} + Tavily Search)
        Scrivi una domanda, oppure 'exit' per uscire.
    """))
    while True:
        try:
            q = input("❓ > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCiao!")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "esci"}:
            print("Ciao!")
            break
        research(q)


if __name__ == "__main__":
    main()
