#!/usr/bin/env python3
"""
Agente di ricerca locale: Tavily Search + Qwen3 via Ollama.

Flusso (modalita' map-reduce, default):
  1. Prende una domanda dal terminale.
  2. Interroga Tavily per i risultati web (con testo integrale delle pagine).
  3. MAP: riassume OGNI fonte separatamente, leggendola per intero.
  4. REDUCE: fonde i riassunti in un'analisi strutturata con fonti citate [n].
  5. Salva la ricerca in un file Markdown.
  6. Permette domande di approfondimento (/f) sulle stesse fonti.

Ottimizzato per MacBook Air M1 / 8GB: le fonti vengono elaborate UNA ALLA VOLTA
(map-reduce), cosi' la RAM non accumula; "thinking" di Qwen3 disabilitato.
"""

import os
import re
import sys
import json
import time
import itertools
import threading
import textwrap
from datetime import datetime

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from dotenv import load_dotenv
    # Carica il .env che sta ACCANTO a questo script, non nella cartella corrente:
    # così l'agente funziona anche se lanciato da una directory diversa (es. ~).
    load_dotenv(os.path.join(SCRIPT_DIR, ".env"))
except ImportError:
    pass  # python-dotenv è opzionale: si può usare anche la variabile d'ambiente

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
TAVILY_URL = "https://api.tavily.com/search"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


# Tuning per 8GB di RAM.
NUM_RESULTS = int(os.getenv("RESEARCH_NUM_RESULTS", "5"))
SNIPPET_CHARS = 320          # taglio dello snippet quando NON si usa il contenuto pieno
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))  # finestra di contesto
TEMPERATURE = 0.3            # poca fantasia: vogliamo aderenza alle fonti

# Map-reduce: ogni fonte viene riassunta da sola, poi i riassunti vengono uniti.
# E' la modalita' "deep" che permette di leggere articoli INTERI e piu' fonti
# senza saturare la RAM, perche' le chiamate al modello sono sequenziali.
MAP_REDUCE = _flag("RESEARCH_MAP_REDUCE", "1")
MAP_CONTENT_CHARS = int(os.getenv("RESEARCH_MAP_CONTENT_CHARS", "6000"))  # per fonte, nel map

# Lettura del contenuto completo delle pagine anche in modalita' semplice (no map-reduce).
FULL_CONTENT = _flag("RESEARCH_FULL_CONTENT", "0")
CONTENT_CHARS = int(os.getenv("RESEARCH_CONTENT_CHARS", "2500"))

# Salvataggio delle ricerche in Markdown.
SAVE_ENABLED = _flag("RESEARCH_SAVE", "1")
SAVE_DIR = os.getenv("RESEARCH_SAVE_DIR", os.path.join(SCRIPT_DIR, "ricerche"))

# Con map-reduce serve sempre il testo integrale delle pagine.
NEED_RAW = FULL_CONTENT or MAP_REDUCE


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def die(msg: str, code: int = 1):
    print(f"\n❌ {msg}\n", file=sys.stderr)
    sys.exit(code)


class Spinner:
    """Indicatore animato per le attese (chiamate al modello). Si attiva solo se
    l'output è un terminale: se rediretto su file/pipe non sporca i log."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, text: str = ""):
        self.text = text
        self._stop = threading.Event()
        self._thread = None
        self._active = sys.stdout.isatty()

    def __enter__(self):
        if self._active:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        else:
            print(self.text)  # niente animazione: stampa una riga statica
        return self

    def _spin(self):
        for ch in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            print(f"\r{ch} {self.text}", end="", flush=True)
            time.sleep(0.1)

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()
            # pulisce la riga dello spinner
            print("\r" + " " * (len(self.text) + 2) + "\r", end="", flush=True)


def strip_think(text: str) -> str:
    """Rimuove un blocco <think>...</think> iniziale da una risposta completa."""
    s = text.lstrip()
    if s.startswith("<think>"):
        end = text.find("</think>")
        return text[end + len("</think>"):].lstrip() if end != -1 else ""
    return text


def clean_stream(acc: str) -> str:
    """Versione per lo streaming: dato il testo cumulativo, restituisce la porzione
    sicura da stampare, nascondendo il blocco <think> finche' non e' chiuso."""
    s = acc.lstrip()
    if s.startswith("<think>"):
        end = acc.find("</think>")
        if end == -1:
            return ""
        return acc[end + len("</think>"):].lstrip()
    if s and "<think>".startswith(s):
        return ""  # potrebbe diventare un tag <think>: aspetta altri chunk
    return acc


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


# ---------------------------------------------------------------------------
# Ricerca web (Tavily)
# ---------------------------------------------------------------------------
def tavily_search(query: str, count: int = NUM_RESULTS):
    """Cerca su Tavily e restituisce una lista di dict {title, url, content}."""
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
        "include_raw_content": NEED_RAW,
    }
    try:
        r = requests.post(TAVILY_URL, json=payload, timeout=20)
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
        # 'content' = snippet breve di Tavily; 'raw_content' = testo integrale pagina.
        out.append({
            "title": item.get("title", "").strip(),
            "url": item.get("url", "").strip(),
            "content": (item.get("content") or "").strip(),
            "raw_content": (item.get("raw_content") or "").strip(),
        })
    return out


# ---------------------------------------------------------------------------
# Dialogo con il modello (Ollama)
# ---------------------------------------------------------------------------
def chat(messages, stream=False, num_predict=None, spinner_text=""):
    """Chiamata a Ollama /api/chat.
    - stream=False: ritorna il testo completo (ripulito dal <think>), senza stampare.
                    Mostra uno spinner durante l'attesa se spinner_text è dato.
    - stream=True : stampa la risposta mentre arriva e ritorna il testo completo.
                    Mostra uno spinner finché non arriva il primo token.
    """
    options = {"num_ctx": NUM_CTX, "temperature": TEMPERATURE}
    if num_predict:
        options["num_predict"] = num_predict
    payload = {"model": MODEL, "messages": messages, "stream": stream, "options": options}

    try:
        if not stream:
            with Spinner(spinner_text) if spinner_text else _nullctx():
                r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
                r.raise_for_status()
            return strip_think(r.json().get("message", {}).get("content", ""))

        with requests.post(f"{OLLAMA_URL}/api/chat", json=payload,
                           stream=True, timeout=300) as r:
            r.raise_for_status()
            acc, emitted = "", 0
            spinner = Spinner(spinner_text) if spinner_text else None
            if spinner:
                spinner.__enter__()
            for line in r.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                piece = obj.get("message", {}).get("content", "")
                if piece:
                    acc += piece
                    safe = clean_stream(acc)
                    if len(safe) > emitted:
                        if spinner:                 # primo testo: spegni lo spinner
                            spinner.__exit__()
                            spinner = None
                        print(safe[emitted:], end="", flush=True)
                        emitted = len(safe)
                if obj.get("done"):
                    break
            if spinner:                              # nessun testo utile arrivato
                spinner.__exit__()
            print()
            return strip_think(acc)
    except requests.exceptions.ConnectionError:
        die("Connessione a Ollama persa durante la generazione. È ancora avviato?")
    except requests.exceptions.RequestException as e:
        die(f"Errore durante la generazione: {e}")


class _nullctx:
    """Context manager neutro: usato quando non serve lo spinner."""
    def __enter__(self): return self
    def __exit__(self, *exc): return False


# --- Prompt -----------------------------------------------------------------
MAP_SYSTEM = (
    "/no_think\n"
    "Sei un assistente di ricerca. Ti viene data UNA fonte e una domanda. "
    "Estrai SOLO le informazioni della fonte utili a rispondere, in forma di punti "
    "concisi e fattuali (max 6 punti). Non inventare nulla. "
    "Se la fonte non è pertinente, scrivi soltanto: NON PERTINENTE."
)

REDUCE_SYSTEM = (
    "/no_think\n"
    "Sei un assistente di ricerca. Rispondi SEMPRE nella lingua della domanda. "
    "Usa esclusivamente i riassunti numerati forniti: non inventare fatti. "
    "Cita ogni affermazione con il numero della fonte tra parentesi quadre, es. [1]. "
    "Se le fonti non bastano, dillo esplicitamente. Sii conciso e fattuale."
)

ANALYSIS_FORMAT = """Produci un'analisi strutturata in italiano con questo formato:

## Risposta sintetica
(2-3 frasi che rispondono direttamente, con citazioni [n])

## Approfondimento
(punti chiave, ognuno con la sua citazione [n])

## Limiti / cose da verificare
(eventuali lacune nelle fonti)

## Fonti
(elenco numerato: [n] Titolo — URL)"""

FOLLOWUP_SYSTEM = REDUCE_SYSTEM  # stesse regole: cita le stesse fonti [n]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def map_summaries(question: str, results):
    """MAP: riassume ogni fonte separatamente. Ritorna una lista di stringhe."""
    summaries = []
    n = len(results)
    for i, res in enumerate(results, 1):
        title = res["title"] or res["url"]
        body = (res["raw_content"] or res["content"])[:MAP_CONTENT_CHARS]
        if not body:
            print(f"   [{i}/{n}] (fonte vuota, saltata): {title[:55]}")
            summaries.append("NON PERTINENTE")
            continue
        msg = [
            {"role": "system", "content": MAP_SYSTEM},
            {"role": "user", "content": f"Domanda: {question}\n\nFonte:\n{title}\n{body}\n\nRiassunto pertinente:"},
        ]
        label = f"  [{i}/{n}] Riassumo: {title[:55]}"
        summaries.append(chat(msg, stream=False, num_predict=400, spinner_text=label).strip())
        print(f"   ✓ [{i}/{n}] {title[:55]}")
    return summaries


def build_numbered(results, summaries=None):
    """Costruisce il blocco numerato [n] Titolo / testo / URL da dare al reduce."""
    lines = []
    for i, res in enumerate(results, 1):
        if summaries is not None:
            text = summaries[i - 1]
        elif FULL_CONTENT:
            text = (res["raw_content"] or res["content"])[:CONTENT_CHARS]
        else:
            text = res["content"][:SNIPPET_CHARS]
        lines.append(f"[{i}] {res['title']}\n{text}\nURL: {res['url']}")
    return "\n\n".join(lines)


def synthesize(question: str, context: str):
    """REDUCE: dai riassunti/fonti produce l'analisi finale (in streaming)."""
    msg = [
        {"role": "system", "content": REDUCE_SYSTEM},
        {"role": "user", "content":
            f"Domanda: {question}\n\nRiassunti delle fonti:\n{context}\n\n{ANALYSIS_FORMAT}"},
    ]
    print()
    return chat(msg, stream=True, spinner_text="Elaboro la sintesi finale...")


def slugify(text: str, maxlen: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:maxlen].strip("-") or "ricerca"


def save_markdown(question: str, analysis: str, results) -> str:
    """Salva la ricerca in un file Markdown e ritorna il percorso."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    ts = datetime.now()
    path = os.path.join(SAVE_DIR, f"{ts:%Y-%m-%d_%H%M%S}_{slugify(question)}.md")
    sources = "\n".join(f"{i}. [{r['title']}]({r['url']})"
                        for i, r in enumerate(results, 1))
    doc = (
        f"# {question}\n\n"
        f"*Ricerca del {ts:%d/%m/%Y %H:%M} — modello {MODEL}*\n\n"
        f"{analysis}\n\n"
        f"---\n\n## Fonti consultate\n{sources}\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path


def research(question: str, session: dict):
    """Esegue una ricerca completa e aggiorna lo stato di sessione."""
    print(f"\n🔎 Cerco: {question}")
    results = tavily_search(question)
    if not results:
        print("⚠️  Nessun risultato da Tavily per questa query. Prova a riformulare.")
        return

    if MAP_REDUCE:
        print(f"   Trovate {len(results)} fonti. Leggo e riassumo una alla volta...")
        summaries = map_summaries(question, results)
        context = build_numbered(results, summaries)
    else:
        print(f"   Trovate {len(results)} fonti. Genero l'analisi con {MODEL}...")
        context = build_numbered(results)

    print(f"\n🧩 Sintesi finale con {MODEL}:")
    analysis = synthesize(question, context)

    # Stato per i follow-up
    session["question"] = question
    session["context"] = context
    session["analysis"] = analysis
    session["results"] = results

    if SAVE_ENABLED and analysis.strip():
        try:
            path = save_markdown(question, analysis, results)
            print(f"\n💾 Salvato in: {path}")
        except OSError as e:
            print(f"\n⚠️  Impossibile salvare il file Markdown: {e}")


def follow_up(question: str, session: dict):
    """Risponde a un approfondimento usando le fonti della ricerca precedente."""
    if not session.get("context"):
        print("⚠️  Nessuna ricerca precedente. Fai prima una domanda normale.")
        return
    print(f"\n↪️  Approfondimento (stesse fonti): {question}")
    msg = [
        {"role": "system", "content": FOLLOWUP_SYSTEM},
        {"role": "user", "content": (
            f"Ricerca precedente su: {session['question']}\n\n"
            f"Fonti:\n{session['context']}\n\n"
            f"Analisi precedente:\n{session['analysis']}\n\n"
            f"Domanda di approfondimento: {question}\n"
            "Rispondi in italiano usando le stesse fonti, citandole con [n].")},
    ]
    print()
    chat(msg, stream=True, spinner_text="Elaboro l'approfondimento...")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    check_ollama()
    session = {}
    args = sys.argv[1:]

    if args:
        research(" ".join(args), session)
        return

    mode = "map-reduce" if MAP_REDUCE else "semplice"
    print(textwrap.dedent(f"""\
        🧠 Agente di ricerca locale ({MODEL} + Tavily, modalità {mode})
        • scrivi una domanda per una nuova ricerca
        • /f <domanda>  → approfondisci sulle stesse fonti (senza nuova ricerca)
        • exit          → esci
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
        if q.startswith("/f ") or q.startswith("/seguito "):
            follow_up(q.split(" ", 1)[1].strip(), session)
        else:
            research(q, session)


if __name__ == "__main__":
    main()
