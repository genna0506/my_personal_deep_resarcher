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
import hashlib
import itertools
import threading
import textwrap
from datetime import datetime
from urllib.parse import urlparse

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

# Multi-query: il modello genera piu' ricerche diverse dalla stessa domanda, i
# risultati vengono uniti, deduplicati e ordinati per rilevanza. Migliora copertura
# e qualita', e permette di raccogliere piu' fonti del tetto (20) di una sola ricerca.
# Costo: ogni sotto-query e' UNA ricerca Tavily (3 query = 3 ricerche del budget).
MULTI_QUERY = _flag("RESEARCH_MULTI_QUERY", "1")
NUM_QUERIES = int(os.getenv("RESEARCH_NUM_QUERIES", "3"))

# Qualita' delle fonti.
# - search_depth "advanced" = ranking ed estrazione migliori (costa 2 crediti vs 1).
# - score minimo = scarta le fonti sotto questa rilevanza (0 = nessun filtro).
# - domini esclusi = via i forum/social piu' rumorosi (modificabile/svuotabile).
SEARCH_DEPTH = os.getenv("RESEARCH_SEARCH_DEPTH", "advanced").strip()
MIN_SCORE = float(os.getenv("RESEARCH_MIN_SCORE", "0"))
EXCLUDE_DOMAINS = [d.strip() for d in os.getenv(
    "RESEARCH_EXCLUDE_DOMAINS",
    "reddit.com,quora.com,x.com,twitter.com,facebook.com,pinterest.com,tripadvisor.com",
).split(",") if d.strip()]

# Filtro forum/community via PATTERN (non lista fissa): cattura anche i forum
# italiani non elencati sopra (es. finanzaonline, forumfree...). Per loro natura i
# forum sono opinioni non verificate, da escludere in una ricerca rigorosa.
FILTER_FORUMS = _flag("RESEARCH_FILTER_FORUMS", "1")

# Gating critico: scarta le fonti che l'analista marca come non pertinenti, così non
# diluiscono la sintesi finale (qualità > quantità).
DROP_IRRELEVANT = _flag("RESEARCH_DROP_IRRELEVANT", "1")

# Diversità: massimo numero di pagine tenute per singolo dominio.
MAX_PER_DOMAIN = int(os.getenv("RESEARCH_MAX_PER_DOMAIN", "2"))

# Ricerca bilingue: aggiunge una query tradotta in inglese (le fonti autorevoli su
# tech/finanza/scienza sono spesso in inglese). Traduzione gratuita via deep-translator.
BILINGUAL = _flag("RESEARCH_BILINGUAL", "1")

# Bonus di rilevanza dato a fonti autorevoli (accademiche/ufficiali): le fa leggere
# per prime rispetto a blog di pari rilevanza testuale.
AUTHORITY_BONUS = float(os.getenv("RESEARCH_AUTHORITY_BONUS", "0.15"))

# Loop iterativo di approfondimento: dopo una sintesi, l'agente individua le LACUNE
# e fa ricerche mirate per colmarle, poi rifonde. MAX_ROUNDS = 1 (iniziale) + N giri.
MAX_ROUNDS = int(os.getenv("RESEARCH_MAX_ROUNDS", "3"))
GAP_QUERIES = int(os.getenv("RESEARCH_GAP_QUERIES", "2"))  # lacune inseguite per giro

# Auto-verifica: passo finale in cui il modello confronta la propria analisi con le
# fonti già raccolte e segnala le affermazioni non supportate (non riscarica nulla).
VERIFY = _flag("RESEARCH_VERIFY", "1")

# Consapevolezza temporale: piccolo bonus di ranking alle fonti recenti (quando Tavily
# fornisce la data). Mite, per non scavalcare la rilevanza.
RECENCY_BONUS = float(os.getenv("RESEARCH_RECENCY_BONUS", "0.10"))

# Modalità accademica: privilegia paper/studi/fonti dense. Si attiva con tag espliciti
# (#paper #studi #scientifico ...) o da sola se la domanda contiene segnali accademici.
ACADEMIC_TAGS = {"#paper", "#papers", "#studi", "#studio", "#scientifico",
                 "#accademico", "#research", "#science"}
ACADEMIC_HINTS = ("studio", "studi ", "ricerca scientifica", "correlazione", "evidenza",
                  "evidenze", "meta-analisi", "metanalisi", "campione", "campioni",
                  "peer review", "paper", "pubblicazione", "statistic", "r=", "p<",
                  "trial", "randomizzato", "letteratura scientifica")
# Repository accademici: in modalità accademica ricevono un bonus di autorevolezza forte
# e vengono aggiunti come angolazione di ricerca dedicata.
ACADEMIC_HOSTS = ("arxiv.org", "ncbi.nlm.nih.gov", "pubmed", "scholar.google",
                  "researchgate.net", "semanticscholar.org", "sciencedirect.com",
                  "springer.com", "nature.com", "jstor.org", "ssrn.com",
                  "doaj.org", "plos.org", "frontiersin.org", "mdpi.com")
ACADEMIC_BONUS = float(os.getenv("RESEARCH_ACADEMIC_BONUS", "0.30"))

# Cache delle ricerche Tavily: evita di ripagare/riscaricare query identiche (entro TTL).
CACHE_ENABLED = _flag("RESEARCH_CACHE", "1")
CACHE_TTL_H = float(os.getenv("RESEARCH_CACHE_TTL_H", "24"))  # ore di validità
CACHE_DIR = os.path.join(SCRIPT_DIR, ".cache")

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
# Cache delle ricerche
# ---------------------------------------------------------------------------
def _cache_path(key: str) -> str:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{h}.json")


def cache_get(key: str):
    """Ritorna i risultati in cache se presenti e non scaduti, altrimenti None."""
    if not CACHE_ENABLED:
        return None
    path = _cache_path(key)
    try:
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
    except (OSError, ValueError):
        return None
    if time.time() - blob.get("ts", 0) > CACHE_TTL_H * 3600:
        return None
    return blob.get("results")


def cache_put(key: str, results):
    if not CACHE_ENABLED:
        return
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(_cache_path(key), "w", encoding="utf-8") as f:
            json.dump({"ts": time.time(), "results": results}, f)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Ricerca web (Tavily)
# ---------------------------------------------------------------------------
def tavily_search(query: str, count: int = NUM_RESULTS, include_domains=None):
    """Cerca su Tavily e restituisce una lista di dict {title, url, content, ...}.
    Usa la cache su disco per non ripagare query identiche entro il TTL."""
    if not TAVILY_API_KEY:
        die(
            "Manca la TAVILY_API_KEY.\n"
            "   Ottieni una chiave gratuita (1000 ricerche/mese) su https://app.tavily.com\n"
            "   poi esportala:   export TAVILY_API_KEY='tvly-...'\n"
            "   oppure mettila in un file .env (vedi README)."
        )

    cache_key = f"{query}|{count}|{SEARCH_DEPTH}|{NEED_RAW}|{include_domains}|{EXCLUDE_DOMAINS}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": count,
        "include_answer": False,
        "search_depth": SEARCH_DEPTH,
        "include_raw_content": NEED_RAW,
    }
    if EXCLUDE_DOMAINS:
        payload["exclude_domains"] = EXCLUDE_DOMAINS
    if include_domains:
        payload["include_domains"] = include_domains
    try:
        r = requests.post(TAVILY_URL, json=payload, timeout=30)
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
        # 'content' = snippet breve; 'raw_content' = testo integrale; 'score' = rilevanza;
        # 'published_date' = data di pubblicazione (quando Tavily riesce a stimarla).
        out.append({
            "title": item.get("title", "").strip(),
            "url": item.get("url", "").strip(),
            "content": (item.get("content") or "").strip(),
            "raw_content": (item.get("raw_content") or "").strip(),
            "score": float(item.get("score", 0) or 0),
            "date": (item.get("published_date") or "").strip(),
        })
    cache_put(cache_key, out)
    return out


# Marcatori che introducono un VINCOLO dell'utente (da NON cercare, ma da usare come
# criterio di valutazione critica delle fonti).
CONSTRAINT_MARKERS = ["escludi", "esclusi", "escludendo", "tranne", "senza ",
                      "eccetto", "evita", "non includere", "ad eccezione",
                      "a meno che", "purché", "purche"]

# Parole da NON usare come keyword: stopword italiane, parole interrogative e soprattutto
# le esche da "listicle/SEO" (migliori, best, top...) che attirano classifiche promozionali.
STOPWORDS = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "a", "da", "in", "con",
    "su", "per", "tra", "fra", "del", "dello", "della", "dei", "degli", "delle", "al",
    "allo", "alla", "ai", "agli", "alle", "dal", "dalla", "nel", "nella", "sul", "sulla",
    "e", "ed", "o", "oppure", "ma", "che", "chi", "cui", "come", "quale", "quali", "quanto",
    "quanti", "quando", "dove", "perché", "perche", "cosa", "ci", "si", "non", "più", "piu",
    "meno", "molto", "poco", "tanto", "sono", "è", "e'", "essere", "ha", "hanno", "avere",
    "ho", "fa", "fare", "attualmente", "adesso", "oggi", "possibile", "possibili", "vorrei",
    "voglio", "mi", "ti", "se", "dei", "alta", "alto", "alte", "alti", "alla",
    # esche da classifica/SEO:
    "migliori", "migliore", "miglior", "best", "top", "classifica", "classifiche",
    "guida", "guide", "consigli", "consigliati",
    # rumore tipico delle frasi sulle "lacune" (usato per costruire le query di gap):
    "fonti", "fonte", "specificano", "specifica", "indicano", "indica", "manca",
    "mancano", "mancanti", "forniscono", "fornisce", "riportano", "riporta", "chiaro",
    "chiari", "esplicitamente", "sufficienti", "quindi", "tuttavia", "inoltre",
}


def split_constraints(question: str):
    """Separa la domanda 'da cercare' dai VINCOLI dell'utente (es. 'escludi ...').
    Ritorna (core, constraints): i vincoli vanno alla valutazione, non alla ricerca."""
    low = question.lower()
    positions = [low.find(m) for m in CONSTRAINT_MARKERS if m in low]
    positions = [p for p in positions if p >= 0]
    if not positions:
        return question.strip(), ""
    cut = min(positions)
    return question[:cut].strip(" ,.;:?!"), question[cut:].strip(" ,.;:?!")


def extract_keywords(text: str) -> str:
    """Riduce una domanda in linguaggio naturale a poche parole chiave per la ricerca,
    togliendo stopword e le esche da listicle. Mantiene l'ordine, niente duplicati."""
    words = re.findall(r"[0-9a-zàèéìòóùç]+", text.lower())
    kept, seen = [], set()
    for w in words:
        if len(w) <= 2 or w in STOPWORDS or w in seen:
            continue
        seen.add(w)
        kept.append(w)
    return " ".join(kept)


def detect_academic(question: str):
    """Rileva se servono fonti dense/accademiche. Si attiva con tag espliciti
    (#paper #studi ...) oppure da sola se la domanda contiene segnali accademici.
    Ritorna (domanda_ripulita_dai_tag, is_academic)."""
    low = question.lower()
    tags = set(re.findall(r"#\w+", low))
    is_acad = bool(tags & {t.lower() for t in ACADEMIC_TAGS})
    clean = re.sub(r"#\w+", "", question).strip()
    if not is_acad:
        is_acad = any(h in low for h in ACADEMIC_HINTS)
    return clean, is_acad


def source_year(date_str: str):
    """Estrae l'anno (int) da una data Tavily tipo '2024-03-01', o None."""
    m = re.search(r"(19|20)\d{2}", date_str or "")
    return int(m.group(0)) if m else None


def recency_bonus(date_str: str) -> float:
    """Bonus mite alle fonti recenti (0 se data assente o vecchia)."""
    y = source_year(date_str)
    if not y:
        return 0.0
    now = datetime.now().year
    if y >= now:
        return RECENCY_BONUS
    if y == now - 1:
        return RECENCY_BONUS * 0.5
    return 0.0


def search_queries(core_question: str, n: int = NUM_QUERIES, academic: bool = False):
    """Costruisce le query di ricerca (DETERMINISTICO, qwen3 va in loop su questo).
    Parte dalle parole chiave pulite; aggiunge la variante inglese e, in modalità
    accademica, una query mirata a paper/studi. Configurabile via RESEARCH_QUERY_MODIFIERS."""
    base = extract_keywords(core_question) or core_question.strip()
    en = translate_en(base) if (BILINGUAL or academic) else ""
    candidates = [base]
    if academic:                        # priorità a una query che punta su paper/studi
        candidates.append(f"{en or base} peer reviewed study")
    if BILINGUAL and en and en.lower() != base.lower():
        candidates.append(en)
    year = str(datetime.now().year)
    mods = [m.strip() for m in os.getenv("RESEARCH_QUERY_MODIFIERS", year).split(",") if m.strip()]
    candidates += [f"{base} {m}" for m in mods]
    out, seen = [], set()
    for q in candidates:
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
        if len(out) >= n:
            break
    return out


def translate_en(text: str) -> str:
    """Traduce in inglese per ampliare il bacino di fonti. Best-effort: se la libreria
    o la rete non rispondono, ritorna '' e la ricerca resta solo in italiano."""
    try:
        from deep_translator import GoogleTranslator
        return (GoogleTranslator(source="auto", target="en").translate(text) or "").strip()
    except Exception:
        return ""


# Domini/TLD autorevoli: accademici, enti, istituzioni, editori scientifici.
AUTHORITY_TLDS = (".edu", ".gov", ".int", ".ac.uk", ".edu.au")
AUTHORITY_HOSTS = ("wikipedia.org", "who.int", "europa.eu", "ecb.europa.eu", "istat.it",
                   "bancaditalia.it", "oecd.org", "imf.org", "nih.gov", "ncbi.nlm.nih.gov",
                   "nature.com", "sciencedirect.com", "springer.com", "jstor.org",
                   "scholar.google", "pubmed", "apa.org", ".unibo.it", ".unimi.it",
                   "treccani.it", "consob.it")


def authority_bonus(url: str, academic: bool = False) -> float:
    """Bonus di score per fonti autorevoli (0 se è un sito qualunque). In modalità
    accademica i repository di paper/studi ricevono un bonus più forte."""
    host = urlparse(url).netloc.lower()
    if academic and any(h in host for h in ACADEMIC_HOSTS):
        return ACADEMIC_BONUS
    if host.endswith(AUTHORITY_TLDS) or any(h in host for h in AUTHORITY_HOSTS):
        return AUTHORITY_BONUS
    return 0.0


# Host di forum/community noti che non contengono "forum" nel nome.
FORUM_HOSTS = ("finanzaonline.com", "forumfree.it", "forumcommunity.net",
               "freeforumzone.com", "mtb-forum.it", "bdc-forum.it",
               "stackexchange.com", "stackoverflow.com")


def is_forumish(url: str) -> bool:
    """True se l'URL è verosimilmente un forum/community (discussioni non verificate)."""
    host = urlparse(url).netloc.lower()
    if any(h in host for h in FORUM_HOSTS):
        return True
    return "forum" in host or "community" in host or host.startswith("forum.")


def search_many(queries, exclude=()):
    """Esegue le ricerche date, unisce e deduplica per URL (saltando gli URL già
    visti in 'exclude'), tenendo per ogni URL la copia con score più alto."""
    merged = {}
    for q in queries:
        for res in tavily_search(q, NUM_RESULTS):
            url = res["url"]
            if not url or url in exclude:
                continue
            if url not in merged or res["score"] > merged[url]["score"]:
                merged[url] = res
    return list(merged.values())


def rank_results(results, academic: bool = False):
    """Filtra forum, applica i bonus (autorevolezza + recency), ordina e impone la
    diversità di dominio. Ritorna le migliori NUM_RESULTS fonti."""
    if FILTER_FORUMS:
        before = len(results)
        results = [r for r in results if not is_forumish(r["url"])]
        skipped = before - len(results)
        if skipped:
            print(f"   🚫 Scartati {skipped} risultati da forum/community.")

    # score effettivo = rilevanza Tavily + autorevolezza + recency
    for r in results:
        r["rank"] = (r["score"] + authority_bonus(r["url"], academic)
                     + recency_bonus(r.get("date", "")))
    if MIN_SCORE > 0:
        results = [r for r in results if r["score"] >= MIN_SCORE]
    results.sort(key=lambda r: r["rank"], reverse=True)

    # Diversità: al massimo MAX_PER_DOMAIN pagine per dominio.
    diverse, per_domain = [], {}
    for r in results:
        host = urlparse(r["url"]).netloc.lower()
        if per_domain.get(host, 0) >= MAX_PER_DOMAIN:
            continue
        per_domain[host] = per_domain.get(host, 0) + 1
        diverse.append(r)
    return diverse[:NUM_RESULTS]


def gather_sources(core_question: str, exclude=(), academic: bool = False):
    """Costruisce le query iniziali (multi-query + bilingue + accademica), cerca e ordina."""
    if MULTI_QUERY:
        queries = search_queries(core_question, academic=academic)
    else:
        kw = extract_keywords(core_question) or core_question
        queries = [kw]
    print("   Ricerche generate:")
    for q in queries:
        print(f"     • {q}")
    return rank_results(search_many(queries, exclude), academic=academic)


# ---------------------------------------------------------------------------
# Dialogo con il modello (Ollama)
# ---------------------------------------------------------------------------
def chat(messages, stream=False, num_predict=None, spinner_text="", show=True):
    """Chiamata a Ollama /api/chat.
    - stream=False: ritorna il testo completo (ripulito dal <think>), senza stampare.
    - stream=True, show=True : stampa la risposta mentre arriva e la ritorna.
    - stream=True, show=False: consuma lo stream SENZA stampare (spinner attivo per
      tutta la durata) e ritorna il testo. Serve alle sintesi intermedie del loop:
      lo streaming evita il timeout di lettura su generazioni molto lunghe (>300s).
    """
    options = {"num_ctx": NUM_CTX, "temperature": TEMPERATURE}
    if num_predict:
        options["num_predict"] = num_predict
    payload = {"model": MODEL, "messages": messages, "stream": stream, "options": options}

    try:
        if not stream:
            with Spinner(spinner_text) if spinner_text else _nullctx():
                r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=600)
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
                    if show:
                        safe = clean_stream(acc)
                        if len(safe) > emitted:
                            if spinner:              # primo testo: spegni lo spinner
                                spinner.__exit__()
                                spinner = None
                            print(safe[emitted:], end="", flush=True)
                            emitted = len(safe)
                if obj.get("done"):
                    break
            if spinner:                              # show=False, o nessun testo utile
                spinner.__exit__()
            if show:
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
    "Sei un analista di ricerca rigoroso. Ti vengono dati UNA fonte e una domanda. "
    "Estrai SOLO ciò che è davvero presente nella fonte, dando PRIORITÀ AI DATI CONCRETI: "
    "numeri, percentuali, correlazioni (es. r=0.x), dimensioni dei campioni, nomi, date, "
    "studi citati. Niente frasi vaghe, niente invenzioni.\n"
    "Indica anche, in una riga finale 'AFFIDABILITÀ:', che tipo di fonte è "
    "(studio scientifico / ente o istituzione / articolo divulgativo / blog promozionale "
    "o classifica 'best/top' senza dati) e quanto ti fidi.\n"
    "Se ci sono CRITERI dell'utente, segnala se la fonte li rispetta o non lo dice. "
    "Se la fonte non è pertinente, scrivi soltanto: NON PERTINENTE."
)

REDUCE_SYSTEM = (
    "/no_think\n"
    "Sei un analista di ricerca CRITICO. Rispondi nella lingua della domanda usando SOLO "
    "i riassunti forniti, citando ogni affermazione con [n]. Applica pensiero critico:\n"
    "- confronta le fonti tra loro: evidenzia accordi e CONTRADDIZIONI;\n"
    "- distingui le affermazioni sostenute da DATI da quelle generiche o promozionali;\n"
    "- DIFFIDA esplicitamente delle classifiche 'migliori/best/top' prive di evidenza;\n"
    "- privilegia studi ed enti rispetto ai blog;\n"
    "- se ci sono CRITERI dell'utente, valuta OGNI elemento rispetto ad essi e scarta o "
    "segnala ciò che non li soddisfa o non fornisce i dati per verificarli.\n"
    "Non inventare: se i dati mancano, dillo chiaramente."
)

ANALYSIS_FORMAT = """Produci un'analisi critica in italiano con questo formato:

## Risposta sintetica
(2-3 frasi che rispondono direttamente, con citazioni [n])

## Evidenze e dati
(i fatti concreti trovati — numeri, correlazioni, campioni — ognuno con [n])

## Valutazione critica delle fonti
(quali fonti sono affidabili e quali no e perché; contraddizioni tra fonti;
classifiche promozionali da prendere con le pinze)
{constraints_section}
## Limiti / cosa manca
(dati assenti o da verificare altrove)

## Fonti
(elenco numerato: [n] Titolo — URL)"""

FOLLOWUP_SYSTEM = REDUCE_SYSTEM  # stesse regole critiche: cita le stesse fonti [n]

VERIFY_SYSTEM = (
    "/no_think\n"
    "Sei un fact-checker rigoroso. Ricevi un'ANALISI e i RIASSUNTI delle fonti numerate "
    "su cui si basa. Per ogni affermazione importante dell'analisi, controlla se è "
    "davvero sostenuta dalla fonte [n] che cita. NON usare conoscenza esterna: giudica "
    "solo in base ai riassunti forniti. Sii conciso e onesto."
)

VERIFY_FORMAT = """Verifica l'analisi rispetto alle fonti e produci in italiano:

## ✅ Verifica delle affermazioni
- ✓ confermate: affermazioni ben sostenute dalla fonte citata
- ⚠️ deboli: affermazioni vaghe, oltre i dati, o con citazione [n] non pertinente
- ✗ errate: affermazioni contraddette dalle fonti o non presenti in esse
(se è tutto corretto, dillo chiaramente; cita sempre i numeri [n])"""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def is_irrelevant(summary: str) -> bool:
    """True se l'analista ha giudicato la fonte non pertinente (da scartare)."""
    return summary.strip().upper().startswith("NON PERTINENTE")


def map_summaries(question: str, results, constraints: str = ""):
    """MAP: riassume ogni fonte separatamente. Ritorna una lista di stringhe."""
    summaries = []
    n = len(results)
    crit = f"\nCRITERI dell'utente: {constraints}" if constraints else ""
    for i, res in enumerate(results, 1):
        title = res["title"] or res["url"]
        body = (res["raw_content"] or res["content"])[:MAP_CONTENT_CHARS]
        if not body:
            print(f"   [{i}/{n}] (fonte vuota, saltata): {title[:55]}")
            summaries.append("NON PERTINENTE")
            continue
        msg = [
            {"role": "system", "content": MAP_SYSTEM},
            {"role": "user", "content":
                f"Domanda: {question}{crit}\n\nFonte:\n{title}\n{body}\n\nEstrazione critica:"},
        ]
        label = f"  [{i}/{n}] Riassumo: {title[:55]}"
        summary = chat(msg, stream=False, num_predict=600, spinner_text=label).strip()
        if not summary:
            # qwen3 ha esaurito il budget pensando: ripieghiamo sullo snippet grezzo,
            # così la fonte contribuisce comunque alla sintesi finale.
            summary = res["content"][:400] or "NON PERTINENTE"
        summaries.append(summary)
        print(f"   ✓ [{i}/{n}] {title[:55]}")
    return summaries


def build_numbered(results, summaries=None):
    """Costruisce il blocco numerato [n] Titolo (data) / testo / URL da dare al reduce."""
    lines = []
    for i, res in enumerate(results, 1):
        if summaries is not None:
            text = summaries[i - 1]
        elif FULL_CONTENT:
            text = (res["raw_content"] or res["content"])[:CONTENT_CHARS]
        else:
            text = res["content"][:SNIPPET_CHARS]
        y = source_year(res.get("date", ""))
        head = f"[{i}] {res['title']}" + (f" (data: {y})" if y else " (data: n.d.)")
        lines.append(f"{head}\n{text}\nURL: {res['url']}")
    return "\n\n".join(lines)


def synthesize(question: str, context: str, constraints: str = "", stream: bool = True):
    """REDUCE: dai riassunti/fonti produce l'analisi critica. In streaming per la
    sintesi finale, silenziosa (con spinner) per i round intermedi del loop."""
    if constraints:
        section = ("\n## Conformità ai tuoi criteri\n"
                   f"(valuta gli elementi rispetto a: «{constraints}» — di' quali li "
                   "rispettano, quali no e quali non forniscono i dati per saperlo)\n")
        crit = f"\n\nCRITERI dell'utente (vincoli da rispettare): {constraints}"
    else:
        section, crit = "", ""
    fmt = ANALYSIS_FORMAT.format(constraints_section=section)
    msg = [
        {"role": "system", "content": REDUCE_SYSTEM},
        {"role": "user", "content":
            f"Domanda: {question}{crit}\n\nRiassunti delle fonti:\n{context}\n\n{fmt}"},
    ]
    if stream:
        print()
        return chat(msg, stream=True, show=True, spinner_text="Elaboro la sintesi critica...")
    # round intermedio: streaming silenzioso (evita il timeout su generazioni lunghe)
    return chat(msg, stream=True, show=False,
                spinner_text="Valuto i risultati e cerco le lacune...")


def verify(analysis: str, context: str):
    """AUTO-VERIFICA: il modello rilegge la propria analisi confrontandola con le fonti
    già raccolte (NON riscarica nulla) e segnala le affermazioni non supportate."""
    msg = [
        {"role": "system", "content": VERIFY_SYSTEM},
        {"role": "user", "content":
            f"ANALISI da verificare:\n{analysis}\n\nRIASSUNTI delle fonti:\n{context}\n\n{VERIFY_FORMAT}"},
    ]
    print()
    return chat(msg, stream=True, show=True, spinner_text="Verifico le affermazioni sulle fonti...")


def extract_gaps(analysis: str):
    """Estrae le lacune dalla sezione 'Limiti / cosa manca' della sintesi.
    Sfrutta il fatto che il modello le elenca in modo affidabile (è ancorato),
    evitando di chiedergli di GENERARLE da zero (cosa su cui qwen3:4b va in loop)."""
    gaps, collecting = [], False
    for line in analysis.splitlines():
        s = line.strip()
        if s.startswith("#"):
            low = s.lower()
            collecting = ("limiti" in low or "manca" in low or "verificare" in low)
            continue
        if collecting and s:
            t = s.lstrip("-*•0123456789. ").strip()
            # scarta righe vuote o che dicono "nessuna lacuna"
            if len(t) > 8 and "nessun" not in t.lower():
                gaps.append(t)
    return gaps


def gap_queries(core_question: str, gaps, academic: bool = False):
    """Trasforma le lacune in query di ricerca mirate (parole chiave della lacuna +
    del tema), aggiungendo la variante inglese. Deterministico: niente generazione."""
    core_kw = extract_keywords(core_question)
    out, seen = [], set()
    for g in gaps[:GAP_QUERIES]:
        gk = extract_keywords(g)
        if not gk:
            continue
        q = f"{core_kw} {gk}".strip()
        variants = [q]
        if BILINGUAL or academic:
            variants.append(translate_en(q))
        if academic:
            variants.append(f"{translate_en(q) or q} study")
        for variant in variants:
            v = variant.strip()
            if v and v.lower() not in seen:
                seen.add(v.lower())
                out.append(v)
    return out


def slugify(text: str, maxlen: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:maxlen].strip("-") or "ricerca"


def save_markdown(question: str, analysis: str, results) -> str:
    """Salva la ricerca in un file Markdown e ritorna il percorso."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    ts = datetime.now()
    path = os.path.join(SAVE_DIR, f"{ts:%Y-%m-%d_%H%M%S}_{slugify(question)}.md")
    sources = "\n".join(
        f"{i}. [{r['title']}]({r['url']})" + (f" — {y}" if (y := source_year(r.get('date', ''))) else "")
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


def read_sources(question: str, results, constraints: str):
    """MAP + gating: legge le fonti, scarta le non pertinenti. Ritorna coppie
    (fonte, riassunto) da accumulare nella base di conoscenza."""
    summaries = map_summaries(question, results, constraints)
    pairs = list(zip(results, summaries))
    if DROP_IRRELEVANT:
        kept = [(r, s) for r, s in pairs if not is_irrelevant(s)]
        dropped = len(pairs) - len(kept)
        if dropped:
            print(f"   🗑️  Scartate {dropped} fonti non pertinenti.")
        return kept
    return pairs


def research(question: str, session: dict):
    """Ricerca iterativa: cerca → legge → sintetizza → individua le LACUNE → cerca in
    modo mirato per colmarle → rifonde. Ripete fino a MAX_ROUNDS o finché restano lacune.
    Chiude con un'auto-verifica delle affermazioni sulle fonti."""
    question, academic = detect_academic(question)
    core, constraints = split_constraints(question)
    print(f"\n🔎 Cerco: {core}")
    if academic:
        print("   🎓 Modalità accademica: priorità a paper, studi e fonti dense.")
    if constraints:
        print(f"   ⚖️  Criterio di valutazione (non cercato): {constraints}")

    knowledge = {}   # url -> (fonte, riassunto): la base di conoscenza che cresce
    next_queries = []
    analysis = ""

    def final_synthesis(kept):
        print(f"\n🧩 Sintesi critica finale con {MODEL} ({len(kept)} fonti):")
        return synthesize(question, build_numbered([r for r, _ in kept],
                                                   [s for _, s in kept]), constraints, stream=True)

    for rnd in range(1, MAX_ROUNDS + 1):
        final = rnd == MAX_ROUNDS
        if rnd == 1:
            results = gather_sources(core, academic=academic)
        else:
            print(f"\n🔁 Round {rnd}: cerco per colmare le lacune...")
            results = rank_results(search_many(next_queries, exclude=set(knowledge)),
                                   academic=academic)

        if results:
            print(f"   {len(results)} nuove fonti. Leggo e riassumo una alla volta...")
            for r, s in read_sources(question, results, constraints):
                knowledge[r["url"]] = (r, s)
        elif rnd > 1:
            print("   (nessuna fonte nuova per le lacune)")

        if not knowledge:
            print("⚠️  Nessuna fonte pertinente. Riformula la domanda.")
            return

        kept = list(knowledge.values())
        context = build_numbered([r for r, _ in kept], [s for _, s in kept])

        if final:
            analysis = final_synthesis(kept)
            break

        # Round intermedio: sintesi silenziosa solo per individuare le lacune
        analysis = synthesize(question, context, constraints, stream=False)
        gaps = extract_gaps(analysis)
        if not gaps:
            print("   ✅ Nessuna lacuna rilevante: chiudo con la sintesi finale.")
            analysis = final_synthesis(kept)
            break

        print("   🔍 Lacune individuate:")
        for g in gaps[:GAP_QUERIES]:
            print(f"     – {g[:90]}")
        next_queries = gap_queries(core, gaps, academic=academic)
        if not next_queries:
            analysis = final_synthesis(kept)
            break

    results_final = [r for r, _ in knowledge.values()]
    context_final = build_numbered(results_final, [s for _, s in knowledge.values()])

    # Auto-verifica: rilegge l'analisi e segnala ciò che le fonti non sostengono
    if VERIFY and analysis.strip():
        verification = verify(analysis, context_final)
        if verification.strip():
            analysis += "\n\n" + verification

    session["question"] = question
    session["context"] = context_final
    session["analysis"] = analysis
    session["results"] = results_final

    if SAVE_ENABLED and analysis.strip():
        try:
            path = save_markdown(question, analysis, results_final)
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
