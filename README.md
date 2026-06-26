# 🧠 Agente di ricerca locale

Agente di ricerca da terminale che gira **completamente in locale e a costo zero**:
cerca sul web con **Tavily Search API** (piano gratuito) e analizza i risultati con
**Qwen3 4B** via **Ollama**, citando le fonti.

Pensato per **MacBook Air M1 / 8GB di RAM**: le fonti vengono lette ed elaborate
**una alla volta** (map-reduce), così si possono leggere articoli interi senza
saturare la memoria; modalità "thinking" di Qwen3 disattivata.

## Funzionalità

- **Loop iterativo (deep research)**: dopo la prima sintesi l'agente individua le
  **lacune** ("manca il dato sui campioni"), fa ricerche mirate per colmarle e rifonde.
  Ripete per più round → da semplice riassuntore a vero ricercatore.
- **Ricerca bilingue**: aggiunge automaticamente una query tradotta in inglese, dove
  vivono le fonti più ricche e autorevoli su tech/finanza/scienza (traduzione gratuita).
- **Modalità accademica**: con tag (`#paper`, `#studi`, `#scientifico`...) — o da sola
  quando la domanda contiene segnali accademici — punta su paper, studi e fonti dense
  (PubMed, arXiv, editori scientifici).
- **Auto-verifica**: passo finale in cui il modello ricontrolla la propria analisi
  contro le fonti raccolte e segnala le affermazioni non supportate (✓/⚠️/✗).
- **Consapevolezza temporale**: usa la data delle fonti, dà un bonus a quelle recenti.
- **Cache**: non ripaga ricerche Tavily identiche entro il TTL (risparmia quota e tempo).
- **Ranking per autorevolezza**: le fonti accademiche/ufficiali (`.edu`, `.gov`, enti,
  editori scientifici) ricevono un bonus e vengono lette per prime.
- **Map-reduce**: ogni fonte viene riassunta singolarmente (letta per intero), poi i
  riassunti vengono fusi in un'analisi citata. Permette di leggere articoli interi e
  più fonti restando entro gli 8GB.
- **Query a parole chiave**: la domanda in linguaggio naturale viene ridotta alle
  parole chiave essenziali; stopword ed esche da classifica (*migliori/best/top/guida*)
  vengono rimosse per non attirare listicle SEO. I vincoli ("escludi ...") sono separati
  e usati come criterio di valutazione, non per la ricerca.
- **Multi-query**: più ricerche con angolazioni diverse, unite e deduplicate. Migliora
  la copertura e supera il tetto di 20 risultati di una singola ricerca.
- **Sintesi critica**: confronta le fonti, segnala contraddizioni, distingue i dati
  dalle affermazioni promozionali, diffida delle classifiche senza evidenza e valuta
  esplicitamente ogni elemento rispetto ai tuoi criteri di esclusione.
- **Fonti di qualità**: ricerca `advanced`, ordinamento per rilevanza, esclusione dei
  forum/social, diversità di dominio e scarto delle fonti giudicate non pertinenti.
- **Salvataggio Markdown**: ogni ricerca finisce in `./ricerche/` per costruire il tuo
  archivio di approfondimenti.
- **Domande di follow-up**: con `/f <domanda>` approfondisci sulle stesse fonti, senza
  consumare una nuova ricerca Tavily.

---

## Perché Qwen3 **4B** e non 8B?

Su 8GB di RAM unificata un modello 8B (Q4 ≈ 5GB di pesi) lascia troppo poco margine:
sistema, browser e Ollama riempiono la RAM e il Mac va in **swap su SSD**, con
inferenza lentissima. `qwen3:4b` (Q4 ≈ 2.6GB) è lo sweet spot: veloce e con qualità
più che sufficiente per riassumere risultati web. Se servisse ancora più leggerezza:
`qwen2.5:3b`.

---

## Requisiti

- macOS con [Homebrew](https://brew.sh/)
- Python 3.9+
- Una chiave **Tavily API** gratuita → https://app.tavily.com
  (il piano *Free* dà 1.000 ricerche/mese, nessuna carta di credito)

---

## Setup

### 1. Ollama + modello
```bash
brew install ollama

# Avvia il server (ottimizzato per 8GB) e lascialo in un terminale dedicato:
OLLAMA_FLASH_ATTENTION="1" OLLAMA_KV_CACHE_TYPE="q8_0" ollama serve

# In un altro terminale, scarica il modello (una volta sola):
ollama pull qwen3:4b
```
In alternativa, per avviarlo come servizio in background: `brew services start ollama`.

### 2. Dipendenze Python
```bash
cd ~/research-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Chiave Tavily
```bash
cp .env.example .env
# apri .env e incolla la chiave dopo TAVILY_API_KEY=
```
Oppure, senza file `.env`:
```bash
export TAVILY_API_KEY='tvly-...'
```
> ⚠️ `cp .env.example .env` va fatto **una sola volta**: rieseguirlo cancella la chiave.

---

## Uso

**Modalità interattiva** (loop di domande):
```bash
python main.py
```

**Una domanda al volo:**
```bash
python main.py "andamento dei tassi BCE nel 2026"
python main.py "migliori rapporti per salita in gravel"
```

**Approfondire (follow-up)** nella modalità interattiva, dopo una ricerca:
```
/f e per il 2025 cosa cambia?
/f approfondisci il punto 2
```

L'output è strutturato in: *Risposta sintetica → Approfondimento → Limiti → Fonti*,
con citazioni `[n]` collegate all'elenco delle fonti, e viene salvato in `./ricerche/`.

---

## Configurazione (variabili d'ambiente opzionali)

| Variabile               | Default                  | Note                                   |
|-------------------------|--------------------------|----------------------------------------|
| `TAVILY_API_KEY`            | —                        | **obbligatoria**                          |
| `OLLAMA_MODEL`              | `qwen3:4b`               | es. `qwen2.5:3b` per più leggerezza       |
| `OLLAMA_URL`                | `http://localhost:11434` | endpoint Ollama                           |
| `OLLAMA_NUM_CTX`            | `8192`                   | finestra di contesto (RAM)                |
| `RESEARCH_NUM_RESULTS`      | `5`                      | numero di fonti web da passare            |
| `RESEARCH_MAP_REDUCE`       | `1`                      | `0` = analisi in un'unica richiesta       |
| `RESEARCH_MAP_CONTENT_CHARS`| `6000`                   | caratteri letti per fonte nel map         |
| `RESEARCH_SAVE`             | `1`                      | salva le ricerche in Markdown             |
| `RESEARCH_SAVE_DIR`         | `./ricerche`             | cartella di salvataggio                    |
| `RESEARCH_MULTI_QUERY`      | `1`                      | genera più ricerche e le unisce           |
| `RESEARCH_NUM_QUERIES`      | `3`                      | quante ricerche generare                  |
| `RESEARCH_QUERY_MODIFIERS`  | `<anno>,guida,confronto` | angolazioni aggiunte alla domanda         |
| `RESEARCH_SEARCH_DEPTH`     | `advanced`               | `basic` per spendere 1 credito invece di 2|
| `RESEARCH_MIN_SCORE`        | `0`                      | scarta fonti sotto questa rilevanza (0-1) |
| `RESEARCH_EXCLUDE_DOMAINS`  | reddit, quora, social…   | domini esclusi (vuoto = nessuno)          |

---

## Risoluzione problemi

- **`Ollama non è in esecuzione`** → avvia `ollama serve` in un altro terminale.
- **`modello non installato`** → `ollama pull qwen3:4b`.
- **`401` su Tavily** → chiave errata o non attivata. Rigenera dal dashboard.
- **`429` su Tavily** → esaurite le 1.000 ricerche mensili gratuite.
- **Mac lento / ventola** → riduci `OLLAMA_NUM_CTX` a `2048` o passa a `qwen2.5:3b`.
