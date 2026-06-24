# 🧠 Agente di ricerca locale

Agente di ricerca da terminale che gira **completamente in locale e a costo zero**:
cerca sul web con **Brave Search API** (piano gratuito) e analizza i risultati con
**Qwen3 4B** via **Ollama**, citando le fonti.

Pensato per **MacBook Air M1 / 8GB di RAM**: contesto breve, finestra ridotta,
modalità "thinking" disattivata per non saturare la memoria.

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

### 3. Chiave Brave
```bash
cp .env.example .env
# apri .env e incolla la chiave dopo BRAVE_API_KEY=
```
Oppure, senza file `.env`:
```bash
export BRAVE_API_KEY='la-tua-chiave'
```

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

L'output è strutturato in: *Risposta sintetica → Approfondimento → Limiti → Fonti*,
con citazioni `[n]` collegate all'elenco delle fonti.

---

## Configurazione (variabili d'ambiente opzionali)

| Variabile               | Default                  | Note                                   |
|-------------------------|--------------------------|----------------------------------------|
| `TAVILY_API_KEY`        | —                        | **obbligatoria**                       |
| `OLLAMA_MODEL`          | `qwen3:4b`               | es. `qwen2.5:3b` per più leggerezza    |
| `OLLAMA_URL`            | `http://localhost:11434` | endpoint Ollama                        |
| `OLLAMA_NUM_CTX`        | `4096`                   | finestra di contesto (RAM)             |
| `RESEARCH_NUM_RESULTS`  | `5`                      | numero di fonti web da passare         |

---

## Risoluzione problemi

- **`Ollama non è in esecuzione`** → avvia `ollama serve` in un altro terminale.
- **`modello non installato`** → `ollama pull qwen3:4b`.
- **`401` su Tavily** → chiave errata o non attivata. Rigenera dal dashboard.
- **`429` su Tavily** → esaurite le 1.000 ricerche mensili gratuite.
- **Mac lento / ventola** → riduci `OLLAMA_NUM_CTX` a `2048` o passa a `qwen2.5:3b`.
