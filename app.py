import os, json, re, math, subprocess, threading, time
from flask import Flask, request, jsonify, render_template_string
import urllib.request
import urllib.parse

app = Flask(__name__)

# ── Dataset cache ──────────────────────────────────────────────────────────────
DATASET_CACHE = []
CACHE_FILE = "dataset_cache.json"

def fetch_dataset():
    """Download dataset from HuggingFace on first run, then cache locally."""
    global DATASET_CACHE

    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            DATASET_CACHE = json.load(f)
        print(f"[dataset] Loaded {len(DATASET_CACHE)} rows from cache.")
        return

    print("[dataset] Downloading from HuggingFace...")
    all_rows = []
    offset = 0
    limit = 100

    while True:
        url = (
            f"https://datasets-server.huggingface.co/rows"
            f"?dataset=talktoyeet/cunnyScrape01"
            f"&config=default&split=train"
            f"&offset={offset}&limit={limit}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "python/3"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
                rows = data.get("rows", [])
                if not rows:
                    break
                for item in rows:
                    row = item["row"]
                    all_rows.append({
                        "submission_title": row.get("submission_title", ""),
                        "comment_author":   row.get("comment_author", ""),
                        "comment_body":     row.get("comment_body", ""),
                    })
                offset += limit
                if offset >= data.get("num_rows_total", 9999):
                    break
        except Exception as e:
            print(f"[dataset] Error at offset {offset}: {e}")
            break

    DATASET_CACHE = all_rows
    with open(CACHE_FILE, "w") as f:
        json.dump(all_rows, f)
    print(f"[dataset] Saved {len(all_rows)} rows to cache.")


# ── Retrieval ──────────────────────────────────────────────────────────────────
def tokenize(text):
    return set(re.findall(r"\w+", text.lower()))

def bm25_score(query_tokens, doc_text, k1=1.5, b=0.75):
    """Lightweight BM25 scoring (no external libs)."""
    doc_tokens = re.findall(r"\w+", doc_text.lower())
    doc_len = len(doc_tokens)
    avg_len = 300  # rough average for this dataset
    freq = {}
    for t in doc_tokens:
        freq[t] = freq.get(t, 0) + 1
    score = 0.0
    for qt in query_tokens:
        if qt in freq:
            tf = freq[qt]
            idf = math.log(1 + (320 - 1 + 0.5) / (1 + 1))  # simplified
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_len))
    return score

def retrieve_examples(name: str, description: str, top_k: int = 5):
    """Find the most relevant comments from the dataset."""
    query = f"{name} {description}".strip()
    q_tokens = tokenize(query)

    scored = []
    for row in DATASET_CACHE:
        doc = f"{row['submission_title']} {row['comment_body']}"
        score = bm25_score(q_tokens, doc)
        scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scored[:top_k]]


# ── Ollama call ────────────────────────────────────────────────────────────────
def call_ollama(prompt: str, model: str = "qwen2.5:1.5b") -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.9, "top_p": 0.95, "num_predict": 300}
    }).encode()

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        result = json.loads(r.read())
        return result.get("response", "").strip()


def build_prompt(name: str, description: str, examples: list) -> str:
    ex_block = ""
    for i, ex in enumerate(examples, 1):
        body = ex["comment_body"][:400].replace("\n", " ")
        ex_block += f"\nExample {i} (posted on: \"{ex['submission_title'][:80]}\"):\n{body}\n"

    desc_line = f"Description: {description}" if description.strip() else ""

    return f"""You are mimicking the posting style of an anime fan community (Blue Archive subreddit).
The community writes highly expressive, chaotic, emotional comments — mixing crying emojis 😭, 💢, rage caps, affectionate absurdity, and over-the-top reactions to anime characters.

Study these real examples from the community:
{ex_block}

Now write ONE comment in the EXACT same unhinged, expressive style about:
Name: {name}
{desc_line}

Rules:
- Use ALL CAPS randomly for emotional peaks
- Include 😭😭😭, 💢💢💢, 👺, 😋 emojis liberally
- Be affectionate, dramatic, and absurd
- Reference "Sensei" naturally if it fits
- Do NOT explain yourself, just write the comment
- 2–5 sentences max

Comment:"""


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    name = (data.get("name") or "").strip()
    description = (data.get("description") or "").strip()

    if not name:
        return jsonify({"error": "Name is required"}), 400
    if not DATASET_CACHE:
        return jsonify({"error": "Dataset not loaded yet, please wait."}), 503

    try:
        examples = retrieve_examples(name, description, top_k=5)
        prompt = build_prompt(name, description, examples)
        output = call_ollama(prompt)
        return jsonify({
            "output": output,
            "examples_used": [
                {"title": e["submission_title"][:60], "snippet": e["comment_body"][:120]}
                for e in examples
            ]
        })
    except ConnectionRefusedError:
        return jsonify({"error": "Ollama is not running. Start it with: ollama serve"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/status")
def status():
    return jsonify({
        "dataset_loaded": len(DATASET_CACHE),
        "ollama_url": "http://localhost:11434"
    })


# ── HTML UI ────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kivotos Comment Generator</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #13131a;
    --border: #2a2a3a;
    --accent: #ff6b6b;
    --accent2: #ffd93d;
    --accent3: #6bcb77;
    --text: #e8e8f0;
    --muted: #6b6b80;
    --glow: rgba(255,107,107,0.15);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Space Mono', monospace;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 40px 20px;
    background-image:
      radial-gradient(ellipse 60% 40% at 50% 0%, rgba(255,107,107,0.08) 0%, transparent 70%),
      repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(255,255,255,0.02) 40px),
      repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(255,255,255,0.02) 40px);
  }

  header {
    text-align: center;
    margin-bottom: 48px;
  }
  .logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(2rem, 5vw, 3.5rem);
    line-height: 1;
    letter-spacing: -2px;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .subtitle {
    color: var(--muted);
    font-size: 0.75rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 8px;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px;
    width: 100%;
    max-width: 640px;
    position: relative;
    overflow: hidden;
  }
  .card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent3));
  }

  label {
    display: block;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
  }
  input, textarea {
    width: 100%;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    color: var(--text);
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    padding: 14px 16px;
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
    resize: vertical;
  }
  input:focus, textarea:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--glow);
  }
  .field { margin-bottom: 24px; }
  textarea { min-height: 90px; }

  button {
    width: 100%;
    padding: 16px;
    background: linear-gradient(135deg, var(--accent) 0%, #ff4757 100%);
    border: none;
    border-radius: 10px;
    color: #fff;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 1px;
    cursor: pointer;
    transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s;
    position: relative;
    overflow: hidden;
  }
  button:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(255,107,107,0.35);
  }
  button:active:not(:disabled) { transform: translateY(0); }
  button:disabled { opacity: 0.5; cursor: not-allowed; }

  .spinner {
    display: none;
    width: 18px; height: 18px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    margin: 0 auto;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .output-card {
    margin-top: 28px;
    display: none;
  }
  .output-label {
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent2);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .output-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }
  .output-text {
    background: var(--bg);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 10px;
    padding: 20px;
    font-size: 0.95rem;
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .examples-section {
    margin-top: 20px;
  }
  .examples-label {
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent3);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .examples-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }
  .example-pill {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.75rem;
    color: var(--muted);
  }
  .example-pill strong { color: var(--text); display: block; margin-bottom: 3px; font-size: 0.7rem; }

  .error-msg {
    background: rgba(255,75,75,0.1);
    border: 1px solid rgba(255,75,75,0.3);
    border-radius: 10px;
    padding: 14px 18px;
    color: #ff7070;
    font-size: 0.85rem;
    margin-top: 16px;
    display: none;
  }

  .status-bar {
    font-size: 0.65rem;
    letter-spacing: 1px;
    color: var(--muted);
    text-align: center;
    margin-top: 24px;
  }
  .dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--muted);
    margin-right: 6px;
    vertical-align: middle;
  }
  .dot.green { background: var(--accent3); box-shadow: 0 0 8px var(--accent3); }
  .dot.red   { background: var(--accent);  box-shadow: 0 0 8px var(--accent); }

  .copy-btn {
    width: auto;
    padding: 8px 16px;
    font-size: 0.75rem;
    background: var(--border);
    margin-top: 10px;
    border-radius: 8px;
    letter-spacing: 0;
  }
  .copy-btn:hover:not(:disabled) {
    box-shadow: none;
    background: var(--muted);
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .animate-in { animation: fadeIn 0.4s ease forwards; }
</style>
</head>
<body>

<header>
  <div class="logo">DEGENGPT</div>
  <div class="subtitle">Blue Archive comment generator · local · offline</div>
</header>

<div class="card">
  <div class="field">
    <label for="name">Character / Subject Name</label>
    <input type="text" id="name" placeholder="e.g. Yuuka, Hina, Arona..." autocomplete="off">
  </div>
  <div class="field">
    <label for="desc">Description <span style="color:var(--muted);font-size:0.65rem;letter-spacing:0">(optional)</span></label>
    <textarea id="desc" placeholder="e.g. she's wearing a swimsuit and looking smug, very dangerous energy..."></textarea>
  </div>
  <button id="gen-btn" onclick="generate()">
    <span id="btn-text">GENERATE COMMENT 😭</span>
    <div class="spinner" id="spinner"></div>
  </button>

  <div class="error-msg" id="error-msg"></div>

  <div class="output-card" id="output-card">
    <div class="output-label">Generated Comment</div>
    <div class="output-text" id="output-text"></div>
    <button class="copy-btn" onclick="copyOutput()">Copy 📋</button>

    <div class="examples-section" id="examples-section">
      <div class="examples-label">Retrieved Examples Used</div>
      <div id="examples-list"></div>
    </div>
  </div>
</div>

<div class="status-bar">
  <span class="dot" id="status-dot"></span>
  <span id="status-text">checking status...</span>
</div>

<script>
async function checkStatus() {
  try {
    const r = await fetch('/status');
    const d = await r.json();
    const dot = document.getElementById('status-dot');
    const txt = document.getElementById('status-text');
    if (d.dataset_loaded > 0) {
      dot.className = 'dot green';
      txt.textContent = `dataset: ${d.dataset_loaded} rows loaded · ollama: ${d.ollama_url}`;
    } else {
      dot.className = 'dot red';
      txt.textContent = 'dataset loading...';
      setTimeout(checkStatus, 3000);
    }
  } catch(e) {
    document.getElementById('status-dot').className = 'dot red';
    document.getElementById('status-text').textContent = 'server error';
  }
}
checkStatus();

async function generate() {
  const name = document.getElementById('name').value.trim();
  const desc = document.getElementById('desc').value.trim();
  const btn  = document.getElementById('gen-btn');
  const spinner = document.getElementById('spinner');
  const btnText = document.getElementById('btn-text');
  const errorMsg = document.getElementById('error-msg');
  const outputCard = document.getElementById('output-card');

  if (!name) { flash('Name is required 😭'); return; }

  btn.disabled = true;
  btnText.style.display = 'none';
  spinner.style.display = 'block';
  errorMsg.style.display = 'none';
  outputCard.style.display = 'none';

  try {
    const r = await fetch('/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ name, description: desc })
    });
    const d = await r.json();

    if (d.error) {
      flash(d.error);
    } else {
      document.getElementById('output-text').textContent = d.output;
      const list = document.getElementById('examples-list');
      list.innerHTML = '';
      (d.examples_used || []).forEach(ex => {
        const div = document.createElement('div');
        div.className = 'example-pill';
        div.innerHTML = `<strong>${ex.title}</strong>${ex.snippet}...`;
        list.appendChild(div);
      });
      outputCard.style.display = 'block';
      outputCard.classList.add('animate-in');
    }
  } catch(e) {
    flash('Network error — is the server running?');
  } finally {
    btn.disabled = false;
    btnText.style.display = 'block';
    spinner.style.display = 'none';
  }
}

function flash(msg) {
  const el = document.getElementById('error-msg');
  el.textContent = msg;
  el.style.display = 'block';
}

function copyOutput() {
  const text = document.getElementById('output-text').textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = event.target;
    btn.textContent = 'Copied! ✅';
    setTimeout(() => btn.textContent = 'Copy 📋', 1500);
  });
}

document.getElementById('name').addEventListener('keydown', e => {
  if (e.key === 'Enter') generate();
});
</script>
</body>
</html>"""


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load dataset in background so Flask starts instantly
    t = threading.Thread(target=fetch_dataset, daemon=True)
    t.start()
    print("\n🚀 Starting Kivotos Comment Generator")
    print("   Open: http://localhost:5000")
    print("   Make sure Ollama is running: ollama serve")
    print("   Model needed: ollama pull qwen2.5:1.5b\n")
    app.run(debug=False, port=5000)
