# RAG Fundamentals: Kenyan data-protection and cybercrime law

A from-scratch Retrieval-Augmented Generation (RAG) pipeline over three Kenyan legal texts:

- Computer Misuse and Cybercrimes Act (Cap. 79C, 2018)
- Data Protection Act (Cap. 411C, 2019)
- Data Protection (General) Regulations, 2021

> Status: scaffold only. The full README (run, test, configuration, results) is added in a later commit.

## Quick setup (Windows PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env      # then edit .env and add your GEMINI_API_KEY
```

Get a free key at <https://aistudio.google.com/apikey>.

## Check that your key works

```powershell
python -c "from google import genai; from src import config as c; cl = genai.Client(api_key=c.get_gemini_api_key()); print(cl.models.generate_content(model=c.GEMINI_MODEL, contents='Reply with exactly: key works').text)"
```

Expected output: `key works`. The client is kept in a variable on purpose: if it is garbage-collected
mid-call the SDK raises `Cannot send a request, as the client has been closed`.
Brand-new free-tier models are sometimes overloaded (`503 high demand`); retry in a minute, or
see `GEMINI_FALLBACK_MODELS` in `src/config.py`.
