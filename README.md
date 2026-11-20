# PS69 - Customer Feedback Classifier

**Problem statement:** Feedback forms pile up unread. This tool classifies customer feedback into categories (delivery, quality, price, etc.) with a priority level, so businesses can act on urgent issues first.

## Features
- Upload a feedback CSV (or use the bundled sample)
- LLM (Llama 3 via Groq API, or Gemini) returns category, priority, sentiment and a short reason per row
- Rows are sent in batches to reduce API calls; invalid JSON is retried and outputs are validated
- Filter the table by category and priority, view charts, download results as CSV

## How to run
```
pip install -r requirements.txt
cp .env.example .env      # then put your free Groq API key in .env
streamlit run app.py
```
Get a free key at https://console.groq.com/keys (Gemini keys also work; the provider is auto-detected)

## Files
- `app.py` - Streamlit UI
- `classifier.py` - prompt, batching, JSON parsing, validation
- `sample_feedback.csv` - 20 test feedback rows
- `test_classifier.py` - offline tests using a stub LLM

## Screenshots
Add 1-2 screenshots here (e.g. `screenshots/result.png`).
