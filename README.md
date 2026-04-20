<div align="center">

# 🧾 Invoice Processing Environment

### 🟢 Active &nbsp;|&nbsp; 🐳 Docker &nbsp;|&nbsp; 🐍 Python 3.8+ &nbsp;|&nbsp; 📄 MIT License

**An OpenEnv-compatible AI agent environment for intelligent invoice processing and accounts payable validation.**

</div>

---

## 📖 Overview

This environment simulates a real-world **accounts payable workflow** where an AI agent must intelligently process invoices. The agent works through a series of structured tasks — from extracting fields to making final approve/reject decisions.

**The agent can:**
- 📄 Extract structured fields from raw invoice text
- 🔍 Match invoices against purchase orders
- ⚠️ Detect discrepancies (price mismatches, duplicates, tax errors)
- ✅ Make approve/reject decisions based on analysis

---

## 🎯 Action Space

| Action | Fields | Description |
|--------|--------|-------------|
| `extract` | `field_name`, `field_value` | Extract a specific field from the invoice |
| `match_po` | — | Match invoice total against the Purchase Order |
| `flag` | `field_name` | Flag an issue (`price_mismatch`, `duplicate_invoice`, `tax_mismatch`) |
| `match_duplicate` | — | Check if invoice was previously processed |
| `approve` | — | Approve the invoice |
| `reject` | — | Reject the invoice |

---

## 👁️ Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `invoice_text` | `string` | Raw invoice text |
| `po_data` | `object` | Purchase order reference data |
| `extracted_fields` | `object` | Fields extracted so far |
| `flags` | `array` | Issues flagged so far |
| `current_step` | `integer` | Current step number |
| `message` | `string` | Feedback from last action |

---

## 📋 Tasks

| Task | Difficulty | Description |
|------|------------|-------------|
| `easy` | 🟢 Easy | Extract fields from a clean invoice, match PO, approve |
| `medium` | 🟡 Medium | Detect a price mismatch between invoice and PO |
| `hard` | 🔴 Hard | Detect duplicate invoice + tax miscalculation |

---

## 🏆 Reward Function

| Event | Reward |
|-------|--------|
| ✅ Correct field extraction | `+0.07` per field |
| ❌ Wrong field value | `-0.02` |
| 🚩 Correct flag raised | `+0.12` |
| 🚫 Wrong flag | `-0.05` |
| 🔗 Successful PO match | `+0.10` |
| 🎯 Final grader score | up to `1.0` |

---

## 📊 Baseline Scores

| Task | Score |
|------|-------|
| 🟢 Easy | ~0.85 |
| 🟡 Medium | ~0.70 |
| 🔴 Hard | ~0.55 |

---

##  Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### 3. Run Inference
```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

python inference.py
```

---

## 🐳 Docker

```bash
# Build the image
docker build -t invoice-env .

# Run the container
docker run -p 7860:7860 invoice-env
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset?task_name=easy` | Start a new episode |
| `POST` | `/step?task_name=easy` | Take an action |
| `GET` | `/state?task_name=easy` | Get current state |
| `GET` | `/tasks` | List all available tasks |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Interactive API documentation |

---

<div align="center">
< Made with ❤️ >
</div>