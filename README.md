# Formulator Agent

**Formulator Agent** is an AI-powered system designed to generate structured, production-ready form schemas and surveys from natural language prompts for [Formulator](https://github.com/naouuud/formulator).

If a user's prompt is ambiguous or lacks necessary detail, the agent intelligently pauses form generation and initiates an interactive clarification loop before producing the final form.

Built with Python 3.12+, **FastAPI**, **Pydantic v2**, and the **OpenAI Python SDK**, it supports Google Gemini (via OpenAI-compatible endpoints) and OpenAI models with built-in schema validation, automated self-repair, and token cost tracking.

---

## Key Features

- **Schema-Driven Form Generation**: Generates multi-page or single-page form structures containing typed questions (`text`, `radio`, `checkbox`, `select`), custom validation rules (`required`), and explanatory notes.
- **Clarification Loop**: If a query is too generic, ambiguous, or underspecified, the agent outputs targeted clarification questions with selectable options (and free-form support) rather than guessing.
- **Automated Self-Repair**: Enforces strict JSON Schema compliance via Pydantic. If an LLM returns invalid JSON or schema violations, the agent triggers an automated feedback loop sending the exact validation errors back to the model for correction (up to a configurable attempt limit).
- **Multi-Provider LLM Support**: Designed around OpenAI-compatible API interfaces, natively supporting **Google Gemini** (`gemini-3.5-flash-lite`) and **OpenAI** (`gpt-4o-mini`, `gpt-4o`).
- **Observability & Cost Tracking**: Thread-safe structured JSONL logging records attempt counts, repair attempts, token usage, and cost estimates for every request.
- **Dual Interface**: Exposes both an asynchronous **FastAPI REST API** and a **CLI interface**.

---

## Architecture & Project Structure

```
formulator-agent/
├── pyproject.toml              # Project dependencies and build configuration (uv)
├── README.md                   # Project documentation
├── src/
│   └── formulator_agent/
│       ├── __init__.py         # Package entrypoint
│       ├── agent/
│       │   ├── errors.py       # Custom agent domain exceptions
│       │   ├── fixtures/       # Example JSON schemas and payloads
│       │   ├── llm.py          # OpenAI SDK client factory & configuration
│       │   ├── logger.py       # Thread-safe JSONL observability logger
│       │   ├── run_agent.py    # Agent orchestration, prompts, and CLI runner
│       │   ├── run_logs/       # Observability logs (logs.jsonl)
│       │   ├── structured_completion.py  # Structured completion with self-repair
│       │   └── models/
│       │       └── output/     # Pydantic schema definitions
│       │           ├── form_response.py    # Form, Page, Question, Option models
│       │           ├── query_response.py   # Clarification Query models
│       │           └── output_container.py # Root discriminated container
│       ├── api/
│       │   ├── app.py          # FastAPI application & endpoints
│       │   └── errors.py       # API error models and exception handlers
│       ├── mcp_server/         # Placeholder for Model Context Protocol integration
│       ├── queue/              # Placeholder for async queue processing
│       └── storage/            # Placeholder for persistent form storage
```

---

## Data Schema & Output Types

Responses produced by the agent conform to `OutputContainer`, which contains step-by-step reasoning and a discriminated output payload:

### 1. `FormResponse` (`output_type: "form"`)

Returned when the agent has sufficient information to construct the form:

- **`draft_title`**: Internal working title for the form.
- **`pages`**: List of form pages. Each page includes:
  - `id`: Auto-generated UUID string.
  - `title`: Page header/title.
  - `elements`: List of either:
    - **`QuestionElement` (`type: "question"`)**:
      - `htmlType`: Input type (`text`, `radio`, `checkbox`, or `select`).
      - `label`: User-facing question text.
      - `validators`: Validation settings (e.g. `{"required": true}`).
      - `options`: Available choices for choice-based questions (`radio`, `checkbox`, `select`), each with `label`, `value`, and `id`.
    - **`NoteElement` (`type: "note"`)**: Explanatory text or instructions.

### 2. `QueryResponse` (`output_type: "query"`)

Returned when the user prompt requires clarification before a form can be generated:

- **`queries`**: 1 to 3 clarification questions, each with a `question` string and recommended multiple-choice `options` (including `"Other"` for open-ended replies).

---

## Getting Started

### Prerequisites

- **Python 3.12+**
- **[uv](https://github.com/astral-sh/uv)** package manager

### Installation

Clone the repository and install dependencies using `uv`:

```bash
git clone <repo-url>
cd formulator-agent

# Install dependencies into a virtual environment
uv sync
```

### Environment Configuration

Create a `.env` file in the project root with your credentials:

```bash
# Provider selection (gemini or openai)
GEMINI_API_KEY="your-gemini-api-key"
GEMINI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL="gemini-3.5-flash-lite"

# OpenAI configuration (optional / alternative)
OPENAI_API_KEY="your-openai-api-key"
OPENAI_MODEL="gpt-4o-mini"
```

---

## Usage

### Running the FastAPI Server

Start the API service in development mode:

```bash
uv run fastapi dev src/formulator_agent/api/app.py
```

Or using `uvicorn`:

```bash
uv run uvicorn formulator_agent.api.app:app --host 0.0.0.0 --port 8000 --reload
```

The interactive API documentation will be available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Running via CLI

You can execute the agent directly from the command line:

```bash
# Run with default query and Gemini provider
uv run python -m formulator_agent.agent.run_agent

# Run with a custom query using OpenAI
uv run python -m formulator_agent.agent.run_agent \
  --query "Create a customer feedback survey for a coffee shop" \
  --provider openai
```

---

## API Reference

### `GET /`

Health check endpoint.

**Response:**

```json
{
  "message": "Formulator Agent API is running"
}
```

---

### `POST /generate`

Generates a form schema or clarification questions based on the provided query and optional conversation history.

#### Request Body

| Field      | Type           | Required | Description                                                                          |
| ---------- | -------------- | -------- | ------------------------------------------------------------------------------------ |
| `query`    | `string`       | **Yes**  | The user prompt describing the desired form or answering previous queries.           |
| `qa_pairs` | `list[object]` | No       | Clarification history to refine generation. Each object has `question` and `answer`. |

#### Example 1: Ambiguous Request (Triggers Clarification)

**Request:**

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I want a registration form"
  }'
```

**Response (`QueryResponse`):**

```json
{
  "output_type": "query",
  "queries": [
    {
      "question": "What kind of event or service is this registration form for?",
      "options": [
        "Conference / Workshop",
        "Sports Event / Tournament",
        "School / Course Enrollment",
        "Membership / Club",
        "Other"
      ]
    }
  ]
}
```

#### Example 2: Follow-up Request with Clarification Answers

**Request:**

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Create the form for a local tournament",
    "qa_pairs": [
      {
        "question": "What kind of tournament is this registration form for?",
        "answer": "Chess tournament"
      }
    ]
  }'
```

**Response (`FormResponse`):**

```json
{
  "output_type": "form",
  "draft_title": "Chess Tournament Registration",
  "pages": [
    {
      "id": "e305e94b-1fd0-4bce-939e-26f53ca1b58a",
      "title": "Participant Details",
      "elements": [
        {
          "type": "question",
          "id": "18fbaeb2-98aa-4e94-8255-a0ca067e2a44",
          "el": {
            "label": "Full Name",
            "htmlType": "text",
            "validators": {
              "required": true
            }
          }
        },
        {
          "type": "question",
          "id": "557a552b-7bc0-4e5a-93f4-34fd772183aa",
          "el": {
            "label": "Skill Category",
            "htmlType": "radio",
            "validators": {
              "required": true
            },
            "options": [
              {
                "label": "Beginner (< 1200 ELO)",
                "value": "Beginner (< 1200 ELO)",
                "id": "1d8eb2a1-e4ea-4c48-8df0-f203875bfdca"
              },
              {
                "label": "Intermediate (1200 - 1800 ELO)",
                "value": "Intermediate (1200 - 1800 ELO)",
                "id": "27fa55f2-9599-4d6b-9d41-eeef975b7b99"
              },
              {
                "label": "Open / Advanced (> 1800 ELO)",
                "value": "Open / Advanced (> 1800 ELO)",
                "id": "3be523d2-3cf7-4fbb-913a-446736ddc0b1"
              }
            ]
          }
        }
      ]
    }
  ]
}
```

---

## Observability & Logging

Structured completion activity is appended to `src/formulator_agent/agent/run_logs/logs.jsonl`. Each record captures:

- `request_id`: Unique UUID for tracing across multiple completion/repair attempts.
- `event`: Lifecycle events (`structured_completion started`, `calling model`, `structured output invalid`, `structured_completion succeeded`, `structured_completion failed`).
- `attempt_number` & `repairs_used`: Number of correction attempts needed.
- `total_tokens_in` / `total_tokens_out`: Token consumption.
- `total_cost`: Estimated execution cost calculated against the model price table.

---

## License

[Apache License 2.0](https://github.com/naouuud/formulator-agent/blob/main/LICENSE)
