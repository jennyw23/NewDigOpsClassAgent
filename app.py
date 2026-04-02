#!/usr/bin/env python3
"""
MBA Digital Operations Course Assistant — Web Application
Run:  ANTHROPIC_API_KEY=sk-... python app.py
"""

import json
import os
from datetime import date

import anthropic
import yaml
from flask import Flask, Response, render_template, request, stream_with_context

COURSE_INFO_FILE = os.path.join(os.path.dirname(__file__), "course_info.yaml")
MODEL = "claude-opus-4-6"

app = Flask(__name__)

# ── Load course data once at startup ─────────────────────────────────────────

with open(COURSE_INFO_FILE) as _f:
    _course_data = yaml.safe_load(_f)


def _build_course_text(data: dict) -> str:
    """Format course YAML into plain text for the system prompt."""
    lines = []

    course = data["course"]
    lines += [
        f"# {course['name']} ({course['code']}) — {course['semester']}",
        f"Professor: {course['professor']}",
        f"Email: {course['email']}",
        f"Class location & time: {course['location']}, {course['schedule']}",
        f"Office hours: {course['office_hours']}",
        "",
        "## Course Description",
        data["description"].strip(),
        "",
        "## Grading",
    ]
    for item in data["grading"]:
        lines.append(f"- {item['component']}: {item['weight']}")
    lines.append("")

    lines.append("## All Deadlines (chronological)")
    for d in data["deadlines"]:
        lines.append(f"- {d['name']}: {d['due']}")
    lines.append("")

    lines.append("## Course Modules")
    for mod in data["modules"]:
        lines.append(f"### Module {mod['module']}: {mod['title']} ({mod['weeks']})")
        lines.append("Topics:")
        for t in mod["topics"]:
            lines.append(f"  - {t}")
        lines.append("Readings:")
        for r in mod["readings"]:
            lines.append(f"  - {r}")
        if mod.get("cases"):
            lines.append("Cases:")
            for c in mod["cases"]:
                lines.append(f"  - {c}")
        lines.append("")

    policies = data["policies"]
    lines += [
        "## Policies",
        f"- Late work: {policies['late_work']}",
        f"- Attendance: {policies['attendance']}",
        f"- AI tools: {policies['ai_tools']}",
        f"- Group project size: {policies['group_project_size']}",
        f"- Collaboration: {policies['collaboration']}",
        "",
    ]

    resources = data["resources"]
    lines += [
        "## Resources",
        f"- LMS: {resources['lms']}",
        f"- Cases: {resources['cases']}",
        f"- Office hours sign-up: {resources['office_hours_signup']}",
        f"- TA: {resources['ta']}",
    ]

    return "\n".join(lines)


_course_text = _build_course_text(_course_data)

# System prompt is cached across requests — only built once per server start.
# Using prompt caching (cache_control) so the large course block is cached on
# Anthropic's side too, reducing cost when many students use the assistant.
_system_prompt = [
    {
        "type": "text",
        "text": (
            "You are a helpful course assistant for an MBA Digital Operations class. "
            "Your role is to answer students' questions about course deadlines, materials, "
            "assignments, policies, and logistics. "
            "Be friendly, concise, and precise. If a student asks something not covered in "
            "the course information below, say so and suggest they email the professor or "
            "check Canvas. Never invent deadlines or policies.\n\n"
            f"Today's date is {date.today().isoformat()}.\n\n"
            "Here is the complete course information:\n\n"
        )
        + _course_text,
        "cache_control": {"type": "ephemeral"},
    }
]

_client = anthropic.Anthropic()


# ── Routes ────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html", course=_course_data)


@app.route("/chat", methods=["POST"])
def chat():
    """
    Accepts JSON: {message: str, history: [{role, content}, ...]}
    Streams back Server-Sent Events: data: {"text": "..."}\n\n
    Ends with:                        data: [DONE]\n\n
    """
    body = request.get_json(silent=True) or {}
    user_message = (body.get("message") or "").strip()
    history = body.get("history") or []

    if not user_message:
        return {"error": "Empty message"}, 400

    messages = list(history) + [{"role": "user", "content": user_message}]

    def generate():
        try:
            with _client.messages.stream(
                model=MODEL,
                max_tokens=1024,
                system=_system_prompt,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'text': text})}\n\n"
            yield "data: [DONE]\n\n"
        except anthropic.APIError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        raise SystemExit(1)

    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on http://localhost:{port}")
    app.run(debug=True, port=port)
