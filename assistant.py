#!/usr/bin/env python3
"""
MBA Digital Operations Course Assistant
Helps students with questions about course deadlines, materials, and policies.

Usage:
    python assistant.py
    python assistant.py --course path/to/course_info.yaml
"""

import argparse
import os
import sys
import yaml
import anthropic


COURSE_INFO_FILE = os.path.join(os.path.dirname(__file__), "course_info.yaml")
MODEL = "claude-opus-4-6"


def load_course_info(path: str) -> str:
    """Load course info from YAML and format it for the system prompt."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)

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
    ]

    lines += ["## Grading"]
    for item in data["grading"]:
        lines.append(f"- {item['component']}: {item['weight']}")
    lines.append("")

    lines += ["## All Deadlines (chronological)"]
    for d in data["deadlines"]:
        lines.append(f"- {d['name']}: {d['due']}")
    lines.append("")

    lines += ["## Course Modules"]
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

    lines += ["## Policies"]
    policies = data["policies"]
    lines += [
        f"- Late work: {policies['late_work']}",
        f"- Attendance: {policies['attendance']}",
        f"- AI tools: {policies['ai_tools']}",
        f"- Group project size: {policies['group_project_size']}",
        f"- Collaboration: {policies['collaboration']}",
        "",
    ]

    lines += ["## Resources"]
    resources = data["resources"]
    lines += [
        f"- LMS: {resources['lms']}",
        f"- Cases: {resources['cases']}",
        f"- Office hours sign-up: {resources['office_hours_signup']}",
        f"- TA: {resources['ta']}",
    ]

    return "\n".join(lines)


def build_system_prompt(course_text: str) -> list[dict]:
    """
    Build the system prompt as a list of blocks.
    The course info block has cache_control so it is cached across student sessions,
    reducing cost when many students use the assistant simultaneously.
    """
    intro = (
        "You are a helpful course assistant for an MBA Digital Operations class. "
        "Your role is to answer students' questions about course deadlines, materials, "
        "assignments, policies, and logistics. "
        "Be friendly, concise, and precise. If a student asks something not covered in "
        "the course information below, say so and suggest they email the professor or "
        "check Canvas. Never invent deadlines or policies."
        "\n\n"
        "Today's date is 2026-04-01.\n\n"
        "Here is the complete course information:\n\n"
    )
    return [
        {
            "type": "text",
            "text": intro + course_text,
            # Cache the course info — it never changes between student sessions,
            # so subsequent requests pay ~0.1x the input token cost for this block.
            "cache_control": {"type": "ephemeral"},
        }
    ]


def stream_response(client: anthropic.Anthropic, messages: list, system: list) -> str:
    """Stream Claude's response to stdout and return the full text."""
    full_text = ""
    with client.messages.stream(
        model=MODEL,
        max_tokens=1024,
        system=system,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_text += text
    print()  # newline after response
    return full_text


def run_chat(course_path: str) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        print("Set it with:  export ANTHROPIC_API_KEY=your-key-here")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print(f"Loading course info from {course_path} ...")
    try:
        course_text = load_course_info(course_path)
    except FileNotFoundError:
        print(f"Error: Course info file not found: {course_path}")
        sys.exit(1)

    system = build_system_prompt(course_text)

    print("\n" + "=" * 60)
    print("  MBA Digital Operations Course Assistant")
    print("  Type 'quit' or press Ctrl+C to exit.")
    print("=" * 60 + "\n")

    messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        print("\nAssistant: ", end="", flush=True)
        response_text = stream_response(client, messages, system)
        print()

        messages.append({"role": "assistant", "content": response_text})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MBA Digital Operations Course Assistant"
    )
    parser.add_argument(
        "--course",
        default=COURSE_INFO_FILE,
        help="Path to the course_info.yaml file (default: course_info.yaml)",
    )
    args = parser.parse_args()
    run_chat(args.course)


if __name__ == "__main__":
    main()
