#!/usr/bin/env python3
"""Minimal offline reproduction of the GEM data synthesis flow.

This is not the paper's full teacher-model pipeline. It mirrors the stages and
data contracts so the process can be tested before replacing heuristics with LLM
calls.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from executable_environment import build_environment, replay_row


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Tool:
    name: str
    description: str
    properties: dict[str, dict[str, str]]
    required: list[str]

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": self.required,
                },
            },
        }


PHOTO_TOOLS = [
    Tool(
        "copy_image",
        "Create a safe working copy of an image before editing.",
        {"image_path": {"type": "string", "description": "Path of the source image."}},
        ["image_path"],
    ),
    Tool(
        "open_editor",
        "Open an image copy in the editor.",
        {"copy_path": {"type": "string", "description": "Path of the copied image."}},
        ["copy_path"],
    ),
    Tool(
        "add_text",
        "Add text to an image with position, font size, and color.",
        {
            "editor_id": {"type": "string", "description": "Active editor session id."},
            "text": {"type": "string", "description": "Text to add."},
            "position": {"type": "string", "description": "Placement on the image."},
            "font_size": {"type": "integer", "description": "Font size between 8 and 96."},
            "color": {"type": "string", "description": "Text color."},
        },
        ["editor_id", "text", "position", "font_size", "color"],
    ),
    Tool(
        "print_image",
        "Send an edited image to a printer.",
        {
            "image_path": {"type": "string", "description": "Path of the edited image."},
            "printer_id": {"type": "string", "description": "Printer identifier."},
        },
        ["image_path", "printer_id"],
    ),
    Tool(
        "list_available_printers",
        "List currently available printers for fallback printing.",
        {},
        [],
    ),
]

RETURN_TOOLS = [
    Tool(
        "sign_in",
        "Authenticate a store customer.",
        {"email": {"type": "string", "description": "Customer email."}},
        ["email"],
    ),
    Tool(
        "get_order",
        "Retrieve order status and item details.",
        {"order_id": {"type": "string", "description": "Store order id."}},
        ["order_id"],
    ),
    Tool(
        "check_return_eligibility",
        "Check return window and item policy constraints.",
        {
            "order_id": {"type": "string", "description": "Store order id."},
            "item_id": {"type": "string", "description": "Item to return."},
            "reason": {"type": "string", "description": "Return reason."},
        },
        ["order_id", "item_id", "reason"],
    ),
    Tool(
        "create_return_label",
        "Create a shipping label for an eligible return.",
        {"order_id": {"type": "string", "description": "Store order id."}, "item_id": {"type": "string", "description": "Item to return."}},
        ["order_id", "item_id"],
    ),
    Tool(
        "schedule_pickup",
        "Schedule carrier pickup for a return shipment.",
        {"label_id": {"type": "string", "description": "Return label id."}, "pickup_date": {"type": "string", "description": "Requested pickup date."}},
        ["label_id", "pickup_date"],
    ),
]

COURSE_TOOLS = [
    Tool(
        "portal_login",
        "Authenticate a student in the learning portal.",
        {"student_id": {"type": "string", "description": "Student account id."}},
        ["student_id"],
    ),
    Tool(
        "search_course",
        "Search for a course by keyword.",
        {"query": {"type": "string", "description": "Course search text."}},
        ["query"],
    ),
    Tool(
        "check_prerequisites",
        "Check whether a student meets course prerequisites.",
        {"student_id": {"type": "string", "description": "Student id."}, "course_id": {"type": "string", "description": "Course id."}},
        ["student_id", "course_id"],
    ),
    Tool(
        "request_instructor_approval",
        "Request instructor approval when prerequisites are missing.",
        {"student_id": {"type": "string", "description": "Student id."}, "course_id": {"type": "string", "description": "Course id."}},
        ["student_id", "course_id"],
    ),
    Tool(
        "confirm_enrollment",
        "Finalize enrollment after checks and payment.",
        {"student_id": {"type": "string", "description": "Student id."}, "course_id": {"type": "string", "description": "Course id."}},
        ["student_id", "course_id"],
    ),
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stage1_filter(row: dict[str, Any]) -> dict[str, Any] | None:
    text = row["text"].lower()
    procedural_markers = ["first", "then", "after", "before", "if ", "checking", "choose", "schedule"]
    if sum(marker in text for marker in procedural_markers) < 2:
        return None
    if "photo" in text or "printer" in text:
        task = "multimedia_processing"
        domain = "computers_and_electronics"
        summary = "Edit photo text and print it while respecting formatting limits."
    elif "return" in text or "order" in text:
        task = "ecommerce_and_retail"
        domain = "shopping"
        summary = "Process an online order return under store policy constraints."
    elif "course" in text or "student" in text:
        task = "education_elearning"
        domain = "jobs_and_education"
        summary = "Enroll a student in an online course with prerequisite checks."
    else:
        task = "customer_support"
        domain = "business_and_industrial"
        summary = "Complete a multi-step support workflow."
    return {**row, "multi_step": True, "summary": summary, "domain": domain, "platform": "computer", "task": task}


def stage2_extract(row: dict[str, Any]) -> dict[str, Any]:
    text = row["text"].lower()
    if "photo" in text:
        tools = PHOTO_TOOLS
        steps = [
            "Ask for missing image path, text, placement, font size, and color.",
            "Copy the source image.",
            "Open the copied image in an editor.",
            "Add text while enforcing font size between 8 and 96.",
            "Print the edited image.",
            "If printing fails, list available printers and retry with one returned printer.",
        ]
        execution_graph = "(copy_image)->(open_editor)->(add_text)->(print_image)->(list_available_printers)->(print_image)"
    elif "return" in text:
        tools = RETURN_TOOLS
        steps = [
            "Sign in to the customer account.",
            "Retrieve the order.",
            "Check delivery status, 30 day window, and item-specific policy.",
            "Create a return label for eligible items.",
            "Schedule pickup for the return package.",
        ]
        execution_graph = "->".join(f"({tool.name})" for tool in tools)
    else:
        tools = COURSE_TOOLS
        steps = [
            "Log in to the learning portal.",
            "Search for the requested course.",
            "Check prerequisites and seat availability.",
            "Request instructor approval if prerequisites are missing.",
            "Confirm enrollment after approval.",
        ]
        execution_graph = "->".join(f"({tool.name})" for tool in tools)
    tool_schemas = [tool.to_openai_schema() for tool in tools]
    return {
        **row,
        "workflow": {
            "description": row["summary"],
            "steps": steps,
            "execution_graph": execution_graph,
        },
        "tools": tool_schemas,
        "environment": build_environment(row["task"], [tool.name for tool in tools]),
    }


def assistant_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "tool_call": {"name": name, "arguments": arguments}}


def tool_response(name: str, content: dict[str, Any]) -> dict[str, Any]:
    return {"role": "tool", "name": name, "content": content}


def stage3_generate(row: dict[str, Any]) -> dict[str, Any]:
    if row["task"] == "multimedia_processing":
        messages = [
            {"role": "system", "content": "You help edit and print photos. Font size must be between 8 and 96. Ask for missing required parameters. Retry printing by listing available printers after a printer failure."},
            {"role": "user", "content": "Please add 'Summer 2024' to my vacation photo and print it."},
            {"role": "assistant", "content": "I need the image path, text position, font size, color, and preferred printer."},
            {"role": "user", "content": "The image is /home/vacation.jpg. Put it bottom right, use font size 150 in blue, and print on printer_A."},
            {"role": "assistant", "content": "Font size 150 is outside the allowed 8 to 96 range. I can use 96 instead."},
            {"role": "user", "content": "Use 96."},
            assistant_tool_call("copy_image", {"image_path": "/home/vacation.jpg"}),
            tool_response("copy_image", {"copy_path": "/tmp/vacation_copy.jpg"}),
            assistant_tool_call("open_editor", {"copy_path": "/tmp/vacation_copy.jpg"}),
            tool_response("open_editor", {"editor_id": "editor_41"}),
            assistant_tool_call("add_text", {"editor_id": "editor_41", "text": "Summer 2024", "position": "bottom right", "font_size": 96, "color": "blue"}),
            tool_response("add_text", {"image_path": "/tmp/vacation_copy_edited.jpg"}),
            assistant_tool_call("print_image", {"image_path": "/tmp/vacation_copy_edited.jpg", "printer_id": "printer_A"}),
            tool_response("print_image", {"status": "failed", "error": "printer offline"}),
            {"role": "assistant", "content": "printer_A is offline. I will list available printers before retrying."},
            assistant_tool_call("list_available_printers", {}),
            tool_response("list_available_printers", {"available_printers": ["printer_B"]}),
            {"role": "assistant", "content": "printer_B is available, so I will retry printing there."},
            assistant_tool_call("print_image", {"image_path": "/tmp/vacation_copy_edited.jpg", "printer_id": "printer_B"}),
            tool_response("print_image", {"status": "queued", "job_id": "job_778"}),
            {"role": "assistant", "content": "The edited photo has been queued on printer B."},
        ]
    elif row["task"] == "ecommerce_and_retail":
        messages = [
            {"role": "system", "content": "You help process store returns. Returns must be delivered, within 30 days, and item policy must allow the return. Opened cosmetics are only returnable when damaged."},
            {"role": "user", "content": "I want to return a moisturizer from order R100 because I changed my mind."},
            {"role": "assistant", "content": "I can check that. What email is on the account?"},
            {"role": "user", "content": "alex@example.com"},
            assistant_tool_call("sign_in", {"email": "alex@example.com"}),
            tool_response("sign_in", {"customer_id": "cust_18"}),
            assistant_tool_call("get_order", {"order_id": "R100"}),
            tool_response("get_order", {"status": "delivered", "days_since_delivery": 12, "items": [{"item_id": "i9", "name": "moisturizer", "category": "cosmetics", "opened": True}]}),
            assistant_tool_call("check_return_eligibility", {"order_id": "R100", "item_id": "i9", "reason": "changed my mind"}),
            tool_response("check_return_eligibility", {"eligible": False, "reason": "opened cosmetics require damage reason"}),
            {"role": "assistant", "content": "This item is not eligible for a change-of-mind return because opened cosmetics can only be returned if damaged."},
        ]
    else:
        messages = [
            {"role": "system", "content": "You help students enroll in courses. Prerequisites and seat availability must be checked before enrollment. If prerequisites are missing, request instructor approval before confirming enrollment."},
            {"role": "user", "content": "Enroll me in Advanced Databases."},
            {"role": "assistant", "content": "I need your student id to continue."},
            {"role": "user", "content": "S123"},
            assistant_tool_call("portal_login", {"student_id": "S123"}),
            tool_response("portal_login", {"status": "ok"}),
            assistant_tool_call("search_course", {"query": "Advanced Databases"}),
            tool_response("search_course", {"course_id": "CS540", "seats_available": 3}),
            assistant_tool_call("check_prerequisites", {"student_id": "S123", "course_id": "CS540"}),
            tool_response("check_prerequisites", {"met": False, "missing": ["CS340"]}),
            {"role": "assistant", "content": "You are missing CS340, so I need to request instructor approval before enrollment."},
            assistant_tool_call("request_instructor_approval", {"student_id": "S123", "course_id": "CS540"}),
            tool_response("request_instructor_approval", {"approved": True, "approval_id": "ap_77"}),
            assistant_tool_call("confirm_enrollment", {"student_id": "S123", "course_id": "CS540"}),
            tool_response("confirm_enrollment", {"status": "enrolled"}),
            {"role": "assistant", "content": "You are enrolled in Advanced Databases."},
        ]
    return {**row, "messages": messages}


def stage4_refine(row: dict[str, Any]) -> dict[str, Any]:
    """Mark simple interaction patterns used by the generated trajectory."""
    patterns = []
    joined = "\n".join(str(message) for message in row["messages"]).lower()
    if "need" in joined and "missing" in joined or "what email" in joined:
        patterns.append("clarification")
    if "outside the allowed" in joined or "not eligible" in joined:
        patterns.append("rule_adherence")
    if "failed" in joined or "retry" in joined:
        patterns.append("error_recovery")
    if "approval" in joined:
        patterns.append("conditional_branch")
    return {**row, "refinement_patterns": sorted(set(patterns))}


def validate(row: dict[str, Any]) -> list[str]:
    errors = []
    tool_schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in row["tools"]}
    if not row["messages"] or row["messages"][0]["role"] != "system":
        errors.append("missing system message")
    for index, message in enumerate(row["messages"]):
        if message["role"] != "assistant" or "tool_call" not in message:
            continue
        call = message["tool_call"]
        name = call.get("name")
        args = call.get("arguments", {})
        if name not in tool_schemas:
            errors.append(f"message {index}: unknown tool {name}")
            continue
        schema = tool_schemas[name]
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in args:
                errors.append(f"message {index}: missing required arg {required} for {name}")
        for arg_name, value in args.items():
            if arg_name not in properties:
                errors.append(f"message {index}: unexpected arg {arg_name} for {name}")
                continue
            expected = properties[arg_name].get("type")
            if expected == "string" and not isinstance(value, str):
                errors.append(f"message {index}: arg {arg_name} should be string")
            if expected == "integer" and not isinstance(value, int):
                errors.append(f"message {index}: arg {arg_name} should be integer")
        if index + 1 >= len(row["messages"]) or row["messages"][index + 1]["role"] != "tool":
            errors.append(f"message {index}: tool call not followed by tool response")
    if row.get("environment"):
        execution = replay_row(row)
        errors.extend(f"execution: {error}" for error in execution["errors"])
    return errors


def summarize(rows: list[dict[str, Any]], validation: list[dict[str, Any]]) -> dict[str, Any]:
    valid_count = sum(1 for item in validation if item["valid"])
    return {
        "total_raw_texts": len(rows),
        "valid_trajectories": valid_count,
        "invalid_trajectories": len(validation) - valid_count,
        "avg_tools": round(sum(len(row["tools"]) for row in rows) / max(len(rows), 1), 2),
        "avg_messages": round(sum(len(row["messages"]) for row in rows) / max(len(rows), 1), 2),
        "avg_tool_calls": round(
            sum(sum(1 for message in row["messages"] if message.get("role") == "assistant" and "tool_call" in message) for row in rows)
            / max(len(rows), 1),
            2,
        ),
        "patterns": sorted({pattern for row in rows for pattern in row["refinement_patterns"]}),
        "execution_grounded": sum(1 for item in validation if item["valid"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/sample_texts.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/toy")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = load_jsonl(args.input)
    filtered = [row for row in (stage1_filter(row) for row in raw_rows) if row is not None]
    extracted = [stage2_extract(row) for row in filtered]
    generated = [stage4_refine(stage3_generate(row)) for row in extracted]
    validation = [{"id": row["id"], "valid": not (errors := validate(row)), "errors": errors} for row in generated]
    report = summarize(generated, validation)

    write_jsonl(args.output_dir / "filtered_texts.jsonl", filtered)
    write_jsonl(args.output_dir / "trajectories.jsonl", generated)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
