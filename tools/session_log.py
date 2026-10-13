"""Session log for the music agent.

Records everything that happens in a session - each request, every tool call
with its arguments and result, errors, NetFree blocks - in three ways:

1. A compact "app log" of each request, which agent.py attaches to the
   conversation history. The agent sees it in later requests, so it knows
   which tools ran, what they returned and which errors came up.
2. get_session_log - a tool the agent can call to read the full details of
   any earlier request in this session (exact tool results).
3. agent_log.jsonl - a file on disk with everything, for debugging.
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_DIR / "agent_log.jsonl"
MAX_LOG_BYTES = 5 * 1024 * 1024  # rotate the file at 5 MB

SESSION_ID = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]

# requests[n-1] = {"number", "user_text", "calls": [...], "reply", "status", "events": [...]}
_requests: list[dict] = []

# Results of these tools contain no song/video titles, so they are safe to keep
# even when NetFree blocked the run (e.g. the ID of a playlist that was created).
_SAFE_AFTER_BLOCK = {"create_playlist", "add_to_playlist", "record_playlist", "remember_songs"}


# ================== Recording ==================

def _write(entry: dict) -> None:
    entry = {"time": datetime.now().isoformat(timespec="seconds"), "session": SESSION_ID, **entry}
    try:
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > MAX_LOG_BYTES:
            LOG_FILE.replace(LOG_FILE.with_suffix(".old.jsonl"))
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass  # logging must never break the agent


def start_request(user_text: str) -> int:
    number = len(_requests) + 1
    _requests.append({"number": number, "user_text": user_text, "calls": [],
                      "events": [], "reply": None, "status": "running"})
    _write({"type": "request", "request": number, "user_text": user_text})
    return number


def _current() -> dict | None:
    return _requests[-1] if _requests else None


def record_call(tool: str, args: dict, result: Any, seconds: float, note: str = "") -> None:
    """Record one tool call. `result` is exactly what the agent received."""
    errors = errors_in(result)
    call = {"tool": tool, "args": args, "result": result, "errors": errors,
            "seconds": round(seconds, 2), "note": note}
    req = _current()
    if req is not None:
        call["blocked_run"] = False
        req["calls"].append(call)
    _write({"type": "tool", "request": req["number"] if req else None, **call})


def record_event(text: str) -> None:
    """Record something that happened outside a tool (e.g. a NetFree block)."""
    req = _current()
    if req is not None:
        req["events"].append(text)
    _write({"type": "event", "request": req["number"] if req else None, "text": text})


def mark_calls_blocked() -> None:
    """The calls so far were part of a run whose results NetFree blocked."""
    req = _current()
    if req:
        for call in req["calls"]:
            call["blocked_run"] = True


def end_request(reply: str | None, status: str) -> None:
    req = _current()
    if req is None:
        return
    req["reply"], req["status"] = reply, status
    _write({"type": "reply", "request": req["number"], "status": status, "reply": reply})


# ================== Summaries ==================

def errors_in(result: Any) -> list[str]:
    """Find error messages in a tool result."""
    errors = []
    if isinstance(result, dict):
        if result.get("error"):
            errors.append(str(result["error"]))
        for item in result.get("failed", []) or []:
            if isinstance(item, dict) and item.get("error"):
                errors.append(f"{item.get('video_id', '')}: {item['error']}")
    elif isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and (item.get("error") or item.get("status") not in (None, "success")):
                if item.get("error"):
                    name = " - ".join(x for x in (item.get("artist"), item.get("track")) if x)
                    errors.append(f"{name}: {item['error']}" if name else str(item["error"]))
    return errors


def _short(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _describe_item(item: Any) -> str:
    if not isinstance(item, dict):
        return _short(item, 60)
    if "artist" in item and "track" in item:
        return f"{item['artist']} - {item['track']}"
    if "title" in item:
        return _short(item["title"], 60)
    if "tag" in item:
        return str(item["tag"])
    if "artist" in item:
        return str(item["artist"])
    return _short(json.dumps(item, ensure_ascii=False), 60)


def summarize_result(result: Any, show: int = 5) -> str:
    """A one-line summary of a successful result."""
    if isinstance(result, list):
        items = [r for r in result if not (isinstance(r, dict) and ("error" in r or "note" in r))]
        notes = [r["note"] for r in result if isinstance(r, dict) and "note" in r]
        if items and isinstance(items[0], dict) and "status" in items[0]:  # download results
            ok = sum(1 for r in items if r.get("status") == "success")
            text = f"{ok}/{len(items)} downloaded"
            if items[0].get("output_dir"):
                text += f" to {items[0]['output_dir']}"
        else:
            shown = "; ".join(_describe_item(r) for r in items[:show])
            more = f" (+{len(items) - show} more)" if len(items) > show else ""
            text = f"{len(items)} results: {shown}{more}" if items else "0 results"
        return text + (f" [{' '.join(notes)}]" if notes else "")
    if isinstance(result, dict):
        if "candidates" in result:
            shown = "; ".join(_describe_item(c) for c in result["candidates"][:3])
            return f"clear_match={result.get('clear_match')}; top: {shown}"
        keys = ("playlist_id", "url", "added", "saved", "recorded", "song_count", "name",
                "songs_missing_video", "opinion")
        parts = [f"{k}={result[k]}" for k in keys if k in result]
        if "songs" in result and isinstance(result["songs"], list):
            parts.append(f"{len(result['songs'])} songs")
        if "liked" in result:
            parts.append(f"{len(result['liked'])} liked, {len(result.get('disliked', []))} disliked, "
                         f"{len(result.get('playlists', []))} playlists")
        return ", ".join(parts) or _short(json.dumps(result, ensure_ascii=False), 150)
    return _short(result, 150)


def _format_args(args: dict) -> str:
    parts = []
    for key, value in args.items():
        if isinstance(value, list):
            parts.append(f"{key}=[{len(value)} items]")
        else:
            parts.append(f"{key}={_short(json.dumps(value, ensure_ascii=False), 50)}")
    return ", ".join(parts)


def request_notes(number: int | None = None) -> str:
    """Compact app log of one request, to attach to the conversation history."""
    req = _requests[number - 1] if number else _current()
    if req is None:
        return ""
    lines = [f"[App log of request #{req['number']} - written automatically by the app, "
             "not by the user or by you. Full details: get_session_log("
             f"request_number={req['number']}).]"]
    for call in req["calls"]:
        head = f"- {call['tool']}({_format_args(call['args'])})"
        if call["errors"]:
            body = "ERROR: " + " | ".join(_short(e, 220) for e in call["errors"][:3])
        elif call.get("blocked_run") and call["tool"] not in _SAFE_AFTER_BLOCK:
            body = "ok (result not kept: NetFree blocked this run)"
        else:
            body = summarize_result(call["result"])
        note = f" [{call['note']}]" if call.get("note") else ""
        lines.append(f"{head} -> {_short(body, 300)}{note}")
    for event in req["events"]:
        lines.append(f"- EVENT: {event}")
    if not req["calls"] and not req["events"]:
        lines.append("- (no tools were used)")
    return "\n".join(lines)


# ================== Tool ==================

def get_session_log(request_number: int = 0, tool_name: str = "") -> dict:
    """Read the full details of what happened in an earlier request of this session.

    Each earlier request in the conversation ends with a short "[App log of
    request #N]" that lists the tools that ran and their errors. Use this tool
    when you need more than that short log - for example when the user asks
    why something failed or what exactly a tool returned earlier. Don't use it
    for the current request: you already see those results.

    Args:
        request_number: The request number from the app log (e.g. 3).
            0 (default) means the previous request.
        tool_name: Optional. Only return calls of this tool
            (e.g. "get_tracks_by_tag").

    Returns:
        {"request_number", "user_text", "status", "events",
         "calls": [{"tool", "args", "result", "errors", "seconds"}]}.
        Long results are shortened. On failure, {"error"}.
    """
    finished = [r for r in _requests if r["status"] != "running"]
    if not finished:
        return {"error": "No earlier requests in this session yet."}
    if request_number == 0:
        req = finished[-1]
    elif 1 <= request_number <= len(_requests):
        req = _requests[request_number - 1]
    else:
        return {"error": f"No request #{request_number}. This session has requests "
                         f"1-{len(_requests)}."}

    calls = []
    for call in req["calls"]:
        if tool_name and call["tool"] != tool_name:
            continue
        if call.get("blocked_run") and call["tool"] not in _SAFE_AFTER_BLOCK:
            result = "(not kept: NetFree blocked this run, re-sending it could be blocked again)"
        else:
            text = json.dumps(call["result"], ensure_ascii=False)
            result = call["result"] if len(text) <= 3000 else text[:3000] + " …(shortened)"
        calls.append({"tool": call["tool"], "args": call["args"], "result": result,
                      "errors": call["errors"], "seconds": call["seconds"]})
    if tool_name and not calls:
        return {"error": f"Request #{req['number']} has no calls to {tool_name}."}
    return {"request_number": req["number"], "user_text": req["user_text"],
            "status": req["status"], "events": req["events"], "calls": calls}
