"""NetFree support for the music agent.

NetFree is a filtered Israeli internet provider. It inspects traffic and can:
- block a request to the Claude API (HTTP 418, "badwords") when the text sent
  to Claude contains words on its list - usually a song or video title that
  came back from YouTube or Last.fm;
- block individual YouTube videos that are not approved on the user's
  filtering profile (yt-dlp then fails to load the video).

This module does NOT try to bypass the filter. It adapts the agent to it:
it keeps unapproved videos out of playlists and downloads, keeps titles with
offensive words out of what is sent to Claude, and explains clearly to the
user and to the agent when something failed because of NetFree.

Mode is set by NETFREE in .env: "on", "off" or "auto" (default).
In auto mode it is turned on at startup if the connection is filtered by
NetFree, or later as soon as any NetFree block is seen.
"""

import json
import os
import re
import socket
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECK_CACHE_FILE = PROJECT_DIR / "netfree_cache.json"

# How long a video check result is trusted. Blocked videos are rechecked
# sooner, because NetFree may approve them later.
BLOCKED_TTL = 24 * 3600
OK_TTL = 30 * 24 * 3600

_state = {"enabled": False, "reason": "off"}
_cache_lock = threading.Lock()


# ================== Mode ==================

def _connection_is_netfree() -> bool:
    """NetFree inspects HTTPS by re-signing certificates with its own CA,
    so the certificate we receive for a known site is issued by NetFree."""
    try:
        from cryptography import x509

        pem = ssl.get_server_certificate(("api.anthropic.com", 443), timeout=5)
        cert = x509.load_pem_x509_certificate(pem.encode())
        return "netfree" in cert.issuer.rfc4514_string().lower()
    except Exception:
        return False


def init() -> str:
    """Decide the mode at startup. Returns a short Hebrew status line for the user."""
    setting = os.environ.get("NETFREE", "auto").strip().lower()
    if setting in ("on", "1", "true", "yes"):
        _state.update(enabled=True, reason="הוגדר ב-.env")
    elif setting in ("off", "0", "false", "no"):
        _state.update(enabled=False, reason="כבוי ב-.env")
    elif _connection_is_netfree():
        _state.update(enabled=True, reason="זוהה אוטומטית")
    else:
        _state.update(enabled=False, reason="לא זוהה נטפרי")
    return status_line()


def status_line() -> str:
    state = "פעיל" if _state["enabled"] else "כבוי"
    return f"מצב נטפרי: {state} ({_state['reason']})"


def is_enabled() -> bool:
    return _state["enabled"]


def enable(reason: str) -> None:
    """Turn NetFree mode on after a block was seen (unless the user turned it off)."""
    if os.environ.get("NETFREE", "auto").strip().lower() in ("off", "0", "false", "no"):
        return
    if not _state["enabled"]:
        _state.update(enabled=True, reason=reason)
        print(f"  [netfree] {status_line()}")


# ================== Claude API blocks ==================

def is_api_block(exc: Exception) -> bool:
    """True if an Anthropic API error is a NetFree block page (HTTP 418)."""
    status = getattr(exc, "status_code", None)
    if status != 418:
        return False
    text = f"{getattr(exc, 'body', '')} {exc}".lower()
    return "netfree" in text


# ================== Offensive words filter ==================
# A small, conservative list. NetFree's real list is unknown, so this only
# lowers the chance of a block. Items containing these words are dropped from
# YouTube/Last.fm results before they reach Claude - for NetFree users these
# songs are unwanted anyway.

_BAD_WORDS = [
    "fuck", "fucking", "fucked", "motherfucker", "shit", "bitch", "bitches",
    "sex", "sexy", "porn", "porno", "nude", "nudes", "naked", "pussy", "dick",
    "cock", "cum", "slut", "whore", "hoe", "hoes", "nigga", "niggas", "nigger",
    "boobs", "tits", "orgasm", "erotic", "hentai", "xxx", "stripper", "strip club",
    "סקס", "סקסי", "פורנו", "עירום", "עירומה", "זונה", "זונות",
]
_BAD_RE = re.compile(r"(?<!\w)(" + "|".join(re.escape(w) for w in _BAD_WORDS) + r")(?!\w)",
                     re.IGNORECASE)


def _has_bad_word(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_BAD_RE.search(value))
    if isinstance(value, dict):
        return any(_has_bad_word(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_bad_word(v) for v in value)
    return False


def filter_result(result: Any) -> Any:
    """Drop list items (songs, videos, tags) whose text contains offensive words."""
    if isinstance(result, list):
        return [filter_result(item) for item in result if not _has_bad_word(item)]
    if isinstance(result, dict):
        return {k: filter_result(v) if isinstance(v, list) else v for k, v in result.items()}
    return result


# ================== YouTube video checks ==================

class _SilentLogger:
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


def classify_ytdlp_error(message: str) -> str:
    """Return "netfree", "maybe_netfree", "unavailable" or "other" for a yt-dlp error."""
    text = message.lower()
    if "netfree" in text or "http error 418" in text:
        return "netfree"
    if "failed to extract any player response" in text:
        # This is how a NetFree-blocked video looks to yt-dlp.
        return "netfree" if is_enabled() else "maybe_netfree"
    if any(s in text for s in ("video unavailable", "private video", "has been removed",
                               "not available in your country", "sign in to confirm your age")):
        return "unavailable"
    return "other"


def _load_cache() -> dict:
    try:
        return json.loads(CHECK_CACHE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    CHECK_CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _check_one(video_id: str) -> str:
    """Try to load the video like a download would, without downloading it."""
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        return "unknown"
    opts = {"quiet": True, "no_warnings": True, "logger": _SilentLogger(),
            "skip_download": True, "noplaylist": True, "socket_timeout": 15}
    try:
        with YoutubeDL(opts) as ydl:
            ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
        return "ok"
    except Exception as e:
        kind = classify_ytdlp_error(str(e))
        if kind == "netfree":
            return "blocked"
        if kind == "unavailable":
            return "unavailable"
        return "unknown"


def check_videos(video_ids: list[str]) -> dict[str, str]:
    """Check which videos can be watched on this connection.

    Returns {video_id: "ok" | "blocked" | "unavailable" | "unknown"}.
    Results are cached so each video is checked rarely.
    """
    now = time.time()
    with _cache_lock:
        cache = _load_cache()
    results, to_check = {}, []
    for vid in video_ids:
        entry = cache.get(vid)
        ttl = BLOCKED_TTL if entry and entry["status"] == "blocked" else OK_TTL
        if entry and now - entry["checked"] < ttl and entry["status"] != "unknown":
            results[vid] = entry["status"]
        else:
            to_check.append(vid)

    if to_check:
        with ThreadPoolExecutor(max_workers=5) as pool:
            for vid, status in zip(to_check, pool.map(_check_one, to_check)):
                results[vid] = status
        with _cache_lock:
            cache = _load_cache()
            for vid in to_check:
                cache[vid] = {"status": results[vid], "checked": now}
            _save_cache(cache)
    return results


def mark_blocked(video_id: str) -> None:
    """Remember that a video turned out to be blocked (e.g. a download failed)."""
    with _cache_lock:
        cache = _load_cache()
        cache[video_id] = {"status": "blocked", "checked": time.time()}
        _save_cache(cache)
