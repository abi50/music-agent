import functools
import json
import time

from anthropic import Anthropic, beta_tool
from dotenv import load_dotenv

from tools import lastfm, memory, netfree, session_log, youtube, youtube_download

load_dotenv()

MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 4096
MAX_ITERATIONS = 60  # safety limit on tool-call rounds per request

SYSTEM_PROMPT = """You are a personal music assistant. You find songs the user will love \
and build YouTube playlists for them. You can also download songs when the user \
explicitly asks for it.

How to work:
1. At the start of every music request (recommendations, playlists, downloads, \
questions about their songs), call get_music_memory to see what the user likes, \
dislikes and has already received. Don't call it for small talk, questions about \
you, or settings - it only costs time and tokens there.
2. If the user mentions songs they like or dislike, save them with remember_songs. \
If the user tells you a personal setting - that they are (or are not) on NetFree, \
or how to address them (e.g. in feminine form) - save it with update_user_settings.
3. Songs without an artist: if the user names a song but not who sings it (for \
any purpose - examples, playlist, download), call find_artist_for_song first. \
Don't guess the artist yourself (only if Last.fm is unavailable: then identify it \
from your knowledge, say whom you chose, and ask the user if unsure). If it returns clear_match true, use the top \
candidate and mention the artist in your reply. If clear_match is false, ask the \
user which artist they meant (show the top options) and wait for the answer before \
doing anything else. find_artist_for_song only identifies songs; it is not for \
recommendations.
4. Choose songs:
   - "Songs like X": use get_similar_tracks, and get_track_tags + get_tracks_by_tag \
to understand and widen the style.
   - "Songs by artist X": use get_artist_top_tracks.
   - A free description of a style or mood: translate it into a few common English \
Last.fm tags and use get_tracks_by_tag. You may also use your own music knowledge to \
pick a few seed songs and expand from them with get_similar_tracks.
   - Artist and song names should be in English when calling Last.fm. For Israeli \
artists, try the English spelling first, then the Hebrew one.
5. Curate the list:
   - Default to 15 songs unless the user asks for a different number.
   - At most 2 songs per artist unless the user asked for a specific artist.
   - Never include disliked songs. Avoid songs from already_suggested unless asked.
   - Don't include the user's example songs themselves, only new ones.
6. For each chosen song, call find_on_youtube and pick the correct candidate as its \
description explains. Skip songs with no good match and pick a replacement instead.
7. Create the playlist with create_playlist (private unless the user asked otherwise), \
or use add_to_playlist if the user wants to extend an existing playlist. \
Then call record_playlist with the songs that were actually added, including the \
video_id you used for each song. If some songs failed or were not attempted (quota or \
network), still record the ones that were added, and tell the user which songs are missing.
8. Reply in the user's language (usually Hebrew). Give the playlist link and a short \
list of the songs, with a few words on why they fit. Keep it short.

Downloading songs:
- Only call download_songs_from_youtube when the user explicitly asks to download. \
Never download as part of a normal playlist request.
- Every song you pass must have the video_id from find_on_youtube or \
get_playlist_songs. Never guess video IDs.
- If the user asks to download songs from one of their playlists (including one you \
just built), get its playlist_id from get_music_memory and call get_playlist_songs. \
Use the saved video IDs as they are; only songs without one need find_on_youtube. \
If they ask to download without a playlist, find the songs with steps 3-6 and skip \
creating a playlist unless they also want one.
- Quality: use "mp3" unless the user asks for video ("480p", "720p" or "best").
- Folder: leave output_dir empty to save to the computer's Downloads folder. Only \
pass output_dir if the user names a folder: a full path as they wrote it, or a plain \
folder name, which is created inside Downloads.
- After downloading, report which songs succeeded, which failed and why, and the \
folder they were saved to.

Session log: each earlier request in this conversation ends with an "[App log of \
request #N]" block, written automatically by the app. It lists the tools that ran, \
short results and every error. Use it to know what actually happened earlier (for \
example why a tool failed), and call get_session_log for the full details when the \
short log is not enough. Never write such logs yourself.

If a tool returns an error, read it and adapt (try another spelling, tag or song) \
instead of giving up. If the YouTube quota is exhausted, tell the user and stop. \
If the download tool reports a missing dependency (yt-dlp or FFmpeg), tell the user \
what to install and stop."""

# Sent only in NetFree mode, so other users don't pay tokens for it.
NETFREE_PROMPT = """NetFree (filtered internet):
Some users browse through NetFree, an Israeli filtered internet provider. It blocks \
YouTube videos that are not approved, and can block messages containing offensive words.
- Any tool error or status that says "BLOCKED BY NETFREE" / "blocked_by_netfree" is \
caused by the user's NetFree filter, not by a bug, quota or outdated software. Never \
suggest updating yt-dlp or retrying the same video for it.
- When something is blocked, say so explicitly in Hebrew, e.g. "השיר X חסום בנטפרי", \
and suggest a replacement or that the user ask NetFree to approve it.
- Most mainstream Western (non-Jewish) music videos are NOT approved on NetFree. \
Unless the user asks for specific Western songs or artists, choose Israeli and Jewish \
music (e.g. Jewish/Hasidic, Israeli soft/acoustic, piyyutim, niggunim), which is far \
more likely to be approved. This saves many blocked searches.
- Prefer clean, family-friendly songs, and avoid songs whose titles contain offensive \
words: they will likely be blocked."""


ADDRESS_FORMS = {
    "female": "Address the user in feminine Hebrew forms (את, רוצה, תגידי...).",
    "male": "Address the user in masculine Hebrew forms (אתה, רוצה, תגיד...).",
    "neutral": "Address the user in gender-neutral Hebrew (plural or impersonal forms).",
}


def system_prompt() -> list[dict]:
    text = SYSTEM_PROMPT
    settings = memory.get_settings()
    if settings.get("address_user_as") in ADDRESS_FORMS:
        text += "\n\n" + ADDRESS_FORMS[settings["address_user_as"]]
    if netfree.is_enabled():
        text += ("\n\n" + NETFREE_PROMPT + "\n\nNetFree mode is ON: this user is on NetFree. "
                 "find_on_youtube already removes blocked videos, so use only the "
                 "candidates it returns.")
    # Cache breakpoint: the tool definitions + system prompt are identical in every
    # call, so they are read from cache at a fraction of the price. They are kept
    # for 1 hour (the longest option), so a pause between requests doesn't force
    # an expensive re-write. 1-hour writes cost more, but happen far less often.
    return [{"type": "text", "text": text,
             "cache_control": {"type": "ephemeral", "ttl": "1h"}}]


def as_tool(fn, external: bool = False, memory_data: bool = False):
    """Wrap one of our functions as a Claude tool that returns a JSON string.

    Every call is recorded in the session log, and errors are printed to the
    console. In NetFree mode, items with offensive words are removed from the
    results of these tools before they are sent to Claude, so NetFree won't
    block the request:
    - external=True: text from YouTube/Last.fm (if everything is removed, the
      agent gets a clear error);
    - memory_data=True: the user's saved songs (removed items are just noted).
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = fn(*args, **kwargs)
        except Exception as e:  # a bug in a tool must not crash the whole request
            result = {"error": f"The tool crashed: {type(e).__name__}: {e}"}
        note = ""
        if (external or memory_data) and netfree.is_enabled():
            result, note = netfree.filter_with_report(result, error_if_all_removed=external)
        session_log.record_call(fn.__name__, dict(kwargs), result, time.time() - start, note)
        for error in session_log.errors_in(result)[:3]:
            print(f"      ↳ {fn.__name__}: שגיאה: {error[:150]}")
        if note:
            print(f"      ↳ {fn.__name__}: נטפרי: {note}")
        return json.dumps(result, ensure_ascii=False)
    return beta_tool(wrapper)


TOOLS = [
    as_tool(memory.get_music_memory, memory_data=True),
    as_tool(memory.remember_songs),
    as_tool(memory.record_playlist),
    as_tool(memory.get_playlist_songs, memory_data=True),
    as_tool(memory.update_user_settings),
    as_tool(lastfm.get_similar_tracks, external=True),
    as_tool(lastfm.get_similar_artists, external=True),
    as_tool(lastfm.get_artist_top_tracks, external=True),
    as_tool(lastfm.get_track_tags, external=True),
    as_tool(lastfm.get_tracks_by_tag, external=True),
    as_tool(lastfm.find_artist_for_song, external=True),
    as_tool(youtube.find_on_youtube, external=True),
    as_tool(youtube.create_playlist),
    as_tool(youtube.add_to_playlist),
    as_tool(youtube_download.download_songs_from_youtube),
    as_tool(session_log.get_session_log),
]


# Tools that change something. After a NetFree block they must not be repeated.
CHANGING_TOOLS = {"create_playlist", "add_to_playlist", "record_playlist",
                  "remember_songs", "download_songs_from_youtube"}


class NetFreeBlocked(Exception):
    """NetFree blocked the conversation with Claude, even after a retry."""

    def __init__(self, tools_used: list[str]):
        super().__init__("Blocked by NetFree")
        self.tools_used = tools_used


def _run(client: Anthropic, messages: list[dict], tools_used: list[str],
         usage: dict) -> str:
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        max_iterations=MAX_ITERATIONS,
        system=system_prompt(),
        tools=TOOLS,
        messages=messages,
        # Also cache the conversation so far: each tool round re-sends everything
        # before it, so every round reads the previous rounds from cache.
        cache_control={"type": "ephemeral"},
    )
    final_message = None
    for message in runner:
        for block in message.content:
            if block.type == "tool_use":
                print(f"  [tool] {block.name}")
                tools_used.append(block.name)
        u = message.usage
        usage["input"] += u.input_tokens or 0
        written = getattr(u, "cache_creation_input_tokens", 0) or 0
        usage["cache_write"] += written
        # Split of the cache writes by lifetime (1-hour writes are priced higher).
        details = getattr(u, "cache_creation", None)
        written_1h = (getattr(details, "ephemeral_1h_input_tokens", 0) or 0) if details else 0
        usage["cache_write_1h"] += written_1h
        usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
        usage["output"] += u.output_tokens or 0
        final_message = message
    return "".join(b.text for b in final_message.content if b.type == "text").strip()


# Price in USD per million tokens for MODEL (claude-sonnet-5-5). Update these if
# you change the model: https://platform.claude.com/docs/en/about-claude/pricing
PRICES = {"input": 2.00, "cache_write_5m": 2.50, "cache_write_1h": 4.00,
          "cache_read": 0.10, "output": 10.00}


def _cost(usage: dict) -> float:
    write_5m = usage["cache_write"] - usage["cache_write_1h"]
    return (usage["input"] * PRICES["input"]
            + write_5m * PRICES["cache_write_5m"]
            + usage["cache_write_1h"] * PRICES["cache_write_1h"]
            + usage["cache_read"] * PRICES["cache_read"]
            + usage["output"] * PRICES["output"]) / 1_000_000


def _usage_line(usage: dict) -> str:
    total_in = usage["input"] + usage["cache_write"] + usage["cache_read"]
    return (f"  [tokens] input {total_in:,} (from cache {usage['cache_read']:,}, "
            f"written to cache {usage['cache_write']:,}, uncached {usage['input']:,}) | "
            f"output {usage['output']:,} | cost ≈ ${_cost(usage):.4f}")


def _retry_note(tools_used: list[str]) -> str:
    done = sorted(set(tools_used) & CHANGING_TOOLS)
    return (
        "\n\n[Note from the app, not from the user: NetFree, the user's filtered internet "
        "provider, blocked the previous attempt to handle this request because some text in "
        "the tool results (probably a song or video title) matched its word filter. "
        f"Tools that already ran before the block: {', '.join(tools_used) or 'none'}. "
        f"These already made changes and must NOT be repeated: {', '.join(done) or 'none'}. "
        "Use get_music_memory / get_playlist_songs to see what was already done. If the task "
        "is not finished, continue it with clean songs only. Start your reply by telling the "
        "user (in Hebrew) that the first attempt was blocked by NetFree and was recovered.]"
    )


KEEP_LOGS_FOR = 15  # requests; older app logs are dropped from history to save tokens


def _assistant_turn(reply: str, number: int) -> dict:
    """The assistant's reply plus the app log of what happened in this request."""
    return {"role": "assistant", "content": [
        {"type": "text", "text": reply or "(no reply)"},
        {"type": "text", "text": session_log.request_notes(number)},
    ]}


def _trim_old_logs(history: list[dict]) -> None:
    turns = [m for m in history if m["role"] == "assistant" and isinstance(m["content"], list)]
    for message in turns[:-KEEP_LOGS_FOR]:
        message["content"] = message["content"][:1]


def ask(client: Anthropic, history: list[dict], user_text: str) -> str:
    history.append({"role": "user", "content": user_text})
    number = session_log.start_request(user_text)
    tools_used: list[str] = []
    usage = {"input": 0, "cache_write": 0, "cache_write_1h": 0, "cache_read": 0, "output": 0}
    try:
        try:
            reply = _run(client, list(history), tools_used, usage)
        except Exception as e:
            if not netfree.is_api_block(e):
                raise
            # NetFree blocked the request. Retry once without the blocked tool
            # results: a fresh request that only says what was already done.
            netfree.enable("זוהתה חסימה של נטפרי")
            if not memory.get_settings().get("netfree"):
                memory.update_user_settings(netfree="on")  # remember for next sessions
                session_log.record_event("NetFree mode turned on and saved, because "
                                         "NetFree blocked a request.")
            session_log.mark_calls_blocked()
            session_log.record_event("NetFree blocked the request to Claude (word filter on "
                                     "tool results); retried once without those results.")
            print("  [netfree] הבקשה ל-Claude נחסמה על ידי נטפרי - מנסה שוב בלי התוכן שנחסם...")
            retry_messages = history[:-1] + [{"role": "user",
                                              "content": user_text + _retry_note(tools_used)}]
            try:
                reply = _run(client, retry_messages, tools_used, usage)
            except Exception as e2:
                if netfree.is_api_block(e2):
                    session_log.mark_calls_blocked()
                    session_log.record_event("NetFree blocked the retry too. The user got no "
                                             "reply to this request.")
                    raise NetFreeBlocked(tools_used) from e2
                raise
    except NetFreeBlocked:
        session_log.end_request(None, "blocked_by_netfree")
        history.append(_assistant_turn("(No reply: blocked by NetFree.)", number))
        raise
    except Exception as e:
        session_log.record_event(f"The request failed with an error: {type(e).__name__}: {e}")
        session_log.end_request(None, "error")
        history.append(_assistant_turn("(No reply: the request failed with an error.)", number))
        print(_usage_line(usage))
        raise

    session_log.end_request(reply, "ok")
    # The history keeps the reply plus a compact log of the tools and errors,
    # so later requests know what happened. Full details: get_session_log.
    history.append(_assistant_turn(reply, number))
    _trim_old_logs(history)
    print(_usage_line(usage))
    return reply


def _blocked_message(tools_used: list[str]) -> str:
    done = sorted(set(tools_used) & CHANGING_TOOLS)
    lines = ["⛔ נטפרי חסם את התקשורת עם Claude, גם אחרי ניסיון נוסף.",
             "הסיבה: סינון המילים של נטפרי מצא מילה אסורה בטקסט שנשלח (כנראה שם של שיר או סרטון).",
             "זו לא תקלה בקוד."]
    if done:
        lines.append(f"פעולות שכבר בוצעו לפני החסימה: {', '.join(done)}.")
        if "create_playlist" in done:
            lines.append("הפלייליסט כנראה נוצר - אפשר לבדוק ביוטיוב.")
    lines.append("אפשר לנסות שוב בניסוח אחר, או לבקש מנטפרי פטור מסינון לכתובת api.anthropic.com.")
    return "\n".join(lines)


def main():
    client = Anthropic()
    history: list[dict] = []
    print("Music agent ready. Type 'exit' to quit.")
    print(netfree.init(memory.get_settings().get("netfree")))
    session_log.record_event(f"Startup. {netfree.status_line()}. "
                             f"Certificate issuer: {netfree.certificate_issuer()}")
    print(f"יומן הסשן נשמר ב-{session_log.LOG_FILE.name}\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_text:
            continue
        if user_text.lower() in ("exit", "quit", "יציאה"):
            break

        try:
            reply = ask(client, history, user_text)
        except NetFreeBlocked as e:
            # ask() already kept the request and its log in history.
            print(f"\n{_blocked_message(e.tools_used)}\n")
            continue
        except Exception as e:
            print(f"\nError: {e}\n")
            continue

        print(f"\nAgent: {reply}\n")


if __name__ == "__main__":
    main()
