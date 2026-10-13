import functools
import json

from anthropic import Anthropic, beta_tool
from dotenv import load_dotenv

from tools import lastfm, memory, netfree, youtube, youtube_download

load_dotenv()

MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 4096
MAX_ITERATIONS = 60  # safety limit on tool-call rounds per request

SYSTEM_PROMPT = """You are a personal music assistant. You find songs the user will love \
and build YouTube playlists for them. You can also download songs when the user \
explicitly asks for it.

How to work:
1. At the start of every request, call get_music_memory to see what the user likes, \
dislikes and has already received.
2. If the user mentions songs they like or dislike, save them with remember_songs.
3. Songs without an artist: if the user names a song but not who sings it (for \
any purpose - examples, playlist, download), call find_artist_for_song first. \
Never guess the artist yourself. If it returns clear_match true, use the top \
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
- Every song you pass must have a full YouTube URL taken from find_on_youtube results \
(if you only have a video ID, use https://www.youtube.com/watch?v=<ID>). Never guess URLs.
- If the user asks to download songs from one of their playlists (including one you \
just built), get its playlist_id from get_music_memory and call get_playlist_songs. \
Use the saved URLs as they are; only songs without a URL need find_on_youtube. \
If they ask to download without a playlist, find the songs with steps 3-6 and skip \
creating a playlist unless they also want one.
- Quality: use "mp3" unless the user asks for video ("480p", "720p" or "best").
- Folder: leave output_dir empty to save to the computer's Downloads folder. Only \
pass output_dir if the user names a folder: a full path as they wrote it, or a plain \
folder name, which is created inside Downloads.
- After downloading, report which songs succeeded, which failed and why, and the \
folder they were saved to.

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
- In NetFree mode prefer clean, family-friendly songs, and avoid songs whose titles \
contain offensive words: they will likely be blocked."""


def system_prompt() -> list[dict]:
    text = SYSTEM_PROMPT
    if netfree.is_enabled():
        text += ("\n\n" + NETFREE_PROMPT + "\n\nNetFree mode is ON: this user is on NetFree. "
                 "find_on_youtube already removes blocked videos, so use only the "
                 "candidates it returns.")
    # Cache breakpoint: the tool definitions + system prompt are identical in every
    # call, so after the first call they are read from cache at ~10% of the price.
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def as_tool(fn, external: bool = False):
    """Wrap one of our functions as a Claude tool that returns a JSON string.

    external=True marks tools that return text from YouTube/Last.fm. In NetFree
    mode, items with offensive words are removed from their results before the
    results are sent to Claude, so NetFree won't block the request.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if external and netfree.is_enabled():
            result = netfree.filter_result(result)
        return json.dumps(result, ensure_ascii=False)
    return beta_tool(wrapper)


TOOLS = [
    as_tool(memory.get_music_memory),
    as_tool(memory.remember_songs),
    as_tool(memory.record_playlist),
    as_tool(memory.get_playlist_songs),
    as_tool(lastfm.get_similar_tracks, external=True),
    as_tool(lastfm.get_similar_artists, external=True),
    as_tool(lastfm.get_artist_top_tracks, external=True),
    as_tool(lastfm.get_track_tags, external=True),
    as_tool(lastfm.get_tracks_by_tag, external=True),
    as_tool(lastfm.find_artist_for_song, external=True),
    as_tool(youtube.find_on_youtube, external=True),
    as_tool(youtube.create_playlist),
    as_tool(youtube.add_to_playlist),
    as_tool(youtube_download.download_songs_from_youtube)
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
        usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
        usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
        usage["output"] += u.output_tokens or 0
        final_message = message
    return "".join(b.text for b in final_message.content if b.type == "text").strip()


def _usage_line(usage: dict) -> str:
    total_in = usage["input"] + usage["cache_write"] + usage["cache_read"]
    return (f"  [tokens] input {total_in:,} (from cache {usage['cache_read']:,}, "
            f"written to cache {usage['cache_write']:,}, uncached {usage['input']:,}) | "
            f"output {usage['output']:,}")


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


def ask(client: Anthropic, history: list[dict], user_text: str) -> str:
    history.append({"role": "user", "content": user_text})
    tools_used: list[str] = []
    usage = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0}
    try:
        reply = _run(client, list(history), tools_used, usage)
    except Exception as e:
        if not netfree.is_api_block(e):
            raise
        # NetFree blocked the request. Retry once without the blocked tool
        # results: a fresh request that only says what was already done.
        netfree.enable("זוהתה חסימה של נטפרי")
        print("  [netfree] הבקשה ל-Claude נחסמה על ידי נטפרי - מנסה שוב בלי התוכן שנחסם...")
        retry_messages = history[:-1] + [{"role": "user",
                                          "content": user_text + _retry_note(tools_used)}]
        try:
            reply = _run(client, retry_messages, tools_used, usage)
        except Exception as e2:
            if netfree.is_api_block(e2):
                raise NetFreeBlocked(tools_used) from e2
            raise

    # Keep only the text of the final answer in the history; the details of the
    # tool calls are not needed for the next request (memory keeps what matters).
    history.append({"role": "assistant", "content": reply or "(no reply)"})
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
    print(netfree.init() + "\n")

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
            message = _blocked_message(e.tools_used)
            # Keep the request in history, so the agent knows what happened.
            history.append({"role": "assistant",
                            "content": "(The reply to this request was blocked by NetFree. "
                                       f"Tools that ran: {', '.join(e.tools_used) or 'none'}.)"})
            print(f"\n{message}\n")
            continue
        except Exception as e:
            history.pop()  # drop the request that failed
            print(f"\nError: {e}\n")
            continue

        print(f"\nAgent: {reply}\n")


if __name__ == "__main__":
    main()
