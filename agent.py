import functools
import json

from anthropic import Anthropic, beta_tool
from dotenv import load_dotenv

from tools import lastfm, memory, youtube, youtube_download

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


def as_tool(fn):
    """Wrap one of our functions as a Claude tool that returns a JSON string."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return json.dumps(fn(*args, **kwargs), ensure_ascii=False)
    return beta_tool(wrapper)


TOOLS = [
    as_tool(memory.get_music_memory),
    as_tool(memory.remember_songs),
    as_tool(memory.record_playlist),
    as_tool(memory.get_playlist_songs),
    as_tool(lastfm.get_similar_tracks),
    as_tool(lastfm.get_similar_artists),
    as_tool(lastfm.get_artist_top_tracks),
    as_tool(lastfm.get_track_tags),
    as_tool(lastfm.get_tracks_by_tag),
    as_tool(lastfm.find_artist_for_song),
    as_tool(youtube.find_on_youtube),
    as_tool(youtube.create_playlist),
    as_tool(youtube.add_to_playlist),
    as_tool(youtube_download.download_songs_from_youtube)
]


def ask(client: Anthropic, history: list[dict], user_text: str) -> str:
    history.append({"role": "user", "content": user_text})

    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        max_iterations=MAX_ITERATIONS,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=list(history),
    )

    final_message = None
    for message in runner:
        for block in message.content:
            if block.type == "tool_use":
                print(f"  [tool] {block.name}")
        final_message = message

    reply = "".join(b.text for b in final_message.content if b.type == "text").strip()
    # Keep only the text of the final answer in the history; the details of the
    # tool calls are not needed for the next request (memory keeps what matters).
    history.append({"role": "assistant", "content": reply or "(no reply)"})
    return reply


def main():
    client = Anthropic()
    history: list[dict] = []
    print("Music agent ready. Type 'exit' to quit.\n")

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
        except Exception as e:
            history.pop()  # drop the request that failed
            print(f"\nError: {e}\n")
            continue

        print(f"\nAgent: {reply}\n")


if __name__ == "__main__":
    main()