import json
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MEMORY_FILE = PROJECT_DIR / "music_memory.json"


def _empty_memory() -> dict:
    return {"liked": [], "disliked": [], "playlists": []}


def _load() -> dict:
    if MEMORY_FILE.exists():
        try:
            data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
            for key, value in _empty_memory().items():
                data.setdefault(key, value)
            return data
        except json.JSONDecodeError:
            pass
    return _empty_memory()


def _save(data: dict) -> None:
    MEMORY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _key(song: dict) -> str:
    return f"{song.get('artist', '').strip().lower()}|{song.get('track', '').strip().lower()}"


# ================== Tools ==================

def remember_songs(songs: list[dict], opinion: str) -> dict:
    """Save songs the user likes or dislikes, so future suggestions match their taste.

    Use this whenever the user says they like or dislike specific songs,
    including songs they mention as examples ("I love X and Y").
    A song moves from one list to the other if the user changes their mind.

    Args:
        songs: A list of {"artist", "track"} dicts.
        opinion: "liked" or "disliked".

    Returns:
        {"saved": number of new songs saved, "opinion": opinion}, or {"error"}.
    """
    if opinion not in ("liked", "disliked"):
        return {"error": 'opinion must be "liked" or "disliked".'}

    other = "disliked" if opinion == "liked" else "liked"
    data = _load()
    new_keys = {_key(s) for s in songs}

    # A song can't be both liked and disliked.
    data[other] = [s for s in data[other] if _key(s) not in new_keys]

    existing = {_key(s) for s in data[opinion]}
    saved = 0
    for song in songs:
        if _key(song) not in existing:
            data[opinion].append({"artist": song["artist"], "track": song["track"]})
            existing.add(_key(song))
            saved += 1

    _save(data)
    return {"saved": saved, "opinion": opinion}


def get_music_memory() -> dict:
    """Get everything known about the user's music taste.

    Call this at the start of every request. Use the liked songs as seeds for
    recommendations, never suggest disliked songs, and avoid songs that
    already appeared in previous playlists unless the user asks for them.

    Returns:
        {"liked": [...], "disliked": [...], "already_suggested": [...],
         "playlists": [{"name", "url", "created", "song_count"}]}.
    """
    data = _load()
    suggested = {}
    for playlist in data["playlists"]:
        for song in playlist["songs"]:
            # Only artist + track here; video details come from get_playlist_songs.
            suggested[_key(song)] = {"artist": song["artist"], "track": song["track"]}
    return {
        "liked": data["liked"],
        "disliked": data["disliked"],
        "already_suggested": list(suggested.values()),
        "playlists": [
            {
                "name": p["name"],
                "playlist_id": p["playlist_id"],
                "url": p["url"],
                "created": p["created"],
                "song_count": len(p["songs"]),
            }
            for p in data["playlists"]
        ],
    }


def _clean_song(song: dict) -> dict:
    clean = {"artist": song["artist"], "track": song["track"]}
    video_id = (song.get("video_id") or "").strip()
    if video_id:
        clean["video_id"] = video_id
        clean["url"] = f"https://www.youtube.com/watch?v={video_id}"
    return clean


def record_playlist(name: str, playlist_id: str, url: str, songs: list[dict]) -> dict:
    """Save a playlist that was created or extended, so its songs aren't suggested again.

    Call this right after create_playlist or add_to_playlist succeeds,
    with the songs that were actually added.

    Args:
        name: The playlist title.
        playlist_id: The ID returned by create_playlist.
        url: The playlist URL.
        songs: A list of {"artist", "track", "video_id"} dicts that were added,
            where video_id is the exact YouTube video ID you added for that song.
            Always include video_id, so the songs can be downloaded later
            without searching YouTube again.

    Returns:
        {"recorded": playlist name, "song_count": total songs in that playlist}.
    """
    data = _load()
    clean = [_clean_song(s) for s in songs]

    for playlist in data["playlists"]:
        if playlist["playlist_id"] == playlist_id:
            known = {_key(s): s for s in playlist["songs"]}
            for song in clean:
                existing = known.get(_key(song))
                if existing is None:
                    playlist["songs"].append(song)
                    known[_key(song)] = song
                elif "video_id" in song and "video_id" not in existing:
                    # Fill in the video for songs saved before video IDs were kept.
                    existing.update(video_id=song["video_id"], url=song["url"])
            _save(data)
            return {"recorded": playlist["name"], "song_count": len(playlist["songs"])}

    data["playlists"].append({
        "name": name,
        "playlist_id": playlist_id,
        "url": url,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "songs": clean,
    })
    _save(data)
    return {"recorded": name, "song_count": len(clean)}


def get_playlist_songs(playlist_id: str) -> dict:
    """Get the songs of a playlist that was saved with record_playlist.

    Use this when the user wants to download songs from one of their playlists
    (for example "download the playlist you just made"). Get the playlist_id
    from get_music_memory. Songs that include a "url" can be passed directly to
    download_songs_from_youtube. Songs without a "url" (older playlists) must
    be looked up with find_on_youtube first.

    Args:
        playlist_id: The playlist ID, as shown in get_music_memory.

    Returns:
        {"name", "url", "songs": [{"artist", "track", "video_id"?, "url"?}],
         "songs_missing_url": count}, or {"error"} if the playlist is unknown.
    """
    for playlist in _load()["playlists"]:
        if playlist["playlist_id"] == playlist_id:
            songs = playlist["songs"]
            return {
                "name": playlist["name"],
                "url": playlist["url"],
                "songs": songs,
                "songs_missing_url": sum(1 for s in songs if not s.get("url")),
            }
    return {"error": f"No saved playlist with ID {playlist_id}. "
                     "Check the playlists list from get_music_memory."}


if __name__ == "__main__":
    from pprint import pprint

    print("--- like two songs ---")
    pprint(remember_songs(
        [{"artist": "Coldplay", "track": "Yellow"}, {"artist": "Keane", "track": "Somewhere Only We Know"}],
        "liked",
    ))
    print("--- dislike one (moves it from liked) ---")
    pprint(remember_songs([{"artist": "Keane", "track": "Somewhere Only We Know"}], "disliked"))
    print("--- record a playlist ---")
    pprint(record_playlist("test", "PL_TEST", "https://example.com",
                           [{"artist": "Snow Patrol", "track": "Chasing Cars"}]))
    print("--- memory ---")
    pprint(get_music_memory())