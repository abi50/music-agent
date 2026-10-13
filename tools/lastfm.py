import functools
import os

import pylast
from dotenv import load_dotenv

from tools import netfree

load_dotenv()

_network = pylast.LastFMNetwork(
    api_key=os.environ["LASTFM_API_KEY"],
    api_secret=os.environ["LASTFM_API_SECRET"],
)


# ================== Error handling ==================
# Every tool catches all errors and returns them as {"error": ...}, so a
# Last.fm problem never crashes the agent; the model reads the message and adapts.

_KEY_PROBLEMS = {pylast.STATUS_INVALID_API_KEY, pylast.STATUS_API_KEY_SUSPENDED,
                 pylast.STATUS_AUTH_FAILED, pylast.STATUS_INVALID_SIGNATURE}
_TEMPORARY = {pylast.STATUS_OFFLINE, pylast.STATUS_TEMPORARILY_UNAVAILABLE,
              pylast.STATUS_OPERATION_FAILED}


# ================== Circuit breaker ==================
# On some networks (e.g. NetFree) Last.fm is blocked: every call fails with an
# empty or unreadable response. After a few such failures in a row, Last.fm is
# turned off for the rest of the session, so the agent stops wasting rounds
# (and tokens) on it and picks songs from its own knowledge instead.

MAX_CONNECTION_FAILURES = 2
_breaker = {"failures": 0, "open": False}

LASTFM_TOOLS = ("get_similar_tracks, get_similar_artists, get_artist_top_tracks, "
                "get_track_tags, get_tracks_by_tag, find_artist_for_song")


def _unavailable_message() -> str:
    cause = ("it is probably blocked by NetFree on the user's network"
             if netfree.is_enabled() else "it cannot be reached from this network right now")
    return (f"LAST.FM IS UNAVAILABLE: {cause}. It is turned off for the rest of this "
            f"session - do NOT call any Last.fm tool again ({LASTFM_TOOLS}). Choose songs "
            "from your own music knowledge, verify each one with find_on_youtube, and "
            "identify unknown artists yourself (ask the user if unsure). Tell the user "
            "briefly that Last.fm is not available, so the picks are based on your knowledge.")


def _guarded(fn):
    """Skip Last.fm entirely once the breaker is open; reset it after a success."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if _breaker["open"]:
            error = {"error": _unavailable_message()}
            return error if fn is find_artist_for_song_impl else [error]
        result = fn(*args, **kwargs)
        failed = (isinstance(result, dict) and "error" in result) or \
                 (isinstance(result, list) and result and "error" in result[0])
        if not failed:
            _breaker["failures"] = 0
        return result
    return wrapper


def _connection_failure() -> str:
    """Count a failure that looks like a blocked or broken connection."""
    _breaker["failures"] += 1
    if _breaker["failures"] >= MAX_CONNECTION_FAILURES:
        _breaker["open"] = True
        return _unavailable_message()
    return ""


def _error(exc: Exception, not_found: str) -> list[dict]:
    """Turn any exception into a clear error message for the model."""
    if not isinstance(exc, pylast.WSError):
        # Not a normal Last.fm answer (like "not found") but an empty, broken
        # or blocked response.
        message = _connection_failure()
        if message:
            return [{"error": message}]
    if isinstance(exc, pylast.WSError):
        status = str(exc.status)
        if status in {str(s) for s in _KEY_PROBLEMS}:
            message = ("Last.fm rejected the API key. Tell the user to check "
                       "LASTFM_API_KEY and LASTFM_API_SECRET in the .env file, and stop.")
        elif status == str(pylast.STATUS_RATE_LIMIT_EXCEEDED):
            message = "Last.fm rate limit exceeded. Wait a moment, then try again once."
        elif status in {str(s) for s in _TEMPORARY}:
            message = (f"Last.fm is temporarily unavailable ({exc}). Try again once; "
                       "if it fails again, tell the user and stop.")
        else:
            message = f"{not_found} ({exc})"
    elif isinstance(exc, pylast.NetworkError):
        message = (f"Could not reach Last.fm (network error: {exc}). Try again once; "
                   "if it fails again, tell the user to check the internet connection, and stop.")
    elif isinstance(exc, pylast.MalformedResponseError):
        message = "Last.fm returned an invalid response. Try again once."
    else:
        hint = " (on NetFree, Last.fm is probably blocked)" if netfree.is_enabled() else ""
        message = (f"Last.fm returned an unreadable response{hint}: "
                   f"{type(exc).__name__}: {exc}. Try one more Last.fm call at most.")
    return [{"error": message}]


def get_similar_tracks(artist: str, track: str, limit: int = 20) -> list[dict]:
    """Find songs that sound similar to a specific song.

    Use this when the user names a song they like and wants more songs like it.
    Requires the artist name. If the user gave only the song title, call
    find_artist_for_song first to find out who sings it.

    Args:
        artist: The artist name, in English if possible (e.g. "Coldplay").
        track: The song title (e.g. "Yellow").
        limit: Maximum number of similar songs to return.

    Returns:
        A list of {"artist", "track", "match"} dicts, where match is a
        similarity score from 0 to 1. On failure, a list with one {"error"} dict.
    """
    try:
        similar = _network.get_track(artist, track).get_similar(limit=limit)
    except Exception as e:
        return _error(e, f"Track not found on Last.fm: {artist} - {track}")
    if not similar:
        return [{"error": f"No similar songs found for {artist} - {track}. "
                          "Try get_similar_artists or get_track_tags instead."}]
    return [
        {"artist": s.item.artist.name, "track": s.item.title, "match": round(float(s.match), 3)}
        for s in similar
    ]


def get_similar_artists(artist: str, limit: int = 10) -> list[dict]:
    """Find artists whose style is similar to a given artist.

    Use this when the user names an artist they like and wants to discover
    other artists, or to widen a playlist beyond a single artist.

    Args:
        artist: The artist name, in English if possible.
        limit: Maximum number of similar artists to return.

    Returns:
        A list of {"artist", "match"} dicts, where match is a similarity
        score from 0 to 1. On failure, a list with one {"error"} dict.
    """
    try:
        similar = _network.get_artist(artist).get_similar(limit=limit)
    except Exception as e:
        return _error(e, f"Artist not found on Last.fm: {artist}")
    if not similar:
        return [{"error": f"No similar artists found for {artist}. "
                          "Try get_artist_top_tracks and get_track_tags instead."}]
    return [{"artist": s.item.name, "match": round(float(s.match), 3)} for s in similar]


def get_artist_top_tracks(artist: str, limit: int = 10) -> list[dict]:
    """Get an artist's most popular songs.

    Use this when the user asks for songs by a specific artist, or after
    get_similar_artists to pick concrete songs from each suggested artist.

    Args:
        artist: The artist name, in English if possible.
        limit: Maximum number of songs to return.

    Returns:
        A list of {"artist", "track"} dicts, ordered by popularity.
        On failure, a list with one {"error"} dict.
    """
    try:
        top = _network.get_artist(artist).get_top_tracks(limit=limit)
    except Exception as e:
        return _error(e, f"Artist not found on Last.fm: {artist}")
    if not top:
        return [{"error": f"No songs found for artist {artist}. "
                          "Check the spelling or try the English name."}]
    # Use Last.fm's canonical artist name, not the user's spelling
    # (e.g. "imagin dragons" -> "Imagine Dragons").
    return [{"artist": t.item.artist.name, "track": t.item.title} for t in top]


def get_track_tags(artist: str, track: str, limit: int = 10) -> list[dict]:
    """Get the style tags that listeners have given to a specific song.

    Use this to understand the style of songs the user likes, then pass the
    most relevant tags to get_tracks_by_tag. This is the best way to turn
    "more songs like these" into concrete style tags, instead of guessing.
    Requires the artist name. If the user gave only the song title, call
    find_artist_for_song first to find out who sings it.

    Args:
        artist: The artist name, in English if possible.
        track: The song title.
        limit: Maximum number of tags to return.

    Returns:
        A list of {"tag", "weight"} dicts, most relevant first.
        On failure, a list with one {"error"} dict.
    """
    try:
        tags = _network.get_track(artist, track).get_top_tags(limit=limit)
    except Exception as e:
        return _error(e, f"Track not found on Last.fm: {artist} - {track}")
    if not tags:
        return [{"error": f"No tags found for {artist} - {track}. "
                          "Try get_similar_tracks instead."}]
    return [{"tag": t.item.name, "weight": int(t.weight)} for t in tags]


def get_tracks_by_tag(tag: str, limit: int = 20) -> list[dict]:
    """Get popular songs for a genre, mood, era or other style tag.

    Use this when the user describes a style rather than a specific song or
    artist. Translate the user's free description (in any language) into one
    or more common Last.fm tags, and call this once per tag if needed.

    Args:
        tag: A Last.fm tag in English, usually lowercase
            (e.g. "indie folk", "chillout", "acoustic", "80s", "israeli").
        limit: Maximum number of songs to return.

    Returns:
        A list of {"artist", "track"} dicts. On failure, a list with one
        {"error"} dict.
    """
    try:
        top = _network.get_tag(tag).get_top_tracks(limit=limit)
    except Exception as e:
        return _error(e, f"Tag not found on Last.fm: {tag}")
    if not top:
        return [{"error": f"No songs found for tag '{tag}'. Try a simpler or more common tag, "
                          "or use get_track_tags on a song the user likes."}]
    return [{"artist": t.item.artist.name, "track": t.item.title} for t in top]



def find_artist_for_song_impl(track: str, limit: int = 8) -> dict:
    """Find out WHO SINGS a song, when the user gave only the song title without the artist.

    This is the only tool for identifying an unknown artist. Use it ONLY when
    the user names a song but not its artist (e.g. "add Bohemian Rhapsody",
    "download Hello"). Never guess the artist yourself.
    Do NOT use it when the artist is already known, and do NOT use it to find
    similar songs or recommendations - it only identifies the one song the
    user meant. After it, continue with the normal tools (get_similar_tracks,
    find_on_youtube, etc.) using the artist it found.

    Args:
        track: The song title exactly as the user wrote it (any language).
            If nothing is found, try once more with the other spelling
            (English <-> Hebrew).
        limit: Maximum number of candidates to return.

    Returns:
        {"query", "candidates": [{"artist", "track", "listeners"}],
         "clear_match": true/false}, candidates sorted by popularity.
        - clear_match true: one song is by far the best known with this title.
          Use the first candidate, and mention the artist you chose in your reply.
        - clear_match false: several real songs share this title. Ask the user
          which artist they meant (show the top 2-4 candidates) before building
          a playlist or downloading anything.
        On failure, {"error"}.
    """
    try:
        results = _network.search_for_track("", track).get_next_page()
    except Exception as e:
        return {"error": _error(e, f"Song search failed for '{track}'")[0]["error"]}

    candidates = {}
    for t in results:
        key = (t.artist.name.strip().lower(), t.title.strip().lower())
        listeners = t.listener_count or 0
        if key not in candidates or listeners > candidates[key]["listeners"]:
            candidates[key] = {"artist": t.artist.name, "track": t.title, "listeners": listeners}

    if not candidates:
        return {"error": f"No song titled '{track}' was found on Last.fm. Try the other "
                         "spelling (English/Hebrew) once; if still nothing, ask the user "
                         "who sings it."}

    ranked = sorted(candidates.values(), key=lambda c: c["listeners"], reverse=True)[:limit]
    # Clear match = the top song has at least 3x the listeners of the next artist's song.
    others = [c for c in ranked[1:] if c["artist"].lower() != ranked[0]["artist"].lower()]
    clear = not others or ranked[0]["listeners"] >= 3 * max(others[0]["listeners"], 1)
    return {"query": track, "candidates": ranked, "clear_match": clear}



# Public tools, protected by the circuit breaker.
get_similar_tracks = _guarded(get_similar_tracks)
get_similar_artists = _guarded(get_similar_artists)
get_artist_top_tracks = _guarded(get_artist_top_tracks)
get_track_tags = _guarded(get_track_tags)
get_tracks_by_tag = _guarded(get_tracks_by_tag)
find_artist_for_song = _guarded(find_artist_for_song_impl)
find_artist_for_song.__name__ = "find_artist_for_song"

if __name__ == "__main__":
    from pprint import pprint

    print("--- similar tracks ---")
    pprint(get_similar_tracks("tom odell", "another love", limit=5))
    print("--- similar artists ---")
    pprint(get_similar_artists("imagin dragons", limit=5))
    print("--- top tracks ---")
    pprint(get_artist_top_tracks("sia", limit=5))
    print("--- track tags ---")
    pprint(get_track_tags("Coldplay", "Yellow"))
    print("--- by tag ---")
    pprint(get_tracks_by_tag("indie folk", limit=5))
    print("--- artist for song ---")
    pprint(find_artist_for_song("Hello"))
    print("--- not found ---")
    pprint(get_similar_tracks("NoSuchArtist123", "NoSuchSong", limit=5))