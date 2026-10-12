import json
from pathlib import Path

import httplib2
from google.auth.exceptions import GoogleAuthError, RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

PROJECT_DIR = Path(__file__).resolve().parent.parent
CLIENT_SECRET_FILE = PROJECT_DIR / "client_secret.json"
TOKEN_FILE = PROJECT_DIR / "token.json"
CACHE_FILE = PROJECT_DIR / "youtube_cache.json"

SCOPES = ["https://www.googleapis.com/auth/youtube"]
MUSIC_CATEGORY_ID = "10"

_service = None


# ================== Auth ==================

def _run_login_flow() -> Credentials:
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_FILE), SCOPES)
    return flow.run_local_server(port=0)


def _get_credentials() -> Credentials:
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                # Refresh tokens of apps in "Testing" mode expire after 7 days.
                creds = _run_login_flow()
        else:
            creds = _run_login_flow()
        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    return creds


def _get_service():
    global _service
    if _service is None:
        _service = build("youtube", "v3", credentials=_get_credentials())
    return _service


# ================== Cache ==================
# Each search costs 100 of the 10,000 daily quota units, so results are cached.

def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _cache_key(artist: str, track: str) -> str:
    return f"{artist.strip().lower()}|{track.strip().lower()}"


# ================== Tools ==================

def find_on_youtube(artist: str, track: str) -> list[dict]:
    """Find the YouTube video for a specific song.

    Requires the artist name. If the user gave only the song title, call
    find_artist_for_song first - never search YouTube by title alone.

    Use this for every song before adding it to a playlist. Returns up to 5
    candidate videos. The first result is NOT always the right one - always
    look at every title and channel and pick the one that best matches the
    real song: prefer the official music video, official audio, or the
    artist's "- Topic" channel. Avoid tutorials, covers, live versions,
    karaoke, reactions and lyric videos by other channels unless the user
    asked for them. If none of the candidates is the real song, skip it.

    Args:
        artist: The artist name.
        track: The song title.

    Returns:
        A list of {"video_id", "title", "channel", "url"} dicts, best match
        first. On failure, a list with one {"error"} dict.
    """
    key = _cache_key(artist, track)
    cache = _load_cache()
    if key in cache:
        return cache[key]

    try:
        response = _get_service().search().list(
            part="snippet",
            q=f"{artist} {track}",
            type="video",
            videoCategoryId=MUSIC_CATEGORY_ID,
            maxResults=5,
        ).execute()
    except HttpError as e:
        if e.resp.status == 403 and "quota" in str(e).lower():
            return [{"error": "YouTube daily quota exceeded. Try again tomorrow."}]
        return [{"error": f"YouTube search failed for {artist} - {track}: {e}"}]
    except Exception as e:
        return [{"error": _other_error_message(e, f"Searching for {artist} - {track}")}]

    results = [
        {
            "video_id": item["id"]["videoId"],
            "title": item["snippet"]["title"],
            "channel": item["snippet"]["channelTitle"],
            "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
        }
        for item in response.get("items", [])
    ]

    if not results:
        return [{"error": f"No YouTube video found for {artist} - {track}."}]

    cache[key] = results
    _save_cache(cache)
    return results


def _youtube_error_message(e: HttpError, action: str) -> str:
    text = str(e).lower()
    if "quota" in text:
        return "YouTube daily quota exceeded. Try again tomorrow."
    if "youtubesignuprequired" in text:
        return ("The YouTube account has no channel. The user must create a channel "
                "on youtube.com before playlists can be created.")
    return f"{action} failed: {e}"


def _other_error_message(e: Exception, action: str) -> str:
    """Errors that aren't YouTube API errors: login, missing files, network."""
    if isinstance(e, FileNotFoundError):
        return (f"{action} failed: missing file {e.filename}. Tell the user that "
                "client_secret.json must be in the project folder, and stop.")
    if isinstance(e, GoogleAuthError):
        return (f"{action} failed: YouTube login problem ({e}). Tell the user to delete "
                "token.json and run the agent again to log in, and stop.")
    if isinstance(e, (OSError, httplib2.HttpLib2Error)):
        return (f"{action} failed: could not reach YouTube (network error: {e}). "
                "Try again once; if it fails again, tell the user to check the "
                "internet connection, and stop.")
    return f"{action} failed: unexpected error {type(e).__name__}: {e}"


def add_to_playlist(playlist_id: str, video_ids: list[str]) -> dict:
    """Add songs to an existing YouTube playlist.

    Use this to extend a playlist that was already created, for example when
    the user asks for "a few more songs" in the same playlist.

    Args:
        playlist_id: The playlist ID returned by create_playlist.
        video_ids: YouTube video IDs to add, in the desired order.
            Get them from find_on_youtube.

    Returns:
        {"added": number of videos added, "failed": list of {"video_id", "error"}}.
        If the run stopped early (quota or network), "not_attempted" lists the
        video IDs that were never tried; they can be added later with this tool.
    """
    try:
        service = _get_service()
    except Exception as e:
        message = _other_error_message(e, "Connecting to YouTube")
        return {"added": 0, "failed": [], "not_attempted": list(video_ids), "error": message}

    added = 0
    failed = []
    for index, video_id in enumerate(video_ids):
        try:
            service.playlistItems().insert(
                part="snippet",
                body={
                    "snippet": {
                        "playlistId": playlist_id,
                        "resourceId": {"kind": "youtube#video", "videoId": video_id},
                    }
                },
            ).execute()
            added += 1
        except HttpError as e:
            message = _youtube_error_message(e, f"Adding video {video_id}")
            failed.append({"video_id": video_id, "error": message})
            if "quota" in message.lower():
                # No point trying the rest today.
                return {"added": added, "failed": failed,
                        "not_attempted": list(video_ids[index + 1:])}
        except Exception as e:
            # Network or login problem: the rest would fail the same way.
            failed.append({"video_id": video_id,
                           "error": _other_error_message(e, f"Adding video {video_id}")})
            return {"added": added, "failed": failed,
                    "not_attempted": list(video_ids[index + 1:])}
    return {"added": added, "failed": failed}


def create_playlist(name: str, video_ids: list[str], description: str = "",
                    privacy: str = "private") -> dict:
    """Create a new YouTube playlist in the user's account and add songs to it.

    Use this as the final step, after choosing the songs and finding each one
    with find_on_youtube. Remove duplicates before calling.

    Args:
        name: Playlist title. Short and descriptive, in the user's language.
        video_ids: YouTube video IDs to add, in the desired order.
        description: Optional short description of the playlist's style.
        privacy: "private" (default), "unlisted" or "public".
            Only use "public" or "unlisted" if the user asked for it.

    Returns:
        {"playlist_id", "url", "added", "failed"}, or {"error"} if the
        playlist could not be created.
    """
    if privacy not in ("private", "unlisted", "public"):
        privacy = "private"

    try:
        playlist = _get_service().playlists().insert(
            part="snippet,status",
            body={
                "snippet": {"title": name, "description": description},
                "status": {"privacyStatus": privacy},
            },
        ).execute()
    except HttpError as e:
        return {"error": _youtube_error_message(e, "Creating playlist")}
    except Exception as e:
        return {"error": _other_error_message(e, "Creating playlist")}

    playlist_id = playlist["id"]
    result = add_to_playlist(playlist_id, video_ids)
    return {
        "playlist_id": playlist_id,
        "url": f"https://www.youtube.com/playlist?list={playlist_id}",
        **result,
    }


if __name__ == "__main__":
    from pprint import pprint

    print("--- found ---")
    pprint(find_on_youtube("Coldplay", "Yellow"))
    print("--- same search again (should come from cache, no quota used) ---")
    pprint(find_on_youtube("coldplay", "yellow"))

    print("--- create test playlist ---")
    songs = [("Coldplay", "Yellow"), ("Keane", "Somewhere Only We Know")]
    ids = []
    for artist, track in songs:
        found = find_on_youtube(artist, track)
        if "error" not in found[0]:
            ids.append(found[0]["video_id"])
    pprint(create_playlist("music-agent test", ids, description="Test playlist"))