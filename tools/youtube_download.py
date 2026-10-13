"""YouTube download tool for the music agent.

This module follows the same structure as tools/lastfm.py:
- plain Python functions
- type hints and detailed docstrings
- structured results that can be returned to the agent as JSON
"""

import os
import re
from pathlib import Path
from typing import Any, Optional

try:
    import truststore

    # Use the operating system's certificate store when truststore is installed.
    truststore.inject_into_ssl()
except ImportError:
    # truststore is optional; Python's default SSL configuration remains in use.
    pass

try:
    from yt_dlp import YoutubeDL
except ImportError:
    YoutubeDL = None

from tools import netfree


def _windows_downloads_folder() -> Optional[Path]:
    """The real Downloads folder on Windows, even if the user moved it (e.g. to D:)."""
    try:
        import ctypes
        from ctypes import wintypes
        import uuid

        class GUID(ctypes.Structure):
            _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                        ("Data3", wintypes.WORD), ("Data4", wintypes.BYTE * 8)]

        folder_id = uuid.UUID("{374DE290-123F-4565-9164-39C4925E467B}")  # FOLDERID_Downloads
        guid = GUID()
        ctypes.memmove(ctypes.byref(guid), folder_id.bytes_le, ctypes.sizeof(guid))

        path_ptr = ctypes.c_wchar_p()
        shell32 = ctypes.windll.shell32
        if shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(path_ptr)) != 0:
            return None
        try:
            return Path(path_ptr.value)
        finally:
            ctypes.windll.ole32.CoTaskMemFree(path_ptr)
    except Exception:
        return None


def get_downloads_folder() -> Path:
    """The computer's official Downloads folder."""
    if os.name == "nt":
        folder = _windows_downloads_folder()
        if folder:
            return folder
    return Path.home() / "Downloads"


def _resolve_output_dir(output_dir: str) -> Path:
    """Empty -> Downloads folder. Relative -> a subfolder inside Downloads. Absolute -> as is."""
    output_dir = (output_dir or "").strip().strip('"')
    if not output_dir:
        return get_downloads_folder()
    path = Path(os.path.expandvars(output_dir)).expanduser()
    if not path.is_absolute():
        path = get_downloads_folder() / path
    return path


def _video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", url)
    return match.group(1) if match else ""


def _describe_error(message: str, url: str) -> dict[str, str]:
    """Turn a yt-dlp error into a status + a clear explanation for the agent."""
    message = message.replace("ERROR: ", "").strip()
    kind = netfree.classify_ytdlp_error(message)
    if kind == "netfree":
        netfree.enable("זוהתה חסימה בהורדה")
        if _video_id(url):
            netfree.mark_blocked(_video_id(url))
        return {"status": "blocked_by_netfree",
                "error": "BLOCKED BY NETFREE: this video is not approved on the user's "
                         "filtered network. Not a bug and not a yt-dlp problem - do not "
                         "retry this URL or suggest updating yt-dlp."}
    if kind == "maybe_netfree":
        return {"status": "error",
                "error": "yt-dlp could not load this video. On a filtered network such as "
                         "NetFree this usually means the video is blocked; otherwise yt-dlp "
                         f"may need an update (pip install -U yt-dlp). Details: {message}"}
    if kind == "unavailable":
        return {"status": "error",
                "error": f"The video is unavailable on YouTube (removed, private or "
                         f"restricted). Try another candidate. Details: {message}"}
    return {"status": "error", "error": message}


def download_songs_from_youtube(
        songs: list[dict[str, str]],
        output_dir: str = "",
        quality: str = "mp3",
) -> list[dict[str, Any]]:
    """Download a list of YouTube songs to a local folder.

    Use this when the user explicitly asks to download songs, not when they
    only ask to create a YouTube playlist.

    Args:
        songs: Songs to download. Each item should contain:
            - artist: Artist name
            - track or title: Song title
            - url or youtube_url: A direct YouTube video URL
        output_dir: Destination folder. Leave empty to save to the computer's
            Downloads folder (the default). If the user names a folder, pass
            it here: a full path (e.g. "D:\\Music" or "~/Music") is used as is,
            and a plain folder name (e.g. "Workout") is created inside Downloads.
        quality: One of "mp3", "480p", "720p", or "best".
            "mp3" downloads audio and converts it to MP3 using FFmpeg.

    Returns:
        A list of result dictionaries, one per song. Successful items contain
        artist, track, url, status="success", and output_dir. Failed items
        contain artist, track, url, status and error, where status is:
        - "blocked_by_netfree": the video is blocked on the user's NetFree
          (filtered) network. Do NOT retry the same URL. Call find_on_youtube
          for that song and try one other candidate; if there is none, tell the
          user this song is blocked by NetFree.
        - "error": any other failure; the error text explains it.
        If yt-dlp is missing, returns a single error item.
    """
    if YoutubeDL is None:
        return [{
            "status": "error",
            "error": "Missing dependency: install yt-dlp (pip install yt-dlp).",
        }]

    if not songs:
        return [{"status": "error", "error": "No songs were provided."}]

    destination = _resolve_output_dir(output_dir)
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return [{
            "status": "error",
            "output_dir": str(destination),
            "error": f"Could not create output directory: {exc}",
        }]

    quality_key = quality.strip().lower()
    ydl_opts: dict[str, Any] = {
        "outtmpl": str(destination / "%(title)s.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
        "retries": 5,
        "fragment_retries": 5,
        "no_warnings": True,
        # Errors are returned to the agent, so don't print them to the console.
        "logger": netfree._SilentLogger(),
    }

    if quality_key in {"mp3", "audio", "audio only", "אודיו בלבד (mp3)"}:
        ydl_opts.update({
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }],
        })
    elif quality_key == "480p":
        ydl_opts.update({
            "format": "bestvideo[height<=480]+bestaudio/best[height<=480]",
            "merge_output_format": "mp4",
        })
    elif quality_key == "720p":
        ydl_opts.update({
            "format": "bestvideo[height<=720]+bestaudio/best[height<=720]",
            "merge_output_format": "mp4",
        })
    elif quality_key in {"best", "maximum", "איכות מקסימלית"}:
        ydl_opts.update({
            "format": "bestvideo+bestaudio/best",
            "merge_output_format": "mp4",
        })
    else:
        return [{
            "status": "error",
            "error": "Unsupported quality. Use 'mp3', '480p', '720p', or 'best'.",
        }]

    results: list[dict[str, Any]] = []

    for song in songs:
        artist = song.get("artist", "").strip()
        track = (song.get("track") or song.get("title") or "").strip()
        url = (song.get("url") or song.get("youtube_url") or "").strip()

        if not url:
            results.append({
                "artist": artist,
                "track": track,
                "status": "error",
                "error": "Missing YouTube URL for this song.",
            })
            continue

        try:
            # Use a fresh YoutubeDL instance per song so each result can be
            # reported independently if one download fails.
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            results.append({
                "artist": artist,
                "track": track,
                "url": url,
                "status": "success",
                "output_dir": str(destination.resolve()),
            })
        except Exception as exc:
            results.append({"artist": artist, "track": track, "url": url,
                            **_describe_error(str(exc), url)})

    return results


if __name__ == "__main__":
    # Minimal manual test; replace the sample URL with a video you may
    # lawfully download. This block is not run when the agent imports the tool.
    sample = [{
        "artist": "Example Artist",
        "track": "Example Track",
        "url": "https://www.youtube.com/watch?v=REPLACE_ME",
    }]
    print(download_songs_from_youtube(sample))
