import os
from dotenv import load_dotenv
import anthropic
import pylast
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

load_dotenv()

# Anthropic
client = anthropic.Anthropic()
msg = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=20,
    messages=[{"role": "user", "content": "Say OK"}],
)
print("Anthropic:", msg.content[0].text)

# Last.fm
network = pylast.LastFMNetwork(
    api_key=os.environ["LASTFM_API_KEY"],
    api_secret=os.environ["LASTFM_API_SECRET"],
)
similar = network.get_artist("Coldplay").get_similar(limit=3)
print("Last.fm:", [s.item.name for s in similar])

# YouTube (נפתח דפדפן להתחברות)
flow = InstalledAppFlow.from_client_secrets_file(
    "client_secret.json",
    scopes=["https://www.googleapis.com/auth/youtube"],
)
creds = flow.run_local_server(port=0)
with open("token.json", "w") as f:
    f.write(creds.to_json())
yt = build("youtube", "v3", credentials=creds)
me = yt.channels().list(part="snippet", mine=True).execute()
print("YouTube:", me["items"][0]["snippet"]["title"] if me.get("items") else "מחובר (אין ערוץ)")