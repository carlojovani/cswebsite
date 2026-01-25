import os
import requests

FACEIT_API_BASE = "https://open.faceit.com/data/v4"
FACEIT_DOWNLOADS_BASE = "https://open.faceit.com/downloads/v1"


class FaceitClient:
    def __init__(self, api_key: str | None = None, timeout: int = 30):
        self.api_key = api_key or os.environ.get("FACEIT_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("FACEIT_API_KEY is not set")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def search_player(self, nickname: str, game: str = "cs2") -> dict:
        r = self.session.get(
            f"{FACEIT_API_BASE}/players",
            params={"nickname": nickname, "game": game},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def player_history(self, player_id: str, game: str = "cs2", limit: int = 1) -> dict:
        r = self.session.get(
            f"{FACEIT_API_BASE}/players/{player_id}/history",
            params={"game": game, "offset": 0, "limit": limit},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def match_details(self, match_id: str) -> dict:
        r = self.session.get(
            f"{FACEIT_API_BASE}/matches/{match_id}",
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def get_signed_download_url(self, resource_url: str) -> str:
        r = self.session.get(
            f"{FACEIT_DOWNLOADS_BASE}/download",
            params={"resource_url": resource_url},
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        signed = data.get("signed_url") or data.get("url")
        if not signed:
            raise RuntimeError(f"Unexpected downloads response: {data}")
        return signed
