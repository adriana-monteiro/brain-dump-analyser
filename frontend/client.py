
# frontend/client.py

import os

import requests


class APIClient:
    def __init__(self, base_url: str | None = None, timeout: int = 120):
        self.base_url = (
            base_url or os.getenv("API_URL", "http://localhost:8000")
        ).rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def request(self, method: str, path: str, **kwargs):
        url = f"{self.base_url}/{path.lstrip('/')}"

        try:
            response = self.session.request(
                method,
                url,
                timeout=self.timeout,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise APIError(
                f"Cannot reach API at {self.base_url}: {exc}"
            ) from exc

        if not response.ok:
            try:
                body = response.json()
                detail = body.get("detail", body)
            except ValueError:
                detail = response.text

            raise APIError(
                f"API error {response.status_code}: {detail}"
            )

        if not response.content:
            return None

        try:
            return response.json()
        except ValueError as exc:
            raise APIError("API returned invalid JSON.") from exc

    def get_topics(self):
        return self.request("GET", "/settings/topics")

    def set_topics(self, topics: list[str]):
        return self.request("PUT", "/settings/topics", json=topics)

    def get_tasks(self):
        return self.request("GET", "/tasks")

    def get_items(
        self,
        kind: str | None = None,
        status: str | None = None,
    ):
        params = {}

        if kind is not None:
            params["kind"] = kind

        if status is not None:
            params["status"] = status

        return self.request("GET", "/items", params=params)

    def chat(self, message: str, conversation_id: str | None = None):
        return self.request(
            "POST",
            "/chat",
            json={
                "message": message,
                "conversation_id": conversation_id,
            },
        )


    def reset_conversation(self, conversation_id: int):
        return self.request(
            "POST",
            f"/conversations/{conversation_id}/reset",
        )


    def resolve_candidate(
        self,
        candidate_id: int,
        action: str,
        text: str | None = None,
    ):
        return self.request(
            "POST",
            f"/candidates/{candidate_id}/resolve",
            json={"action": action, "text": text},
        )

    def get_pending(self, conversation_id: str | None):
        return self.request(
            "GET",
            f"/conversations/{conversation_id}/candidates",
        )

    
    def update_task_status(self, task_id: int, status: str):
        return self.request(
            "PATCH",
            f"/tasks/{task_id}/status",
            json={"status": status},
        )



class APIError(Exception):
    """Raised when an API request fails."""
