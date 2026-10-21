
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

    def get_metrics(
        self,
        step: str | None = None,
        model: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ):
        params = {}

        if step is not None:
            params["step"] = step

        if model is not None:
            params["model"] = model

        if status is not None:
            params["status"] = status

        if limit is not None:
            params["limit"] = limit

        return self.request("GET", "/metrics", params=params)

    def get_metrics_summary(
        self,
        step: str | None = None,
        model: str | None = None,
    ):
        params = {}

        if step is not None:
            params["step"] = step

        if model is not None:
            params["model"] = model

        return self.request("GET", "/metrics/summary", params=params)

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

    def update_candidate_kind(self, candidate_id: int, kind: str):
        return self.request(
            "PATCH",
            f"/candidates/{candidate_id}/kind",
            json={"kind": kind},
        )

    def update_candidate_topic(self, candidate_id: int, topic: str):
        return self.request(
            "PATCH",
            f"/candidates/{candidate_id}/topic",
            json={"topic": topic},
        )

    def force_new_candidate(self, candidate_id: int):
        return self.request(
            "POST",
            f"/candidates/{candidate_id}/force-new",
        )

    def get_pending(self, conversation_id: str | None):
        return self.request(
            "GET",
            f"/conversations/{conversation_id}/candidates",
        )

    def get_conversation(self, conversation_id):
        """Full conversation state for rehydrating the UI after a refresh:
        messages, pending candidates, and closed flag."""
        return self.request(
            "GET",
            f"/conversations/{conversation_id}/messages",
        )

    def get_latest_open_conversation(self):
        """Most recent open conversation with pending items (hydrated), or
        {"conversation_id": None} when there is nothing to resume."""
        return self.request(
            "GET",
            "/conversations/latest-open",
        )

    
    def update_task_status(self, task_id: int, status: str):
        return self.request(
            "PATCH",
            f"/tasks/{task_id}/status",
            json={"status": status},
        )



class APIError(Exception):
    """Raised when an API request fails."""
