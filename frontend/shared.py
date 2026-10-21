
import streamlit as st

from frontend.client import APIClient, APIError


def get_api() -> APIClient:
    return APIClient()


def call_api(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except APIError as exc:
        st.error(str(exc))
        return None


def sync_chat_response(response):
    """Apply a successful chat or candidate-resolution response."""
    st.session_state["cands"] = response["candidates"]
    if response.get("reply"):
        st.session_state["msgs"].append(
            ("assistant", response["reply"])
        )
        # Key the trace by the message's index so the UI can render it beneath
        # that reply. Only chat responses carry one; resolve/reset leave None.
        trace = response.get("trace")
        if trace:
            traces = st.session_state.setdefault("traces", {})
            traces[len(st.session_state["msgs"]) - 1] = trace
    st.session_state["conv"] = (
        None if response["closed"]
        else response["conversation_id"]
    )


def show_item(item, show_status=False):
    st.markdown(f"**{item['text']}**")

    details = [item.get("topic") or "INBOX"]

    if show_status and item.get("status"):
        details.append(item["status"].capitalize())

    if item.get("created_at"):
        details.append(item["created_at"])

    st.caption(" · ".join(details))


def render_collection(kind: str, title: str, description: str):
    api = get_api()

    st.title(title)
    st.caption(description)

    items = call_api(api.get_items, kind=kind)
    if items is None:
        return

    if not items:
        st.info(
            f"No saved {kind}s yet. Add one in Brain dump "
            "and confirm it."
        )
        return

    topics = sorted({
        item.get("topic") or "INBOX"
        for item in items
    })

    search_col, topic_col = st.columns([2, 1])

    with search_col:
        search = st.text_input(
            "Search",
            placeholder=f"Search {kind}s...",
            key=f"{kind}_search",
        ).strip().casefold()

    with topic_col:
        selected_topic = st.selectbox(
            "Topic",
            ["All topics"] + topics,
            key=f"{kind}_topic",
        )

    filtered = [
        item for item in items
        if (
            not search
            or search in item["text"].casefold()
        )
        and (
            selected_topic == "All topics"
            or (item.get("topic") or "INBOX") == selected_topic
        )
    ]

    st.caption(f"{len(filtered)} {kind}{'s' if kind != 'idea' else 's'}")

    if not filtered:
        st.info("No items match these filters.")
        return

    for item in filtered:
        with st.container(border=True):
            show_item(item)
