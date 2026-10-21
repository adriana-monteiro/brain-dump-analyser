
import streamlit as st

from frontend.shared import (
    call_api,
    get_api,
    sync_chat_response,
)

api = get_api()

state = st.session_state
state.setdefault("conv", None)
state.setdefault("msgs", [])
state.setdefault("cands", [])
# Interpret traces keyed by the index of the assistant message they belong to.
state.setdefault("traces", {})


_OP_LABELS = {
    "confirm": "Confirmed",
    "reject": "Rejected",
    "change_text": "Reworded",
    "change_kind": "Changed kind",
    "change_topic": "Changed topic",
    "needs_human": "Asked you a question",
}


def _render_trace(trace):
    """Show what the interpret agent decided, so edits aren't silent."""
    decisions = trace.get("decisions") or []
    new_dump = (trace.get("new_dump") or "").strip()
    source = trace.get("source", "")

    # Nothing happened — skip the expander to avoid noise.
    if not decisions and not new_dump:
        return

    summary = f"🔍 How this was interpreted ({len(decisions)} decision"
    summary += "s" if len(decisions) != 1 else ""
    if new_dump:
        summary += ", + new item"
    summary += ")"

    with st.expander(summary):
        if source:
            st.caption(f"Resolved via: {source.replace('_', ' ')}")

        for decision in decisions:
            op = decision.get("operation", "")
            label = _OP_LABELS.get(op, op or "action")
            display_no = decision.get("display_no")
            st.markdown(f"**#{display_no} — {label}**")

            before = decision.get("before_text")
            after = decision.get("after_text")
            if op == "change_text" and (before or after):
                st.markdown(f"- was: {before or '—'}")
                st.markdown(f"- now: {after or '—'}")
            elif op == "change_kind" and decision.get("kind"):
                st.markdown(f"- new kind: {decision['kind']}")
            elif op == "change_topic" and decision.get("topic"):
                st.markdown(f"- new topic: {decision['topic']}")
            elif op == "needs_human" and decision.get("question"):
                st.markdown(f"- question: {decision['question']}")

        if new_dump:
            st.markdown(f"**New item added:** {new_dump}")

# Topic picker options. New topics come only via chat, so the panel is a
# constrained picker of existing topics + INBOX.
_topics = call_api(api.get_topics) or []
TOPIC_OPTIONS = list(dict.fromkeys([*_topics, "INBOX"]))


def _sync_conv_url():
    """Mirror the conversation id into the URL so a refresh can find it."""
    if state.conv:
        st.query_params["conv"] = str(state.conv)
    elif "conv" in st.query_params:
        del st.query_params["conv"]


def _apply_hydration(data):
    """Populate session_state from a hydrated conversation payload.

    Returns True when an open conversation was restored. A closed or empty
    payload restores nothing.
    """
    if not data or data.get("conversation_id") is None or data.get("closed"):
        return False

    state.msgs = [
        (m["role"], m["content"]) for m in data.get("messages", [])
    ]
    state.cands = data.get("candidates", [])
    state.conv = data["conversation_id"]
    return True


def _rehydrate():
    """Resume the open review after a refresh wipes session_state.

    Asks the backend for the latest open conversation; ?conv=<id> is only a
    fallback hint, since the URL can't be trusted across a Streamlit refresh.
    """
    if state.conv is not None:
        return

    # Primary path: whatever open conversation still has pending items.
    latest = call_api(api.get_latest_open_conversation)
    if _apply_hydration(latest):
        return

    # Fallback: a conversation id left in the URL from a prior interaction.
    raw = st.query_params.get("conv")
    if not raw:
        return
    try:
        conv_id = int(raw)
    except (TypeError, ValueError):
        del st.query_params["conv"]
        return

    data = call_api(api.get_conversation, conv_id)
    if not _apply_hydration(data) and "conv" in st.query_params:
        # Dead/closed/unreachable id — don't strand the user on it.
        del st.query_params["conv"]


_rehydrate()
_sync_conv_url()

st.title("🧠 Brain dump")
st.caption(
    "Get thoughts out of your head. Review the items before saving them."
)

# Conversation controls
header_col, reset_col = st.columns([4, 1])

with reset_col:
    if st.button(
        "Reset",
        icon=":material/refresh:",
        use_container_width=True,
        disabled=state.conv is None,
    ):
        state["confirm_reset"] = True

if state.get("confirm_reset"):
    st.warning(
        "This erases this conversation's chat history, pending items, "
        "and questions. Confirmed tasks, notes, and ideas are preserved."
    )

    yes_col, no_col = st.columns(2)

    with yes_col:
        if st.button(
            "Reset conversation",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Resetting conversation..."):
                response = call_api(
                    api.reset_conversation,
                    state.conv,
                )

            if response is not None:
                # Clear UI state and candidate edit widget state.
                for key in list(state.keys()):
                    if (
                        key.startswith("candidate_text_")
                        or key.startswith("candidate_server_text_")
                        or key.startswith("candidate_kind_")
                        or key.startswith("candidate_server_kind_")
                        or key.startswith("candidate_topic_")
                        or key.startswith("candidate_server_topic_")
                        or key in {"msgs", "cands"}
                    ):
                        del state[key]

                state["msgs"] = []
                state["cands"] = []
                state["traces"] = {}
                state["conv"] = response["conversation_id"]
                state["confirm_reset"] = False

                _sync_conv_url()
                st.rerun()

    with no_col:
        if st.button("Keep conversation", use_container_width=True):
            state["confirm_reset"] = False
            st.rerun()

prompt = st.chat_input(
    "Dump your thoughts, or reply to a question...",
    disabled=bool(state.get("confirm_reset")),
)

if prompt:
    state.msgs.append(("user", prompt))

    with st.spinner("Processing your thoughts..."):
        response = call_api(
            api.chat,
            prompt,
            state.conv,
        )

    if response is not None:
        sync_chat_response(response)
        _sync_conv_url()

for index, (role, content) in enumerate(state.msgs):
    with st.chat_message(role):
        st.markdown(content)
        trace = state.traces.get(index)
        if role == "assistant" and trace:
            _render_trace(trace)

if state.cands:
    st.divider()
    st.subheader("Review proposed items")
    st.caption(
        "Confirm to save, edit to change the text, "
        "or reject to discard."
    )

    for position, candidate in enumerate(state.cands):
        candidate_id = candidate["id"]
        existing_text = candidate.get("existing_item_text")
        operation = candidate.get("op", "new")

        # Three-wide grid: new column row every third card. This grid is
        # Streamlit's one allowed level of column nesting, so card contents
        # below all stack vertically rather than using more columns().
        if position % 3 == 0:
            grid_cols = st.columns(3)
        card_col = grid_cols[position % 3]

        with card_col, st.container(border=True):
            st.caption(
                f"#{candidate['display_no']} · "
                f"{candidate['kind'].capitalize()} · "
                f"{candidate.get('topic') or 'INBOX'} · "
                f"{operation.replace('_', ' ').capitalize()}"
            )

            if existing_text:
                st.caption(f"Matched saved task: {existing_text}")

                # Override a wrong judge match: save as a new item instead.
                # Kept a panel control, not an agent op, because the agent
                # can't reliably tell a wrong match from a genuine duplicate.
                if st.button(
                    "Force create new task",
                    key=f"force_new_{candidate_id}",
                    help=(
                        "Ignore the match above and save this as a new item."
                    ),
                    use_container_width=True,
                ):
                    with st.spinner("Creating new item..."):
                        response = call_api(
                            api.force_new_candidate,
                            candidate_id,
                        )

                    if response is not None:
                        sync_chat_response(response)
                        st.rerun()

            # Streamlit pins a widget value under its key and ignores `value=`
            # on rerun. So when the server changes the text, reset the pinned
            # value. The companion server_key records the last server text, so
            # a user's unsaved edit (user-driven mismatch) is preserved.
            text_key = f"candidate_text_{candidate_id}"
            server_key = f"candidate_server_text_{candidate_id}"

            if text_key not in state or state.get(server_key) != candidate["text"]:
                state[text_key] = candidate["text"]
                state[server_key] = candidate["text"]

            edited_text = st.text_input(
                "Item text",
                label_visibility="hidden",
                key=text_key,
            )

            if candidate.get("question"):
                st.warning(candidate["question"])

            has_unsaved_edits = (
                bool(edited_text.strip())
                and edited_text != candidate["text"]
            )

            kinds = ["note", "idea", "task"]
            current_kind = candidate["kind"]

            # Same stale-widget guard as the text box (see above). Without the
            # reset, the dropdown shows the old kind AND trips the change
            # detection below, firing a request that reverts the server change.
            kind_key = f"candidate_kind_{candidate_id}"
            kind_server_key = f"candidate_server_kind_{candidate_id}"

            if kind_key not in state or state.get(kind_server_key) != current_kind:
                state[kind_key] = current_kind
                state[kind_server_key] = current_kind

            # Topic selector — same stale-widget guard as kind. Options include
            # the current topic in case it isn't in the list (e.g. set via chat).
            current_topic = candidate.get("topic") or "INBOX"
            topic_options = list(
                dict.fromkeys([*TOPIC_OPTIONS, current_topic])
            )
            topic_key = f"candidate_topic_{candidate_id}"
            topic_server_key = f"candidate_server_topic_{candidate_id}"

            if topic_key not in state or state.get(topic_server_key) != current_topic:
                state[topic_key] = current_topic
                state[topic_server_key] = current_topic

            new_kind = st.selectbox(
                "Kind",
                kinds,
                disabled=has_unsaved_edits,
                key=kind_key,
            )

            new_topic = st.selectbox(
                "Topic",
                topic_options,
                disabled=has_unsaved_edits,
                key=topic_key,
            )

            # Apply a kind change only after edits are saved, so it operates
            # on the saved text, not a stale value.
            if new_kind != current_kind and not has_unsaved_edits:
                with st.spinner("Changing kind..."):
                    response = call_api(
                        api.update_candidate_kind,
                        candidate_id,
                        new_kind,
                    )

                if response is not None:
                    sync_chat_response(response)
                    st.rerun()

            if new_topic != current_topic and not has_unsaved_edits:
                with st.spinner("Changing topic..."):
                    response = call_api(
                        api.update_candidate_topic,
                        candidate_id,
                        new_topic,
                    )

                if response is not None:
                    sync_chat_response(response)
                    st.rerun()

            # Confirm and kind changes are blocked while the text box has
            # edits that have not been saved yet.
            if has_unsaved_edits:
                st.caption("Save your edit before confirming or changing kind.")

            action = None
            edited_value = None

            if st.button(
                "Save edit",
                key=f"edit_{candidate_id}",
                disabled=not has_unsaved_edits,
                use_container_width=True,
            ):
                action = "update"
                edited_value = edited_text.strip()

            if st.button(
                "✅ Confirm",
                key=f"confirm_{candidate_id}",
                disabled=(
                    bool(candidate.get("question"))
                    or has_unsaved_edits
                ),
                use_container_width=True,
            ):
                action = "confirm"

            if st.button(
                "Reject",
                key=f"reject_{candidate_id}",
                use_container_width=True,
            ):
                action = "reject"

            if action:
                with st.spinner("Saving your choice..."):
                    response = call_api(
                        api.resolve_candidate,
                        candidate_id,
                        action,
                        edited_value,
                    )

                if response is not None:
                    sync_chat_response(response)
                    st.rerun()
