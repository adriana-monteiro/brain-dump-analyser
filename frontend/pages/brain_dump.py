
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
                        or key in {"msgs", "cands"}
                    ):
                        del state[key]

                state["msgs"] = []
                state["cands"] = []
                state["conv"] = response["conversation_id"]
                state["confirm_reset"] = False

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

for role, content in state.msgs:
    with st.chat_message(role):
        st.markdown(content)

if state.cands:
    st.divider()
    st.subheader("Review proposed items")
    st.caption(
        "Confirm to save, edit to change the text, "
        "or reject to discard."
    )

    for candidate in state.cands:
        candidate_id = candidate["id"]
        existing_text = candidate.get("existing_item_text")
        operation = candidate.get("op", "new")

        with st.container(border=True):
            st.caption(
                f"#{candidate['display_no']} · "
                f"{candidate['kind'].capitalize()} · "
                f"{candidate.get('topic') or 'INBOX'} · "
                f"{operation.replace('_', ' ').capitalize()}"
            )

            if existing_text:
                st.caption(f"Matching saved item: {existing_text}")

            edited_text = st.text_input(
                "Item text",
                label_visibility="hidden",
                value=candidate["text"],
                key=f"candidate_text_{candidate_id}",
            )

            if candidate.get("question"):
                st.warning(candidate["question"])

            confirm_col, reject_col, edit_col = st.columns(3)
            action = None
            edited_value = None

            with confirm_col:
                if st.button(
                    "✅ Confirm",
                    key=f"confirm_{candidate_id}",
                    disabled=bool(candidate.get("question")),
                    use_container_width=True,
                ):
                    action = "confirm"

            with reject_col:
                if st.button(
                    "Reject",
                    key=f"reject_{candidate_id}",
                    use_container_width=True,
                ):
                    action = "reject"

            with edit_col:
                if st.button(
                    "Save edit",
                    key=f"edit_{candidate_id}",
                    disabled=(
                        not edited_text.strip()
                        or edited_text == candidate["text"]
                    ),
                    use_container_width=True,
                ):
                    action = "update"
                    edited_value = edited_text.strip()

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
