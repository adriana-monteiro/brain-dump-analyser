
import streamlit as st

from frontend.shared import call_api, get_api, show_item

api = get_api()

st.title("✅ Tasks")
st.caption("Keep track of what needs doing and what you've finished.")

statuses = {
    "Open": "open",
    "Completed": "done",
    "Cancelled": "cancelled",
}

selected_label = st.radio(
    "Show tasks",
    list(statuses.keys()),
    horizontal=True,
    key="tasks_status_filter",
)

with st.spinner("Loading tasks..."):
    tasks = call_api(
        api.get_items,
        kind="task",
        status=statuses[selected_label],
    )

if tasks is not None:
    if not tasks:
        st.info(f"No {selected_label.lower()} tasks.")
    else:
        st.caption(f"{len(tasks)} task(s)")

        for task in tasks:
            task_id = task["id"]

            with st.container(border=True):
                show_item(task, show_status=True)

                if task["status"] == "open":
                    done_col, cancel_col = st.columns(2)

                    with done_col:
                        mark_done = st.button(
                            "✓ Mark done",
                            key=f"task_done_{task_id}",
                            use_container_width=True,
                        )

                    with cancel_col:
                        cancel = st.button(
                            "Cancel task",
                            key=f"task_cancel_{task_id}",
                            use_container_width=True,
                        )

                    if mark_done or cancel:
                        new_status = "done" if mark_done else "cancelled"

                        with st.spinner("Updating task..."):
                            result = call_api(
                                api.update_task_status,
                                task_id,
                                new_status,
                            )

                        if result is not None:
                            st.toast("Task updated.")
                            st.rerun()

                else:
                    if st.button(
                        "Reopen task",
                        key=f"task_reopen_{task_id}",
                    ):
                        with st.spinner("Reopening task..."):
                            result = call_api(
                                api.update_task_status,
                                task_id,
                                "open",
                            )

                        if result is not None:
                            st.toast("Task reopened.")
                            st.rerun()
