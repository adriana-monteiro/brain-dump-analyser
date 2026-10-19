
import streamlit as st

from frontend.shared import call_api, get_api

st.set_page_config(
    page_title="Brain dump",
    page_icon="🧠",
    layout="wide",
)

api = get_api()

pages = [
    st.Page(
        "pages/brain_dump.py",
        title="Brain dump",
        icon="🧠",
        default=True,
    ),
    st.Page("pages/tasks.py", title="Tasks", icon="✅"),
    st.Page("pages/notes.py", title="Notes", icon="📝"),
    st.Page("pages/ideas.py", title="Ideas", icon="💡"),
]

navigation = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.divider()

    with st.expander("⚙️ Topic settings"):
        topics = call_api(api.get_topics)

        if topics is not None:
            topic_text = st.text_area(
                "One topic per line",
                value="\n".join(topics),
                help="INBOX is the fallback topic.",
                key="topic_editor",
            )

            if st.button("Save topics", use_container_width=True):
                new_topics = [
                    topic.strip()
                    for topic in topic_text.splitlines()
                    if topic.strip()
                ]

                saved = call_api(api.set_topics, new_topics)

                if saved is not None:
                    st.toast("Topics saved.")
                    st.rerun()

navigation.run()
