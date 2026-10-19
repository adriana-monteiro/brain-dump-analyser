
import streamlit as st

from frontend.shared import render_collection

render_collection(
    kind="note",
    title="📝 Notes",
    description="Useful information you want to keep and find later.",
)
