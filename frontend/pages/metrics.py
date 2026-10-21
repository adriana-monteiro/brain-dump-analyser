
import streamlit as st

from frontend.shared import call_api, get_api

api = get_api()

st.title("📊 Metrics")
st.caption("Per-call LLM usage, cost, and latency across the pipeline.")

# Distinct filter options come from the unfiltered records so selecting a
# filter never hides values the user might want to switch to next.
with st.spinner("Loading metrics..."):
    all_records = call_api(api.get_metrics)

if all_records is None:
    st.stop()

if not all_records:
    st.info("No metrics available yet. Run a brain dump to record LLM calls.")
    st.stop()

steps = sorted({r["step"] for r in all_records})
models = sorted({r["model"] for r in all_records})

step_col, model_col = st.columns(2)

with step_col:
    selected_step = st.selectbox("Step", ["All"] + steps, key="metrics_step")

with model_col:
    selected_model = st.selectbox(
        "Model", ["All"] + models, key="metrics_model"
    )

step_filter = None if selected_step == "All" else selected_step
model_filter = None if selected_model == "All" else selected_model

with st.spinner("Loading metrics..."):
    summary = call_api(
        api.get_metrics_summary, step=step_filter, model=model_filter
    )
    records = call_api(
        api.get_metrics, step=step_filter, model=model_filter
    )

# A None here means call_api already surfaced the APIError via st.error; stop
# before drawing tables so a partial/missing response never renders.
if summary is None or records is None:
    st.stop()

cost_col, in_col, out_col = st.columns(3)
cost_col.metric("Total cost (USD)", f"${summary['total_cost_usd']:.4f}")
in_col.metric("Input tokens", f"{summary['total_input_tokens']:,}")
out_col.metric("Output tokens", f"{summary['total_output_tokens']:,}")

latency_col, calls_col, errors_col = st.columns(3)
latency_col.metric("Avg latency (ms)", f"{summary['avg_latency_ms']:.0f}")
calls_col.metric("Calls", summary["call_count"])
errors_col.metric("Errors", summary["error_count"])

if not records:
    st.info("No metrics match these filters.")
    st.stop()

st.subheader("Calls")
st.dataframe(
    [
        {
            "step": r["step"],
            "model": r["model"],
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "latency_ms": r["latency_ms"],
            "cost_usd": r["cost_usd"],
            "status": r["status"],
            "error": r["error"],
            "created_at": r["created_at"],
        }
        for r in records
    ],
    use_container_width=True,
    hide_index=True,
)

step_breakdown_col, model_breakdown_col = st.columns(2)

with step_breakdown_col:
    st.subheader("By step")
    st.dataframe(
        [
            {
                "step": b["key"],
                "calls": b["call_count"],
                "cost_usd": b["cost_usd"],
                "tokens": b["tokens"],
            }
            for b in summary["by_step"]
        ],
        use_container_width=True,
        hide_index=True,
    )

with model_breakdown_col:
    st.subheader("By model")
    st.dataframe(
        [
            {
                "model": b["key"],
                "calls": b["call_count"],
                "cost_usd": b["cost_usd"],
                "tokens": b["tokens"],
            }
            for b in summary["by_model"]
        ],
        use_container_width=True,
        hide_index=True,
    )
