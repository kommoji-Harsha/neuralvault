"""Streamlit benchmark table and playground dashboard with highlighted citations."""

import json
from pathlib import Path

from neuralvault.contract import SearchRequest
from neuralvault.service.service import RagService


def load_benchmark_results(json_path: Path | str = "results/results.json") -> dict:
    """Load benchmark results JSON if available."""
    p = Path(json_path)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def run_dashboard() -> None:
    """Run Streamlit dashboard UI."""
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit is not installed. Install via 'pip install streamlit'")
        return

    st.set_page_config(page_title="NeuralVault Dashboard", page_icon="⚡", layout="wide")
    st.title("⚡ NeuralVault Benchmark & Search Playground")

    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Benchmark Results", "Search Playground"])

    if page == "Benchmark Results":
        st.header("Offline Benchmark Ablation Results")
        res_data = load_benchmark_results()

        if not res_data:
            st.warning("No benchmark results found at 'results/results.json'.")
        else:
            col_name = res_data.get("collection", "N/A")
            t_stamp = res_data.get("timestamp", "N/A")
            st.subheader(f"Collection: {col_name} (Ran at: {t_stamp})")
            strategies = res_data.get("strategies", [])

            if strategies:
                st.table([
                    {
                        "Strategy": s["strategy"],
                        "Recall@5": f"{s['recall_at_5']:.4f}",
                        "MRR": f"{s['mrr']:.4f}",
                        "nDCG@5": f"{s['ndcg_at_5']:.4f}",
                        "p50 Latency (ms)": f"{s['latency_p50_ms']:.2f}",
                        "p95 Latency (ms)": f"{s['latency_p95_ms']:.2f}",
                    }
                    for s in strategies
                ])

    elif page == "Search Playground":
        st.header("Interactive Search Playground")
        service = RagService()
        cols = service.list_collections()

        if not cols:
            st.error("No collections found. Run 'neuralvault ingest <collection>' first.")
            return

        col_names = [c.name for c in cols]
        selected_col = st.selectbox("Select Collection", col_names)
        profile = st.selectbox("Profile", ["balanced", "fast", "best"])
        top_k = st.slider("Top K", min_value=1, max_value=10, value=3)

        query = st.text_input("Enter Search Query", "NeuralVault hybrid retrieval")

        if st.button("Search") and query.strip():
            req = SearchRequest(
                query=query,
                collection=selected_col,
                top_k=top_k,
                profile=profile,
            )
            resp = service.search(req)

            st.subheader("Results")
            st.caption(f"Latency: {resp.latency_ms:.2f} ms")

            if resp.note:
                st.info(resp.note)

            for idx, chunk in enumerate(resp.results, 1):
                exp_label = f"[{idx}] {chunk.location} (Score: {chunk.score:.4f})"
                with st.expander(exp_label, expanded=True):
                    st.markdown(f"**Source:** `{chunk.source}`")
                    st.code(chunk.text, language="markdown")


if __name__ == "__main__":
    run_dashboard()
