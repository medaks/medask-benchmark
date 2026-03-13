"""
Streamlit GUI for the MedAsk Triage Benchmarking Research Stack.

Run with: streamlit run research/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.db.schema import init_db
from research.db.store import ResultStore
from research.prompts.manager import PromptManager
from research.charts import (
    build_overall_accuracy_fig,
    build_per_category_fig,
    build_settings_comparison_fig,
    build_safety_analysis_fig,
)
from research.lmstudio_api import fetch_models, fetch_models_detailed

st.set_page_config(
    page_title="MedAsk Triage Benchmark",
    page_icon="🏥",
    layout="wide",
)

# ── Sidebar Navigation ─────────────────────────────────────────────

page = st.sidebar.radio(
    "Navigation",
    ["Run Benchmark", "Results", "Charts", "Compare", "Import", "Prompts"],
)

st.sidebar.markdown("---")
st.sidebar.caption("MedAsk Triage Benchmarking Research Stack")


# ── Helpers ─────────────────────────────────────────────────────────


def get_store():
    conn = init_db()
    return ResultStore(conn), conn


def get_model_names(store, vignette_set="semigran"):
    runs = store.get_runs(vignette_set=vignette_set)
    return sorted(set(r["model_name"] for r in runs))


# ── Page: Run Benchmark ────────────────────────────────────────────

if page == "Run Benchmark":
    st.title("Run Benchmark")
    st.markdown("Configure and run a triage benchmark against an LMStudio model.")

    # ── LMStudio connection ────────────────────────────────────────
    base_url = st.text_input("LMStudio API URL", "http://localhost:1234/v1")

    if st.button("Refresh models", key="refresh_models"):
        st.session_state.pop("lmstudio_models_detailed", None)

    if "lmstudio_models_detailed" not in st.session_state:
        st.session_state.lmstudio_models_detailed = fetch_models_detailed(base_url)

    all_model_info = st.session_state.lmstudio_models_detailed
    model_ids = [m["id"] for m in all_model_info] if all_model_info else []
    model_info_map = {m["id"]: m for m in all_model_info}

    if not model_ids:
        st.warning("Could not connect to LMStudio or no models loaded. "
                   "Check that LMStudio is running and at least one model is loaded.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Configuration")
        if model_ids:
            # Build display labels: "model_id  (state, quant, arch)"
            model_labels = []
            for mid in model_ids:
                info = model_info_map[mid]
                state = info.get("state", "?")
                badge = "🟢" if state == "loaded" else "⚪"
                model_labels.append(f"{badge} {mid}")
            sel_idx = st.selectbox(
                "Model",
                range(len(model_ids)),
                format_func=lambda i: model_labels[i],
                key="bench_model",
            )
            model = model_ids[sel_idx]
            sel_info = model_info_map[model]
        else:
            model = st.text_input("Model name (LMStudio not reachable)", placeholder="qwen2.5-7b-instruct")
            sel_info = {}

        # Auto-populate from LMStudio metadata (editable)
        default_family = sel_info.get("publisher", "")
        default_quant = sel_info.get("quantization", "")
        default_arch = sel_info.get("arch", "")
        default_compat = sel_info.get("compatibility_type", "")
        default_ctx = sel_info.get("loaded_context_length") or sel_info.get("max_context_length") or 0
        default_max_ctx = sel_info.get("max_context_length", 0)

        model_family = st.text_input("Model family / publisher", value=default_family, placeholder="qwen")
        quant = st.text_input("Quantization", value=default_quant, placeholder="Q4_K_M")
        params = st.text_input("Parameter count (optional)", placeholder="7B")

        # Show auto-detected info
        if sel_info:
            info_parts = []
            if default_arch:
                info_parts.append(f"**Arch:** {default_arch}")
            if default_compat:
                info_parts.append(f"**Format:** {default_compat}")
            state = sel_info.get("state", "?")
            info_parts.append(f"**State:** {state}")
            caps = sel_info.get("capabilities", [])
            if caps:
                info_parts.append(f"**Capabilities:** {', '.join(caps)}")
            if default_max_ctx:
                info_parts.append(f"**Max context:** {default_max_ctx:,}")
            if sel_info.get("loaded_context_length"):
                info_parts.append(f"**Loaded context:** {sel_info['loaded_context_length']:,}")
            st.caption(" · ".join(info_parts))

    with col2:
        st.subheader("Inference Settings")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.0, 0.1)
        max_tokens = st.number_input("Max tokens", 50, 16384, 4096)
        top_p = st.number_input("Top-p (0 = not set)", 0.0, 1.0, 0.0, 0.05)
        top_k = st.number_input("Top-k (0 = not set)", 0, 200, 0)
        context_length = st.number_input(
            "Context length (0 = default)",
            0, 131072,
            value=default_ctx if default_ctx else 0,
        )
        gpu_layers = st.number_input("GPU layers (-1 = all/max offload)", -1, 200, -1)

    st.subheader("Grader (LLM-as-Judge)")
    st.markdown("Optionally use a second model to evaluate responses instead of regex matching.")
    use_grader = st.checkbox("Enable LLM grader")
    grader_model = None
    grader_base_url = None
    if use_grader:
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            grader_base_url = st.text_input(
                "Grader LMStudio URL (leave same to use same server)",
                value=base_url,
                key="grader_url",
            )
        with gcol2:
            if st.button("Refresh grader models", key="refresh_grader"):
                st.session_state.pop("grader_models", None)

        if "grader_models_detailed" not in st.session_state or st.session_state.get("_grader_url_prev") != grader_base_url:
            st.session_state.grader_models_detailed = fetch_models_detailed(grader_base_url)
            st.session_state._grader_url_prev = grader_base_url

        grader_all_info = st.session_state.grader_models_detailed
        grader_ids = [m["id"] for m in grader_all_info] if grader_all_info else []
        grader_info_map = {m["id"]: m for m in grader_all_info}

        if grader_ids:
            grader_labels = []
            for gid in grader_ids:
                gi = grader_info_map[gid]
                badge = "🟢" if gi.get("state") == "loaded" else "⚪"
                grader_labels.append(f"{badge} {gid}")
            g_idx = st.selectbox(
                "Grader model",
                range(len(grader_ids)),
                format_func=lambda i: grader_labels[i],
                key="grader_model_sel",
            )
            grader_model = grader_ids[g_idx]
        else:
            grader_model = st.text_input("Grader model name", placeholder="llama-3.1-8b-instruct")

    st.subheader("Benchmark Settings")
    col3, col4 = st.columns(2)
    with col3:
        pm = PromptManager()
        prompts = pm.list_prompts()
        prompt_options = [p["version_tag"] for p in prompts] if prompts else ["v1_original"]
        prompt_version = st.selectbox("Prompt version", prompt_options)
        vignette_set = st.selectbox("Vignette set", ["semigran", "kopka"])
    with col4:
        runs = st.number_input("Number of runs", 1, 20, 1)

    notes = st.text_area("Notes (optional)", placeholder="First test of this model...")

    if st.button("Run Benchmark", type="primary", disabled=not model):
        if not model or not model.strip():
            st.error("Model name is required.")
        else:
            st.markdown("---")
            st.subheader("Progress")

            # Progress display elements
            run_label = st.empty()
            run_bar = st.progress(0)
            case_label = st.empty()
            case_bar = st.progress(0)
            ticker = st.empty()

            correct_so_far = [0]
            total_so_far = [0]

            def on_progress(run_num, total_runs, case_idx, total_cases,
                            pred, gold, correct, err_count):
                # Update run-level progress
                run_frac = (run_num - 1) / total_runs + (case_idx / total_cases) / total_runs
                run_label.markdown(f"**Run {run_num} of {total_runs}**")
                run_bar.progress(min(run_frac, 1.0))

                # Update case-level progress
                case_frac = case_idx / total_cases
                case_label.markdown(f"**Case {case_idx} of {total_cases}**")
                case_bar.progress(min(case_frac, 1.0))

                # Running accuracy ticker
                if correct:
                    correct_so_far[0] += 1
                total_so_far[0] += 1
                acc = correct_so_far[0] / total_so_far[0]
                mark = "✅" if correct else "❌"
                err_text = f" · ⚠️ {err_count} error(s)" if err_count else ""
                ticker.markdown(
                    f"{mark} Case {case_idx}: predicted **{pred}**, "
                    f"actual **{gold}** — "
                    f"Running accuracy: **{acc:.0%}** "
                    f"({correct_so_far[0]}/{total_so_far[0]}){err_text}"
                )

            try:
                from research.runner import run_benchmark

                batch_id = run_benchmark(
                    model=model.strip(),
                    model_family=model_family.strip() or None,
                    quantization=quant.strip() or None,
                    parameter_count=params.strip() or None,
                    temperature=temperature,
                    top_p=top_p if top_p > 0 else None,
                    top_k=top_k if top_k > 0 else None,
                    max_tokens=max_tokens,
                    context_length=context_length if context_length > 0 else None,
                    gpu_layers=gpu_layers if gpu_layers != 0 else None,
                    prompt_version=prompt_version,
                    vignette_set=vignette_set,
                    runs=runs,
                    notes=notes.strip() or None,
                    base_url=base_url,
                    grader_model=grader_model.strip() if grader_model else None,
                    grader_base_url=grader_base_url if grader_model else None,
                    progress_callback=on_progress,
                )
                run_bar.progress(1.0)
                case_bar.progress(1.0)
                run_label.markdown(f"**Run {runs} of {runs}** — Complete")
                acc_final = correct_so_far[0] / total_so_far[0] if total_so_far[0] else 0
                case_label.markdown(
                    f"**Done!** Final accuracy: **{acc_final:.1%}** "
                    f"({correct_so_far[0]}/{total_so_far[0]})"
                )
                ticker.empty()
                st.success(f"Batch **{batch_id}** finished. {runs} run(s) stored.")
            except Exception as e:
                st.error(f"Benchmark failed: {e}")


# ── Page: Results ──────────────────────────────────────────────────

elif page == "Results":
    st.title("Results")

    store, conn = get_store()

    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "All Runs", "Baselines", "Run Detail"])

    with tab1:
        vs = st.selectbox("Vignette set", ["semigran", "kopka"], key="summary_vs")
        summaries = store.get_model_summary(vs)
        if summaries:
            df = pd.DataFrame(summaries)
            df["mean_overall"] = (df["mean_overall"] * 100).round(1)
            df["sd_overall"] = (df["sd_overall"] * 100).round(1)
            df["mean_em"] = (df["mean_em"] * 100).round(1)
            df["mean_ne"] = (df["mean_ne"] * 100).round(1)
            df["mean_sc"] = (df["mean_sc"] * 100).round(1)
            df["mean_safety"] = (df["mean_safety"] * 100).round(1)
            df["mean_run_secs"] = df["mean_run_secs"].round(1)
            df["mean_per_vignette_secs"] = df["mean_per_vignette_secs"].round(2)
            display_cols = ["model_name", "quantization", "num_runs", "mean_overall",
                            "sd_overall", "mean_em", "mean_ne", "mean_sc", "mean_safety",
                            "temperature", "mean_run_secs", "mean_per_vignette_secs"]
            st.dataframe(
                df[display_cols].rename(columns={
                    "model_name": "Model",
                    "quantization": "Quant",
                    "num_runs": "Runs",
                    "mean_overall": "Overall %",
                    "sd_overall": "SD %",
                    "mean_em": "EM %",
                    "mean_ne": "NE %",
                    "mean_sc": "SC %",
                    "mean_safety": "Safety %",
                    "temperature": "Temp",
                    "mean_run_secs": "Avg Run (s)",
                    "mean_per_vignette_secs": "Avg/Case (s)",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No data. Import results first.")

    with tab2:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            model_filter = st.text_input("Filter by model", key="runs_model")
        with col_f2:
            vs_filter = st.selectbox("Vignette set", ["", "semigran", "kopka"], key="runs_vs")
        with col_f3:
            source_filter = st.selectbox("Source", ["", "local", "imported_jsonl"], key="runs_source")

        all_runs = store.get_runs(
            model_name=model_filter or None,
            vignette_set=vs_filter or None,
            source=source_filter or None,
        )
        if all_runs:
            df = pd.DataFrame(all_runs)
            show_cols = ["run_id", "model_name", "quantization", "vignette_set",
                         "run_number", "overall_accuracy", "safety_rate", "temperature",
                         "total_seconds", "source", "created_at"]
            available = [c for c in show_cols if c in df.columns]
            display = df[available].copy()
            if "overall_accuracy" in display.columns:
                display["overall_accuracy"] = (display["overall_accuracy"] * 100).round(1)
            if "safety_rate" in display.columns:
                display["safety_rate"] = (display["safety_rate"] * 100).round(1)
            if "total_seconds" in display.columns:
                display["total_seconds"] = display["total_seconds"].round(1)
            st.dataframe(display, use_container_width=True, hide_index=True)
        else:
            st.info("No runs found.")

    with tab3:
        vs_bl = st.selectbox("Vignette set", ["semigran", "kopka"], key="bl_vs")
        baselines = store.get_baselines(vs_bl)
        if baselines:
            df = pd.DataFrame(baselines)
            for col in ["overall_accuracy", "em_accuracy", "ne_accuracy", "sc_accuracy"]:
                if col in df.columns:
                    df[col] = (df[col] * 100).round(1)
            st.dataframe(
                df[["model_name", "overall_accuracy", "em_accuracy", "ne_accuracy", "sc_accuracy"]].rename(columns={
                    "model_name": "Model",
                    "overall_accuracy": "Overall %",
                    "em_accuracy": "EM %",
                    "ne_accuracy": "NE %",
                    "sc_accuracy": "SC %",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No baselines. Use Import page to load them.")

    with tab4:
        run_id_input = st.number_input("Enter Run ID", min_value=1, step=1, key="detail_run_id")
        if st.button("Load Run Details"):
            results = store.get_run_results(run_id_input)
            if results:
                df = pd.DataFrame(results)
                correct_count = df["correct"].sum()
                total = len(df)
                st.metric("Accuracy", f"{correct_count}/{total} ({correct_count/total:.1%})")

                cat_labels = {"em": "Emergency", "ne": "Non-Emergency", "sc": "Self-Care"}
                df["category"] = df["true_urgency"].map(cat_labels)
                df["result"] = df["correct"].map({1: "Correct", 0: "Incorrect"})
                st.dataframe(
                    df[["case_id", "category", "true_urgency", "llm_output", "result", "latency_ms"]].rename(columns={
                        "case_id": "Case",
                        "category": "Category",
                        "true_urgency": "True",
                        "llm_output": "Predicted",
                        "result": "Result",
                        "latency_ms": "Latency (ms)",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.warning(f"No results found for run_id={run_id_input}")

    conn.close()


# ── Page: Charts ───────────────────────────────────────────────────

elif page == "Charts":
    st.title("Charts")

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        chart_vs = st.selectbox("Vignette set", ["semigran", "kopka"], key="chart_vs")
    with col_ctrl2:
        include_bl = st.checkbox("Include baselines", True)

    tab1, tab2, tab3, tab4 = st.tabs(["Overall Accuracy", "Per-Category", "Settings Comparison", "Safety"])

    with tab1:
        fig = build_overall_accuracy_fig(chart_vs, include_bl)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = build_per_category_fig(chart_vs, include_bl)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        store, conn = get_store()
        models = get_model_names(store, chart_vs)
        conn.close()

        if models:
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                selected_model = st.selectbox("Model", models, key="settings_model")
            with col_s2:
                vary_by = st.selectbox("Vary by", ["temperature", "quantization", "prompt_id", "context_length"])

            fig = build_settings_comparison_fig(selected_model, vary_by, chart_vs)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No runs found for this model.")
        else:
            st.info("No benchmark runs available yet.")

    with tab4:
        fig = build_safety_analysis_fig(chart_vs)
        st.plotly_chart(fig, use_container_width=True)


# ── Page: Compare ──────────────────────────────────────────────────

elif page == "Compare":
    st.title("McNemar Paired Comparison")
    st.markdown("Compare two models using McNemar's statistical test on the same vignettes.")

    store, conn = get_store()
    compare_vs = st.selectbox("Vignette set", ["semigran", "kopka"], key="compare_vs")
    models = get_model_names(store, compare_vs)
    conn.close()

    if len(models) < 2:
        st.info("Need at least 2 models with benchmark runs to compare.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            model_a = st.selectbox("Model A", models, key="compare_a")
        with col2:
            model_b = st.selectbox("Model B", [m for m in models if m != model_a] or models, key="compare_b")

        if st.button("Run Comparison", type="primary"):
            from research.analyze import paired_comparison

            result = paired_comparison(
                model_a=model_a,
                model_b=model_b,
                vignette_set=compare_vs,
            )

            if result is None:
                st.error("Could not run comparison. Check that both models have runs.")
            else:
                st.markdown("### Results")

                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric(f"Model A: {model_a}", f"{result['accuracy_a']:.1%}")
                with col_m2:
                    st.metric(f"Model B: {model_b}", f"{result['accuracy_b']:.1%}")
                with col_m3:
                    st.metric("Accuracy Diff (B-A)", f"{result['accuracy_diff']:+.1%}")

                st.markdown("### McNemar Test")
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("p-value", f"{result['p_value']:.4g}")
                with col_s2:
                    or_display = f"{result['odds_ratio']:.2f}" if result['odds_ratio'] != float('inf') else "inf"
                    st.metric("Odds Ratio (A vs B)", or_display)
                with col_s3:
                    sig = "Yes" if result['p_value'] < 0.05 else "No"
                    st.metric("Significant (p<0.05)?", sig)

                st.markdown(
                    f"**Discordant pairs:** A-right/B-wrong = {result['a_right_b_wrong']}, "
                    f"A-wrong/B-right = {result['a_wrong_b_right']} "
                    f"(exact={result['exact']})"
                )


# ── Page: Import ───────────────────────────────────────────────────

elif page == "Import":
    st.title("Import Data")

    st.subheader("Import Published Baselines")
    st.markdown("Import Table 1 data from the MedAsk blog (6 models, Semigran dataset).")
    if st.button("Import Table 1 Baselines"):
        from research.importer import import_table1_baselines
        import_table1_baselines()
        st.success("Baselines imported.")
        st.rerun()

    st.markdown("---")

    st.subheader("Import Existing JSONL Results")
    st.markdown("Bulk-import all JSONL files from `triage_bench/results/`.")
    if st.button("Import All Existing JSONL"):
        from research.importer import import_all_existing
        results_root = Path(__file__).parent.parent / "triage_bench" / "results"
        with st.status("Importing...", expanded=True) as status:
            import_all_existing(results_root)
            status.update(label="Import complete", state="complete")
        st.success("Import finished.")

    st.markdown("---")

    st.subheader("Import Single JSONL File")
    uploaded = st.file_uploader("Upload a JSONL results file", type=["jsonl"])
    if uploaded:
        import tempfile
        from research.importer import import_jsonl_file

        col1, col2 = st.columns(2)
        with col1:
            override_model = st.text_input("Override model name (optional)", key="imp_model")
        with col2:
            override_vs = st.selectbox("Override vignette set", ["", "semigran", "kopka"], key="imp_vs")

        if st.button("Import Uploaded File"):
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = Path(tmp.name)

            try:
                run_ids = import_jsonl_file(
                    tmp_path,
                    model_name=override_model or None,
                    vignette_set=override_vs or None,
                )
                st.success(f"Imported {len(run_ids)} run(s). IDs: {run_ids}")
            except Exception as e:
                st.error(f"Import failed: {e}")


# ── Page: Prompts ──────────────────────────────────────────────────

elif page == "Prompts":
    st.title("Prompt Management")

    pm = PromptManager()
    prompts = pm.list_prompts()

    st.subheader("Registered Prompts")
    if prompts:
        st.dataframe(
            pd.DataFrame(prompts).rename(columns={
                "prompt_id": "ID",
                "version_tag": "Version",
                "description": "Description",
                "created_at": "Created",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No prompts registered yet.")

    st.markdown("---")

    st.subheader("Available Prompt Files")
    library_dir = Path(__file__).parent / "prompts" / "library"
    txt_files = sorted(library_dir.glob("*.txt"))
    if txt_files:
        selected_file = st.selectbox("View prompt file", [f.stem for f in txt_files])
        content = (library_dir / f"{selected_file}.txt").read_text()
        st.code(content, language="text")
    else:
        st.info("No prompt files found in prompts/library/.")

    st.markdown("---")

    st.subheader("Create New Prompt")
    new_tag = st.text_input("Version tag", placeholder="v2_cot")
    new_content = st.text_area(
        "Prompt template (must include {vignette} placeholder)",
        height=200,
        placeholder="Your prompt here...\n\n{vignette}",
    )

    if st.button("Save Prompt"):
        if not new_tag.strip():
            st.error("Version tag is required.")
        elif "{vignette}" not in new_content:
            st.error("Prompt must contain {vignette} placeholder.")
        else:
            fp = library_dir / f"{new_tag.strip()}.txt"
            fp.write_text(new_content)
            prompt_id, _ = pm.get_or_create(new_tag.strip())
            st.success(f"Prompt saved as {fp.name} (ID: {prompt_id})")
            st.rerun()
