"""
Unified CLI for the triage benchmarking research stack.

Usage:
    python -m research.cli <command> [options]

Commands:
    run        Run a benchmark against an LMStudio model
    import     Import existing JSONL results or Table 1 baselines
    list       List runs, models, or prompts in the database
    chart      Generate Plotly charts
    compare    McNemar paired comparison
"""
import argparse
import sys


def cmd_run(args):
    from research.runner import run_benchmark
    run_benchmark(
        model=args.model,
        model_family=args.model_family,
        quantization=args.quant,
        parameter_count=args.params,
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        context_length=args.context_length,
        gpu_layers=args.gpu_layers,
        prompt_version=args.prompt,
        vignette_set=args.vignette_set,
        runs=args.runs,
        notes=args.notes,
        base_url=args.base_url,
        grader_model=args.grader_model,
        grader_base_url=args.grader_base_url,
    )


def cmd_import(args):
    from pathlib import Path
    from research.importer import import_jsonl_file, import_all_existing, import_table1_baselines

    if args.baselines:
        import_table1_baselines()
    elif args.all:
        results_root = Path(__file__).parent.parent / "triage_bench" / "results"
        import_all_existing(results_root)
    elif args.file:
        run_ids = import_jsonl_file(
            Path(args.file),
            model_name=args.model,
            vignette_set=args.vignette_set,
        )
        print(f"Imported -> run_ids={run_ids}")
    else:
        print("Specify --baselines, --all, or --file")


def cmd_list(args):
    from research.db.schema import init_db
    from research.db.store import ResultStore

    conn = init_db()
    store = ResultStore(conn)

    if args.what == "runs":
        runs = store.get_runs(
            model_name=args.model,
            vignette_set=args.vignette_set,
            source=args.source,
        )
        if not runs:
            print("No runs found.")
            conn.close()
            return

        print(f"{'ID':>4}  {'Model':<30}  {'Quant':<10}  {'VS':<10}  {'Run#':>4}  {'Acc':>6}  {'Source':<15}  {'Date'}")
        print("-" * 110)
        for r in runs:
            acc = f"{r['overall_accuracy']*100:.1f}%" if r['overall_accuracy'] else "N/A"
            print(
                f"{r['run_id']:>4}  {r['model_name']:<30}  "
                f"{(r['quantization'] or '-'):<10}  {r['vignette_set']:<10}  "
                f"{r['run_number']:>4}  {acc:>6}  {r['source']:<15}  "
                f"{r['created_at']}"
            )

    elif args.what == "summary":
        summaries = store.get_model_summary(args.vignette_set or "semigran")
        if not summaries:
            print("No data.")
            conn.close()
            return
        print(f"{'Model':<30}  {'Quant':<10}  {'Runs':>4}  {'Overall':>10}  {'em':>8}  {'ne':>8}  {'sc':>8}")
        print("-" * 90)
        for s in summaries:
            sd = f" +/-{s['sd_overall']*100:.1f}" if s['sd_overall'] else ""
            print(
                f"{s['model_name']:<30}  {(s['quantization'] or '-'):<10}  "
                f"{s['num_runs']:>4}  {s['mean_overall']*100:.1f}%{sd:>8}  "
                f"{s['mean_em']*100:.1f}%  {s['mean_ne']*100:.1f}%  {s['mean_sc']*100:.1f}%"
            )

    elif args.what == "prompts":
        from research.prompts.manager import PromptManager
        pm = PromptManager()
        for p in pm.list_prompts():
            print(f"  [{p['prompt_id']}] {p['version_tag']}  ({p['created_at']})")

    elif args.what == "baselines":
        baselines = store.get_baselines(args.vignette_set or "semigran")
        if not baselines:
            print("No baselines. Run: python -m research.cli import --baselines")
            conn.close()
            return
        for b in baselines:
            print(f"  {b['model_name']:<20}  {b['overall_accuracy']*100:.1f}%  "
                  f"em={b['em_accuracy']*100:.1f}%  ne={b['ne_accuracy']*100:.1f}%  "
                  f"sc={b['sc_accuracy']*100:.1f}%")

    conn.close()


def cmd_chart(args):
    from research import charts

    if args.type == "overall":
        charts.chart_overall_accuracy(
            vignette_set=args.vignette_set,
            include_baselines=not args.no_baselines,
        )
    elif args.type == "categories":
        charts.chart_per_category_breakdown(
            vignette_set=args.vignette_set,
            include_baselines=not args.no_baselines,
        )
    elif args.type == "settings":
        if not args.model:
            print("--model required for settings chart")
            sys.exit(1)
        charts.chart_settings_comparison(
            model_name=args.model,
            vary_by=args.vary_by or "temperature",
            vignette_set=args.vignette_set,
        )
    elif args.type == "safety":
        charts.chart_safety_analysis(vignette_set=args.vignette_set)


def cmd_models(args):
    from research.lmstudio_api import fetch_models
    models = fetch_models(args.base_url)
    if not models:
        print(f"No models found at {args.base_url} (is LMStudio running?)")
        return
    print(f"Available models at {args.base_url}:")
    for m in models:
        print(f"  {m}")


def cmd_compare(args):
    from research.analyze import paired_comparison
    result = paired_comparison(
        model_a=args.model_a,
        model_b=args.model_b,
        vignette_set=args.vignette_set,
        run_id_a=args.run_id_a,
        run_id_b=args.run_id_b,
    )
    if result is None:
        print("Could not run comparison. Check model names or run IDs.")
        return
    print(f"\n=== McNemar Paired Test ===")
    print(f"Run A (id={result['run_id_a']}): accuracy = {result['accuracy_a']:.2%}")
    print(f"Run B (id={result['run_id_b']}): accuracy = {result['accuracy_b']:.2%}")
    print(f"Discordant pairs: A-right/B-wrong = {result['a_right_b_wrong']}, A-wrong/B-right = {result['a_wrong_b_right']}")
    print(f"p-value: {result['p_value']:.4g}  (exact={result['exact']})")
    if result['odds_ratio'] == float('inf'):
        print("Odds ratio: inf (B never correct when A is wrong)")
    else:
        print(f"Odds ratio (A vs B): {result['odds_ratio']:.2f}")
    print(f"Accuracy difference (B - A): {result['accuracy_diff']:+.2%}")


def main():
    parser = argparse.ArgumentParser(
        prog="research.cli",
        description="Triage Benchmarking Research Stack",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- run ----
    p_run = subparsers.add_parser("run", help="Run a benchmark against LMStudio")
    p_run.add_argument("--model", required=True, help="Model name as loaded in LMStudio")
    p_run.add_argument("--model-family", help="Model family (e.g., qwen2.5, llama3.1)")
    p_run.add_argument("--quant", help="Quantization level (e.g., Q4_K_M, Q8_0)")
    p_run.add_argument("--params", help="Parameter count (e.g., 7B, 14B)")
    p_run.add_argument("--temp", type=float, default=0.0, help="Temperature (default: 0.0)")
    p_run.add_argument("--top-p", type=float, help="Top-p sampling")
    p_run.add_argument("--top-k", type=int, help="Top-k sampling")
    p_run.add_argument("--max-tokens", type=int, default=4096, help="Max tokens (default: 4096)")
    p_run.add_argument("--context-length", type=int, help="Context window size")
    p_run.add_argument("--gpu-layers", type=int, help="GPU layers offloaded")
    p_run.add_argument("--prompt", default="v1_original", help="Prompt version tag")
    p_run.add_argument("--vignette-set", default="semigran", choices=["semigran", "kopka"])
    p_run.add_argument("--runs", type=int, default=1, help="Number of runs (default: 1)")
    p_run.add_argument("--notes", help="Free-form notes about this run")
    p_run.add_argument("--base-url", default="http://localhost:1234/v1", help="LMStudio API URL")
    p_run.add_argument("--grader-model", help="LLM grader model name (enables LLM-as-judge evaluation)")
    p_run.add_argument("--grader-base-url", help="LMStudio API URL for grader (defaults to --base-url)")

    # ---- import ----
    p_import = subparsers.add_parser("import", help="Import existing results")
    p_import.add_argument("--all", action="store_true", help="Import all JSONL from triage_bench/results")
    p_import.add_argument("--baselines", action="store_true", help="Import Table 1 baseline data")
    p_import.add_argument("--file", help="Import a single JSONL file")
    p_import.add_argument("--model", help="Override model name")
    p_import.add_argument("--vignette-set", help="Override vignette set")

    # ---- list ----
    p_list = subparsers.add_parser("list", help="List database contents")
    p_list.add_argument("what", choices=["runs", "summary", "prompts", "baselines"])
    p_list.add_argument("--model", help="Filter by model name")
    p_list.add_argument("--vignette-set", help="Filter by vignette set")
    p_list.add_argument("--source", help="Filter by source")

    # ---- chart ----
    p_chart = subparsers.add_parser("chart", help="Generate Plotly charts")
    p_chart.add_argument("type", choices=["overall", "categories", "settings", "safety"])
    p_chart.add_argument("--vignette-set", default="semigran")
    p_chart.add_argument("--model", help="Model name (required for settings chart)")
    p_chart.add_argument("--vary-by", choices=["temperature", "quantization", "prompt_id", "context_length"])
    p_chart.add_argument("--no-baselines", action="store_true", help="Exclude Table 1 baselines")

    # ---- compare ----
    p_compare = subparsers.add_parser("compare", help="McNemar paired comparison")
    p_compare.add_argument("--model-a", help="First model name")
    p_compare.add_argument("--model-b", help="Second model name")
    p_compare.add_argument("--run-id-a", type=int, help="Specific run_id for model A")
    p_compare.add_argument("--run-id-b", type=int, help="Specific run_id for model B")
    p_compare.add_argument("--vignette-set", default="semigran")

    # ---- models ----
    p_models = subparsers.add_parser("models", help="List models available in LMStudio")
    p_models.add_argument("--base-url", default="http://localhost:1234/v1", help="LMStudio API URL")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    {
        "run": cmd_run,
        "import": cmd_import,
        "list": cmd_list,
        "chart": cmd_chart,
        "compare": cmd_compare,
        "models": cmd_models,
    }[args.command](args)


if __name__ == "__main__":
    main()
