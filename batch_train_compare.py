from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_LABELS = {"rnn": "lstm", "transformer": "transformer"}
ADVANTAGE_LABELS = {"mc": "mc_adv", "gae": "gae_adv"}
REWARD_LABELS = {"simple": "simple_reward", "multi": "multi_reward"}

try:
    import torch
except ImportError:
    torch = None


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    model_arg: str
    model_label: str
    advantage_mode: str
    reward_mode: str
    seed: int


def build_experiment_specs(
    models: Sequence[str],
    advantage_modes: Sequence[str],
    reward_modes: Sequence[str],
    seeds: Sequence[int],
) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    multi_seed = len(seeds) > 1

    for model_arg in models:
        model_label = MODEL_LABELS[model_arg]
        for advantage_mode in advantage_modes:
            advantage_label = ADVANTAGE_LABELS[advantage_mode]
            for reward_mode in reward_modes:
                reward_label = REWARD_LABELS[reward_mode]
                for seed in seeds:
                    seed_suffix = f"_seed{seed}" if multi_seed else ""
                    specs.append(
                        ExperimentSpec(
                            name=(
                                f"{model_label}_{advantage_label}_{reward_label}"
                                f"{seed_suffix}"
                            ),
                            model_arg=model_arg,
                            model_label=model_label,
                            advantage_mode=advantage_mode,
                            reward_mode=reward_mode,
                            seed=seed,
                        )
                    )
    return specs


def _default_run_name() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _quote_command(command: Sequence[str]) -> str:
    if sys.platform.startswith("win"):
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def _json_load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _flatten_scalar_dict(
    row: dict[str, Any], prefix: str, payload: dict[str, Any] | None
) -> None:
    if not isinstance(payload, dict):
        return
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            row[f"{prefix}_{key}"] = value


def collect_experiment_result(
    spec: ExperimentSpec,
    save_dir: Path,
    return_code: int,
    elapsed_sec: float | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "experiment": spec.name,
        "model": spec.model_label,
        "model_arg": spec.model_arg,
        "advantage_mode": spec.advantage_mode,
        "reward_mode": spec.reward_mode,
        "seed": spec.seed,
        "status": status or ("ok" if return_code == 0 else "failed"),
        "return_code": return_code,
        "save_dir": str(save_dir),
        "elapsed_sec": elapsed_sec,
        "run_log": str(save_dir / "run.log"),
    }

    summary = _json_load(save_dir / "training_summary.json")
    if summary is not None:
        row["best_selection_score"] = summary.get("best_selection_score")
        row["history_length"] = summary.get("history_length")
        _flatten_scalar_dict(row, "final_train", summary.get("final_train_metrics"))
        _flatten_scalar_dict(row, "final_val", summary.get("final_val_metrics"))

    validation = _json_load(save_dir / "validation_report.json")
    if validation is not None:
        _flatten_scalar_dict(row, "best_val_factor", validation.get("factor_metrics"))
        _flatten_scalar_dict(row, "strategy", validation.get("strategy_backtest"))
        _flatten_scalar_dict(row, "benchmark", validation.get("benchmark_backtest"))
        row["benchmark_code"] = validation.get("benchmark_code")
        row["warmup_days"] = validation.get("warmup_days")

    strategy_ann = row.get("strategy_annual_return")
    benchmark_ann = row.get("benchmark_annual_return")
    if isinstance(strategy_ann, (int, float)) and isinstance(
        benchmark_ann, (int, float)
    ):
        row["excess_annual_return"] = strategy_ann - benchmark_ann

    strategy_sharpe = row.get("strategy_sharpe_ratio")
    benchmark_sharpe = row.get("benchmark_sharpe_ratio")
    if isinstance(strategy_sharpe, (int, float)) and isinstance(
        benchmark_sharpe, (int, float)
    ):
        row["excess_sharpe_ratio"] = strategy_sharpe - benchmark_sharpe

    return row


def build_train_command(
    args: argparse.Namespace, spec: ExperimentSpec, save_dir: Path
) -> list[str]:
    return [
        sys.executable,
        "train.py",
        "--train_start",
        args.train_start,
        "--train_end",
        args.train_end,
        "--val_start",
        args.val_start,
        "--val_end",
        args.val_end,
        "--iterations",
        str(args.iterations),
        "--episodes",
        str(args.episodes),
        "--pool_size",
        str(args.pool_size),
        "--horizon",
        str(args.horizon),
        "--lr",
        str(args.lr),
        "--n_hold",
        str(args.n_hold),
        "--n_swap",
        str(args.n_swap),
        "--commission",
        str(args.commission),
        "--benchmark_code",
        args.benchmark_code,
        "--save_dir",
        str(save_dir),
        "--gamma",
        str(args.gamma),
        "--gae_lambda",
        str(args.gae_lambda),
        "--advantage_mode",
        spec.advantage_mode,
        "--reward_mode",
        spec.reward_mode,
        "--device",
        args.device,
        "--seed",
        str(spec.seed),
        "--model",
        spec.model_arg,
        "--tf_embed_dim",
        str(args.tf_embed_dim),
        "--tf_nhead",
        str(args.tf_nhead),
        "--tf_num_layers",
        str(args.tf_num_layers),
        "--tf_dim_feedforward",
        str(args.tf_dim_feedforward),
        "--tf_dropout",
        str(args.tf_dropout),
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_results_csv(path: Path, results: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in results:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def _format_metric(value: Any, fmt: str = "{:.4f}") -> str:
    if not isinstance(value, (int, float)):
        return ""
    return fmt.format(value)


def _sort_results(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    def score(row: dict[str, Any]) -> tuple[int, float, float]:
        ok = 1 if row.get("status") == "ok" else 0
        best_val_ic = row.get("best_val_factor_ic")
        sharpe = row.get("strategy_sharpe_ratio")
        best_val_ic_num = (
            float(best_val_ic)
            if isinstance(best_val_ic, (int, float))
            else float("-inf")
        )
        sharpe_num = (
            float(sharpe) if isinstance(sharpe, (int, float)) else float("-inf")
        )
        return (ok, best_val_ic_num, sharpe_num)

    return sorted(results, key=score, reverse=True)


def write_markdown_report(path: Path, results: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Batch Training Comparison",
        "",
        f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Experiment | Model | Advantage | Reward | Seed | Status | Best Val IC | Best Val ICIR | Final Val IC | Strategy Sharpe | Excess Ann Return |",
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in _sort_results(results):
        lines.append(
            "| {experiment} | {model} | {advantage_mode} | {reward_mode} | {seed} | {status} | {best_val_ic} | {best_val_icir} | {final_val_ic} | {strategy_sharpe} | {excess_ann} |".format(
                experiment=row.get("experiment", ""),
                model=row.get("model", ""),
                advantage_mode=row.get("advantage_mode", ""),
                reward_mode=row.get("reward_mode", ""),
                seed=row.get("seed", ""),
                status=row.get("status", ""),
                best_val_ic=_format_metric(row.get("best_val_factor_ic")),
                best_val_icir=_format_metric(row.get("best_val_factor_icir")),
                final_val_ic=_format_metric(row.get("final_val_ic")),
                strategy_sharpe=_format_metric(row.get("strategy_sharpe_ratio")),
                excess_ann=_format_metric(
                    row.get("excess_annual_return"), "{:+.2%}"
                ),
            )
        )

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_batch_artifacts(
    group_dir: Path,
    args: argparse.Namespace,
    specs: Sequence[ExperimentSpec],
    results: Sequence[dict[str, Any]],
) -> None:
    write_json(
        group_dir / "batch_config.json",
        {
            "runner_args": vars(args),
            "experiments": [asdict(spec) for spec in specs],
        },
    )
    write_json(group_dir / "comparison_summary.json", list(results))
    write_results_csv(group_dir / "comparison_summary.csv", results)
    write_markdown_report(group_dir / "comparison_report.md", results)


def run_command_with_tee(command: Sequence[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        quoted = _quote_command(command)
        header = f"$ {quoted}\n\n"
        print(header, end="")
        log_file.write(header)

        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

        return process.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ablations for advantage estimation, reward shaping, and transformer."
    )
    parser.add_argument("--train_start", default="20240101")
    parser.add_argument("--train_end", default="20251231")
    parser.add_argument("--val_start", default="20260101")
    parser.add_argument("--val_end", default="20260501")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--pool_size", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_hold", type=int, default=20)
    parser.add_argument("--n_swap", type=int, default=3)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--benchmark_code", default="000300.SH")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument(
        "--device",
        default="cuda" if torch is not None and torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--tf_embed_dim", type=int, default=64)
    parser.add_argument("--tf_nhead", type=int, default=4)
    parser.add_argument("--tf_num_layers", type=int, default=3)
    parser.add_argument("--tf_dim_feedforward", type=int, default=256)
    parser.add_argument("--tf_dropout", type=float, default=0.1)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["rnn", "transformer"],
        default=["rnn", "transformer"],
    )
    parser.add_argument(
        "--advantage_modes",
        nargs="+",
        choices=["mc", "gae"],
        default=["mc", "gae"],
    )
    parser.add_argument(
        "--reward_modes",
        nargs="+",
        choices=["simple", "multi"],
        default=["simple", "multi"],
    )
    parser.add_argument("--save_root", default="output/batch_compare")
    parser.add_argument("--run_name", default=_default_run_name())
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip an experiment if its training_summary.json already exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands and create the batch manifest without starting training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = build_experiment_specs(
        models=args.models,
        advantage_modes=args.advantage_modes,
        reward_modes=args.reward_modes,
        seeds=args.seeds,
    )

    group_dir = Path(args.save_root) / args.run_name
    group_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    failures = 0

    for index, spec in enumerate(specs, start=1):
        save_dir = group_dir / spec.name
        save_dir.mkdir(parents=True, exist_ok=True)

        command = build_train_command(args, spec, save_dir)
        command_text = _quote_command(command)
        print(f"[{index}/{len(specs)}] {spec.name}")
        print(f"Command: {command_text}")

        if args.dry_run:
            results.append(
                {
                    "experiment": spec.name,
                    "model": spec.model_label,
                    "advantage_mode": spec.advantage_mode,
                    "reward_mode": spec.reward_mode,
                    "seed": spec.seed,
                    "status": "dry_run",
                    "return_code": 0,
                    "save_dir": str(save_dir),
                    "command": command_text,
                }
            )
            write_batch_artifacts(group_dir, args, specs, results)
            continue

        if args.skip_existing and (save_dir / "training_summary.json").is_file():
            print("Skipping existing completed run.\n")
            results.append(
                collect_experiment_result(
                    spec,
                    save_dir,
                    return_code=0,
                    elapsed_sec=0.0,
                    status="skipped_existing",
                )
            )
            write_batch_artifacts(group_dir, args, specs, results)
            continue

        started = time.time()
        return_code = run_command_with_tee(
            command,
            cwd=PROJECT_ROOT,
            log_path=save_dir / "run.log",
        )
        elapsed_sec = time.time() - started
        print(
            f"Finished {spec.name} with return code {return_code} in {elapsed_sec:.1f}s.\n"
        )

        results.append(
            collect_experiment_result(
                spec,
                save_dir,
                return_code=return_code,
                elapsed_sec=elapsed_sec,
            )
        )
        if return_code != 0:
            failures += 1

        write_batch_artifacts(group_dir, args, specs, results)

    print(f"Summary written to: {group_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
