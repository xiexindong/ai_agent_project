"""
评测命令行入口
==============
跑一轮评测、对比基线、产出报告，并可作为 CI 的回归门禁。

用法：
    # 1) 跑一轮，看报告
    python scripts/run_eval.py

    # 2) 效果达标后，把这次结果存为基线（下次改动就有对比对象了）
    python scripts/run_eval.py --update-baseline

    # 3) 改完提示词/路由后再跑，自动标出回归项
    python scripts/run_eval.py

    # 4) CI 里当门禁用：出现回归就 exit 1
    python scripts/run_eval.py --fail-on-regression --tolerance 0.02

    # 5) 只跑某一类用例
    python scripts/run_eval.py --tag risk

【为什么基线要单独一个文件而不是写死在代码里】
    基线是"上一次被认可的效果"，会随代码演进而变；
    代码是"怎么测"，不该跟着效果走。两者分开，才能看出
    "这次代码没变但分数变了"（环境抖动）还是"改了代码导致分数变了"。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 【为什么要把项目根塞进 sys.path】
#   直接 `python scripts/run_eval.py` 时 sys.path[0] 是 scripts/ 目录，
#   项目根不在其中，`import modules.xxx` 会 ModuleNotFoundError。
#   这里按文件位置算出项目根并插入，脚本从任意目录调用都能跑。
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 【为什么要在 import 项目模块之前压住日志级别】
#   logger 是首次 get_logger() 时才按 Settings.log_level 装配 sink，
#   默认 INFO 会把 40 条用例 × 每个环节的执行日志全打出来（几百行），
#   评测结果被淹在日志里看不见。这里提前把级别抬到 WARNING，
#   只留真正的告警；--log-level 可调回来看细节。
#   必须早于 import modules.*，否则 logger 已装配，改环境变量无效。
os.environ.setdefault("LOG_LEVEL", "WARNING")

from modules.eval import (  # noqa: E402
    diff_summary,
    load_cases,
    render_ci_summary,
    render_markdown,
    run_eval,
    validate_cases,
)

_DEFAULT_CASES = "eval_cases.jsonl"
_DEFAULT_BASELINE = "eval_baseline.json"
_DEFAULT_REPORT_MD = "eval_report.md"
_DEFAULT_REPORT_JSON = "eval_report.json"


def _read_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[warn] 基线文件解析失败，按无基线处理: {e}")
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="AI Agent 效果评测（含基线对比与回归门禁）")
    ap.add_argument("--cases", default=_DEFAULT_CASES, help="用例文件（JSONL）")
    ap.add_argument("--mode", default="offline", choices=["offline", "online"],
                    help="offline=确定性链路（无外部依赖）；online=完整链路（需密钥与网络）")
    ap.add_argument("--tag", action="append", default=[],
                    help="只跑带该标签的用例，可重复传")
    ap.add_argument("--max-tools", type=int, default=8, help="单轮工具名单上限")
    ap.add_argument("--baseline", default=_DEFAULT_BASELINE, help="基线文件路径")
    ap.add_argument("--update-baseline", action="store_true",
                    help="把本次结果写为基线")
    ap.add_argument("--out", default=_DEFAULT_REPORT_MD, help="Markdown 报告路径")
    ap.add_argument("--json-out", default=_DEFAULT_REPORT_JSON, help="JSON 报告路径")
    ap.add_argument("--tolerance", type=float, default=0.005,
                    help="指标变化小于该值视为持平，避免浮点噪声误判回归")
    ap.add_argument("--fail-on-regression", action="store_true",
                    help="出现回归项时以退出码 1 结束（CI 门禁用）")
    ap.add_argument("--verbose", action="store_true", help="报告里带上全量用例明细")
    ap.add_argument("--log-level", default="WARNING",
                    help="评测期间的项目日志级别（WARNING/ERROR/INFO）")
    args = ap.parse_args()

    os.environ["LOG_LEVEL"] = args.log_level

    # ------------------------------------------------------------------
    # 【为什么先校验用例集再跑】
    #   id 重复会让基线对比按 id 对齐时错位，跑完才发现等于白跑一轮。
    #   这一步很便宜（毫秒级），先做掉省时间。
    # ------------------------------------------------------------------
    try:
        cases = load_cases(args.cases)
    except FileNotFoundError as e:
        print(f"[error] {e}")
        return 2
    problems = validate_cases(cases)
    if problems:
        print("[error] 用例集自检未通过：")
        for p in problems:
            print(f"  - {p}")
        return 2
    print(f"用例集自检通过：{len(cases)} 条" + (f"（过滤标签: {args.tag}）" if args.tag else ""))

    report = run_eval(
        cases_path=args.cases,
        mode=args.mode,
        tags=args.tag or None,
        max_tools=args.max_tools,
        verbose=args.verbose,
    )

    base = _read_json(args.baseline)
    diff = None
    if base:
        diff = diff_summary(base.get("summary", {}), report["summary"], args.tolerance)

    # ---- 落盘 ----
    Path(args.json_out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(args.out).write_text(
        render_markdown(report, diff, args.baseline if base else ""), encoding="utf-8"
    )
    if args.update_baseline:
        Path(args.baseline).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"基线已更新: {args.baseline}")

    print()
    print(render_ci_summary(report, diff))
    print(f"\n报告已写入: {args.out}（Markdown） / {args.json_out}（JSON）")
    if not base:
        print("提示：当前无基线。效果确认后执行 --update-baseline 存档，"
              "下次改动即可自动对比。")

    # ---- 门禁 ----
    if args.fail_on_regression and diff:
        regressed = [d for d in diff if d.get("status") == "regressed"]
        if regressed:
            print("\n[门禁未通过] 检测到回归项：")
            for d in regressed:
                print(f"  - {d['metric']}: {d['base']} → {d['cur']}（{d['delta']}）")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
