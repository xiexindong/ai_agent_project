"""
接口压测 / 成本度量脚本
========================
产出面试和汇报里真正需要的三个数字：QPS、P95 延迟、单次对话成本（估算）。

用法：
    python main.py serve                       # 先起服务
    python scripts/bench.py --requests 50 --concurrency 8 --api-key <你的key>
    python scripts/bench.py --out bench_report.json    # 结果落盘，方便对比前后优化

设计取舍：
    - 用 SSE 流式读取直到 done 事件：真实用户感知的是「最后一个字出来的时间」，
      只测首个响应头会严重低估延迟。
    - 并发用线程池而不是 asyncio：Agent 内部本身就是线程池 + 阻塞 IO，
      用线程池压更贴近真实部署形态。
    - 分位数用 P50/P90/P95/P99 全套：只看平均值会掩盖长尾，
      而长尾正是用户投诉的来源。
"""
import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx


def pct(sorted_values: list, p: float) -> float:
    """取分位数（最近邻法）：空列表返回 0，避免除零。"""
    if not sorted_values:
        return 0.0
    idx = min(int(round((p / 100) * (len(sorted_values) - 1))), len(sorted_values) - 1)
    return sorted_values[idx]


def _drain_sse(client: httpx.Client, url: str, payload: dict, headers: dict, timeout: float) -> dict:
    """
    发一次 /chat 并把 SSE 流读完。

    返回：{"ok": bool, "throttled": bool, "ttfb_ms": int, "total_ms": int,
           "events": int, "error": str}

    【为什么不能只看 HTTP 状态码】
        本项目出错时也会推 SSE 的 error 事件并以 done 收尾，HTTP 依然是 200。
        只判状态码会把"重试 3 次后失败"的对话统计成成功 —— 实测就把 28/28 全失败
        报成了 28/28 成功。所以必须读流内容，认 event: error。

    【为什么要把 429 单独拎出来】
        429 是限流生效，不是服务故障。混在错误率里会得出"错误率 20%"这种
        误导性结论 —— 实际上那是治理在起作用，恰恰是要展示的能力。
    """
    started = time.time()
    ttfb = None
    events = 0
    biz_error = ""
    try:
        with client.stream("POST", url, json=payload, headers=headers, timeout=timeout) as resp:
            resp.raise_for_status()
            cur_event = ""
            for line in resp.iter_lines():
                if ttfb is None:
                    ttfb = (time.time() - started) * 1000     # 首字节：用户看到"在思考"的时间
                if line.startswith("event:"):
                    cur_event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    events += 1
                    try:
                        data = json.loads(line[5:].strip())
                    except Exception:
                        data = {}
                    # 输入：event: error / data: {"message":"Agent推理异常: ..."}
                    # 输出：biz_error = "Agent推理异常: ..."
                    if cur_event == "error":
                        biz_error = str(data.get("message") or data)[:200]
                if cur_event == "done":
                    break
    except httpx.HTTPStatusError as e:
        # 输入：服务端返回 401/403/429 等状态码 → 输出：throttled=True 仅当 429
        return {"ok": False, "throttled": e.response.status_code == 429, "ttfb_ms": 0,
                "total_ms": int((time.time() - started) * 1000), "events": events,
                "error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        return {"ok": False, "throttled": False, "ttfb_ms": 0,
                "total_ms": int((time.time() - started) * 1000), "events": events,
                "error": f"{type(e).__name__}: {e}"}
    return {"ok": not biz_error, "throttled": False, "ttfb_ms": int(ttfb or 0),
            "total_ms": int((time.time() - started) * 1000), "events": events,
            "error": biz_error}


def main() -> int:
    ap = argparse.ArgumentParser(description="AI Agent 接口压测（QPS / 分位延迟 / 成本估算）")
    ap.add_argument("--url", default="http://127.0.0.1:8000/chat", help="对话接口地址")
    ap.add_argument("--api-key", default="", help="X-API-Key 请求头的值")
    ap.add_argument("--api-keys", default="",
                    help="多个 Key（逗号分隔）轮流使用；服务端按用户限流时用它避免 429 干扰")
    ap.add_argument("--query", default="帮我算一下 3*7，然后用一句话总结结果", help="压测用的提问")
    ap.add_argument("--requests", type=int, default=20, help="总请求数")
    ap.add_argument("--concurrency", type=int, default=4, help="并发数")
    ap.add_argument("--timeout", type=float, default=120.0, help="单请求超时（秒）")
    ap.add_argument("--price-per-1k", type=float, default=0.0,
                    help="每千 token 单价（元）；填了才估算成本")
    ap.add_argument("--avg-tokens", type=int, default=0,
                    help="单次对话平均 token 数（从 /admin/bill 拿到后填这里）")
    ap.add_argument("--out", default="", help="结果写入 JSON 文件路径")
    args = ap.parse_args()

    # 【为什么要支持多 Key 轮换】服务端限流是按「用户」算的（本项目 20 次/60 秒），
    #   单 Key 压 20 次以上必然吃 429，测出来的 QPS 是限流器的上限而不是服务的上限。
    #   多 Key 轮换才能量到服务本身的吞吐。
    key_pool = [k.strip() for k in args.api_keys.split(",") if k.strip()]
    if args.api_key and args.api_key not in key_pool:
        key_pool.insert(0, args.api_key)

    payload = {"query": args.query, "session_id": ""}
    results: list[dict] = []

    def _headers_for(i: int) -> dict:
        h = {"Content-Type": "application/json"}
        if key_pool:
            h["X-API-Key"] = key_pool[i % len(key_pool)]
        return h

    print(f"压测开始: {args.requests} 请求 / 并发 {args.concurrency} → {args.url}"
          f"（轮换 {len(key_pool)} 个 Key）")
    wall_start = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        # 【为什么每个线程单独建 client】httpx.Client 不是线程安全的，
        #   共用实例会在高并发下报连接池错乱，那是压测工具的 bug 不是服务的。
        futures = []
        for _i in range(args.requests):
            futures.append(pool.submit(
                lambda idx=_i: _drain_sse(httpx.Client(), args.url, payload,
                                          _headers_for(idx), args.timeout)
            ))
        for f in futures:
            results.append(f.result())
    wall = time.time() - wall_start

    ok_list = [r for r in results if r["ok"]]
    err_list = [r for r in results if not r["ok"]]
    # 429 是限流生效（治理能力），真实错误才是问题：两者必须分开统计
    throttled = [r for r in err_list if r.get("throttled")]
    real_err = [r for r in err_list if not r.get("throttled")]
    # 业务失败（HTTP 200 但流里有 error 事件）与网络/鉴权失败分开看，定位更快
    biz_err = [r for r in real_err if r.get("error") and not r["error"].startswith("HTTP")]
    durations = sorted(r["total_ms"] for r in ok_list)
    ttfbs = sorted(r["ttfb_ms"] for r in ok_list)

    qps = len(ok_list) / wall if wall > 0 else 0.0
    report = {
        "url": args.url,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "success": len(ok_list),
        "throttled_429": len(throttled),
        "failed": len(real_err),
        "error_rate": round(len(real_err) / max(len(results), 1), 4),
        "wall_seconds": round(wall, 2),
        "qps": round(qps, 2),
        "latency_ms": {
            "avg": round(statistics.mean(durations), 1) if durations else 0,
            "p50": pct(durations, 50), "p90": pct(durations, 90),
            "p95": pct(durations, 95), "p99": pct(durations, 99),
            "max": durations[-1] if durations else 0,
        },
        "ttfb_ms": {"p50": pct(ttfbs, 50), "p95": pct(ttfbs, 95)},
        "biz_failed": len(biz_err),
        "errors_sample": [e["error"] for e in real_err[:5]],
    }

    # 【成本估算】只在给了单价与 token 数时算，避免编造数字
    if args.price_per_1k and args.avg_tokens:
        report["cost"] = {
            "price_per_1k_tokens": args.price_per_1k,
            "avg_tokens_per_chat": args.avg_tokens,
            "yuan_per_chat": round(args.price_per_1k * args.avg_tokens / 1000, 4),
            "yuan_per_1k_chats": round(args.price_per_1k * args.avg_tokens / 1000 * 1000, 2),
        }

    print("\n" + "=" * 56)
    print(f"  成功 {report['success']}/{report['requests']}  "
          f"限流(429) {report['throttled_429']}  真实错误率 {report['error_rate']:.1%}"
          f"（其中业务失败 {report['biz_failed']}）")
    print(f"  QPS        {report['qps']}")
    print(f"  延迟(ms)   平均 {report['latency_ms']['avg']} | "
          f"P50 {report['latency_ms']['p50']} | P90 {report['latency_ms']['p90']} | "
          f"P95 {report['latency_ms']['p95']} | P99 {report['latency_ms']['p99']}")
    print(f"  首字节(ms)  P50 {report['ttfb_ms']['p50']} | P95 {report['ttfb_ms']['p95']}")
    if "cost" in report:
        c = report["cost"]
        print(f"  成本        {c['yuan_per_chat']} 元/次 | {c['yuan_per_1k_chats']} 元/千次")
    if report["errors_sample"]:
        print(f"  错误样例    {report['errors_sample']}")
    print("=" * 56)

    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已写入: {args.out}")
    return 0 if not err_list else 1


if __name__ == "__main__":
    sys.exit(main())
