"""Read-only aggregation of this run, with a bounded final delivery write."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from native_common import ROOT, read_json, sha, utc, write_json


def optional(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def collect_counts() -> dict[str, Any]:
    workers = {p.parent.name: read_json(p) for p in (ROOT / "models").glob("*/WORKER.json")}
    predictions = {
        p.parent.name: read_json(p) for p in (ROOT / "results").glob("*/PREDICTION.json")
    }
    all_records = [*workers.values(), *predictions.values()]
    prep = optional(ROOT / "audit/PREPROCESSING_ATTEMPTS.json")
    fit_ledger = optional(ROOT / "audit/FIT_ATTEMPTS.json")
    return {
        "reserved_model_attempts": fit_ledger.get("reserved_attempts", 0),
        "model_fit_calls_entered": sum(x.get("fit_calls_entered", 0) for x in workers.values()),
        "model_fit_calls_returned": sum(x.get("fit_calls_returned", 0) for x in workers.values()),
        "successful_training_tasks": sum(x.get("status") == "SUCCESS" for x in workers.values()),
        "project_preprocessor_fit_calls_entered": prep.get("fit_calls_entered", 0),
        "successful_project_preprocessors": sum(
            x.get("status") == "SUCCESS" for x in prep.get("attempts", [])
        ),
        "artificial_preprocessor_fits": optional(ROOT / "artificial/NATIVE_ACCEPTANCE.json").get(
            "artificial_preprocessor_fit_calls", 0
        ),
        "model_load_calls_entered": sum(x.get("model_loads", 0) for x in all_records),
        "model_load_calls_returned": sum(x.get("model_loads_returned", 0) for x in all_records),
        "prediction_calls_entered": sum(x.get("prediction_calls", 0) for x in all_records),
        "prediction_calls_returned": sum(
            x.get("prediction_calls_returned", 0) for x in all_records
        ),
        "training_check_row_predictions_returned": sum(
            x.get("training_check_rows_per_call", 0) * x.get("prediction_calls_returned", 0)
            for x in workers.values()
        ),
        "review_row_predictions_returned": sum(
            x.get("rows", 0) * x.get("prediction_calls_returned", 0) for x in predictions.values()
        ),
        "extra_model_fits": 0,
        "bootstrap_draws": 0,
        "retries": 0,
        "workers": workers,
        "predictions": predictions,
    }


def md_table(rows: list[list[Any]], header: list[str]) -> str:
    def text(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, float):
            return f"{value:.10g}"
        return str(value).replace("|", "/")

    return "\n".join(
        [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
            *["| " + " | ".join(text(x) for x in row) + " |" for row in rows],
        ]
    )


def write_delivery(
    resource: dict[str, Any], clocks: dict[str, Any], startup: dict[str, Any]
) -> dict[str, Any]:
    config = read_json(ROOT / "config/EXECUTION_CONFIG.json")
    native = optional(ROOT / "artificial/NATIVE_ACCEPTANCE.json")
    pipeline = optional(ROOT / "audit/PIPELINE_STATUS.json")
    results = optional(ROOT / "results/DESCRIPTIVE_RESULTS.json")
    counts = collect_counts()
    complete = (
        resource.get("stop_reason") == "NORMAL_EXIT"
        and native.get("status") == "PASS"
        and pipeline.get("status") == "COMPLETED"
        and bool(results)
        and counts["model_fit_calls_entered"] == counts["model_fit_calls_returned"] == 6
        and counts["successful_training_tasks"] == counts["successful_project_preprocessors"] == 6
        and counts["model_load_calls_entered"] == counts["model_load_calls_returned"] == 12
        and counts["prediction_calls_entered"] == counts["prediction_calls_returned"] == 18
    )
    protected: list[dict[str, Any]] = []
    baseline = read_json(ROOT / "config/PROTECTED_BASELINE.json")
    entries = baseline["files"] if isinstance(baseline, dict) and "files" in baseline else baseline
    if isinstance(entries, dict):
        entries = [{"path": path, "sha256": value} for path, value in entries.items()]
    for entry in entries:
        path = Path(entry["path"])
        actual = sha(path)
        protected.append({**entry, "actual_sha256": actual, "unchanged": actual == entry["sha256"]})
    write_json(ROOT / "audit/PROTECTED_AFTER.json", protected, exclusive=True)
    complete = complete and all(p["unchanged"] for p in protected)
    data_reads = []
    if (ROOT / "audit/DATA_READS.jsonl").exists():
        data_reads = [
            json.loads(line)
            for line in (ROOT / "audit/DATA_READS.jsonl").read_text(encoding="utf-8").splitlines()
        ]
    final = {
        "status": "PENDING_FINAL_BUDGET" if complete else "STOPPED_NO_RETRY",
        "candidate_complete": complete,
        "real_execution_started": pipeline.get("stage") not in (None, "phase1", "entry_startup"),
        "time_utc": utc(),
        "stopped_stage": None if complete else pipeline.get("stage", "entry_startup"),
        "native_acceptance_status": native.get("status", "NOT_STARTED"),
        "native_checks_passed": sum(c["passed"] for c in native.get("checks", [])),
        "native_checks_recorded": len(native.get("checks", [])),
        "real_execution_status": (
            pipeline.get("status", "NOT_STARTED")
            if pipeline.get("stage") not in (None, "phase1", "entry_startup")
            else "NOT_STARTED"
        ),
        "counts": counts,
        "external_resource": resource,
        "clocks": clocks,
        "startup": startup,
        "data_read_receipts": data_reads,
        "source_alignment": optional(ROOT / "audit/SOURCE_ALIGNMENT.json"),
        "train_label_scope": optional(ROOT / "audit/TRAIN_LABEL_SCOPE.json"),
        "review_label_scope": optional(ROOT / "audit/REVIEW_LABEL_SCOPE.json"),
        "protected_unchanged": all(p["unchanged"] for p in protected),
        "no_independent_validation_no_statistics_no_paper_changes": True,
    }
    write_json(ROOT / "FINAL_STATUS.json", final, exclusive=True)
    costs = optional(ROOT / "audit/COSTS.json")
    lines = [
        "# 六组回顾开发结果与下一步投入判断",
        "",
        (
            "**六组计算完成，最终预算核对待定。**"
            if complete
            else f"**在{final['stopped_stage']}阶段停止；不重试。**"
        )
        + f" 原生接入：{final['native_acceptance_status']}；"
        + f"阶段二启动：{final['real_execution_started']}；"
        + f"真实执行：{final['real_execution_status']}。",
        "",
        "本次最小修复：规范化一次后携带源行身份进行排列回归，按ID核对公开内容、"
        "特征和来源；同步修复尾部预算与最终状态。生产历史、规范化、C07及评价定义保持原样。",
        "",
        f"实际项目模型fit进入/返回：{counts['model_fit_calls_entered']}/"
        f"{counts['model_fit_calls_returned']}；项目预处理fit进入/成功："
        f"{counts['project_preprocessor_fit_calls_entered']}/"
        f"{counts['successful_project_preprocessors']}；"
        f"模型加载进入/返回：{counts['model_load_calls_entered']}/{counts['model_load_calls_returned']}；"
        f"预测调用进入/返回：{counts['prediction_calls_entered']}/{counts['prediction_calls_returned']}。"
        f" 人工预处理规则fit另计{counts['artificial_preprocessor_fits']}次，不是项目fit。",
        "",
        f"正式入口外开发与静态检查历时{clocks.get('development_seconds', 'NA')}秒，"
        "包含编写、检查和工具等待，未计作免费投入；见audit/DEVELOPMENT_COST.json。",
    ]
    if data_reads:
        lines.extend(
            [
                "",
                "数据及标签实际范围：仅本轮BankSim原包。整份CSV的字节遍历包含0–179全部区域的fraud字节；"
                "特征模块只收到公共列。"
                + (
                    "标签投影器已进入，会暂时解码全部字段字符串。"
                    if (ROOT / "audit/TRAIN_LABEL_ACCESS_ENTERED.json").exists()
                    or (ROOT / "audit/REVIEW_LABEL_ACCESS_ENTERED.json").exists()
                    else "标签投影器尚未进入。"
                )
                + "标签使用范围限制为训练0–107及全部模型/评分锁定后的回顾126–179；"
                "108–125仅作无标签事件历史，无新的选型。读取详情逐次见audit/DATA_READS.jsonl。",
                "实际训练标签投影完成记录："
                + str(
                    final["train_label_scope"]
                    or optional(ROOT / "audit/TRAIN_LABEL_ACCESS_ENTERED.json")
                    or "未进入"
                )
                + "；实际回顾标签投影完成记录："
                + str(
                    final["review_label_scope"]
                    or optional(ROOT / "audit/REVIEW_LABEL_ACCESS_ENTERED.json")
                    or "未进入"
                )
                + "。中断且无完成流水时不声称标签使用数为零。",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "数据及标签实际范围：没有项目源读取流水；原生接入若未通过，真实入口未开放。"
                "没有用旧模型或评分缓存替代本轮任务。",
            ]
        )
    lines.extend(
        [
            "",
            "## 原生接入与固定配置",
            "",
            f"原生检查记录{final['native_checks_recorded']}项，通过{final['native_checks_passed']}项。"
            "人工内核与集合oracle只作参照，实际被核对的是本轮最终批处理实现。"
            "C07完整参数和实际导入版本见config/C07_PARAMETERS.json、artificial/NATIVE_C07_PARAMETERS.json"
            "及artificial/ENVIRONMENT_METADATA.json。HS为既定7step模拟访问情境，HL仍含全严格过去recency/available。",
            "",
            "下表按本次接入流水区分资源验收的通过、失败和未执行；没有玩具模型fit。"
            "Windows Job用于成员追踪与整体终止；"
            "Python审计仅覆盖相应进程的Python open事件，两者均不构成完整操作系统安全隔离。",
        ]
    )
    recorded_checks = {c["name"]: c["passed"] for c in native.get("checks", [])}
    lines.extend(
        [
            "",
            md_table(
                [
                    [
                        action,
                        ("PASS" if recorded_checks[name] else "FAIL")
                        if name in recorded_checks
                        else "NOT_RUN",
                    ]
                    for action, name in [
                        ("正常退出", "real tiny subprocess toy_normal"),
                        ("超时及树终止", "real tiny subprocess toy_descendant"),
                        ("缩小内存阈值", "real tiny subprocess toy_memory"),
                        ("缩小磁盘阈值", "real tiny subprocess toy_disk"),
                    ]
                ],
                ["资源子检查", "本次实际状态"],
            ),
        ]
    )
    old = optional(ROOT / "audit/OLD_ATTEMPT_COSTS.json")
    lines.extend(
        [
            "",
            f"旧run保留：入口1次、人工预处理6次、项目模型/预处理fit均为0；"
            f"旧开发{old.get('old_development_seconds')}秒，"
            f"旧正式调用{old.get('old_invocation', {}).get('wall_seconds')}秒，"
            "停止后复核成本亦保留。详见audit/OLD_ATTEMPT_COSTS.json，不抹去或虚构返还名额。",
        ]
    )
    if complete:
        effects = results["effects"]
        rows = results["whole"]
        if effects["J"]:
            finding = (
                f"在当前固定C07及已暴露BankSim窗口，加入商户直接编码改变了额外历史的观察捕获增量："
                f"无直接编码时HL−HS捕获差为{effects['delta0_TP']}，有直接编码时为{effects['delta1_TP']}，"
                f"整数交互J={effects['J']}，共同P={effects['common_P']}，I={effects['I']:.10g}。"
            )
        else:
            finding = (
                f"在当前固定C07及已暴露BankSim窗口，商户直接编码没有改变本轮观察到的HL−HS捕获增量"
                f"（两者均为{effects['delta0_TP']}，J=0，P={effects['common_P']}，I=0）。"
            )
        lines.extend(
            [
                "",
                "## 主窗口结果",
                "",
                finding,
                "",
                "这是所定义训练流程的一次描述性差值，不解释成信息替代机制、等效性或可推广历史访问规则。",
                "",
                md_table(
                    [
                        [
                            c,
                            *[
                                rows[c][k]
                                for k in ("N", "P", "K", "TP", "Recall", "Precision", "AP")
                            ],
                            rows[c]["boundary_tie"]["tied_rows"],
                            rows[c]["boundary_tie"]["slots_from_tie"],
                            rows[c]["boundary_tie"]["tp_min"],
                            rows[c]["boundary_tie"]["tp_max"],
                        ]
                        for c in config["cells"]
                    ],
                    [
                        "组",
                        "N",
                        "P",
                        "K",
                        "TP",
                        "Recall",
                        "Precision",
                        "AP",
                        "边界同分数",
                        "同分入选槽",
                        "并列TP下限",
                        "并列TP上限",
                    ],
                ),
                "",
                md_table(
                    [
                        [
                            window,
                            *[
                                block[k]
                                for k in (
                                    "common_P",
                                    "delta0_TP",
                                    "delta1_TP",
                                    "J",
                                    "delta0",
                                    "delta1",
                                    "I",
                                )
                            ],
                        ]
                        for window, block in [("126–179", effects)]
                    ],
                    ["窗口", "P", "Δ0捕获差", "Δ1捕获差", "J", "Δ0", "Δ1", "I"],
                ),
                "",
                f"整数换算：I=({effects['delta1_TP']}−({effects['delta0_TP']}))/"
                f"{effects['common_P']}={effects['J']}/{effects['common_P']}。"
                "Δ与I均按共同正例分母换算，未作四舍五入后的相减。",
                "",
                md_table(
                    [
                        [name, tp, effects["history_vs_none_recall"][name]]
                        for name, tp in effects["history_vs_none"].items()
                    ],
                    ["历史相对H0", "整数捕获增量", "Recall增量"],
                ),
                "",
                "HS捕获余量为min(P,K)−TP："
                + "，".join(f"{c}={value}" for c, value in effects["HS_headroom"].items())
                + "。",
            ]
        )
        for segment, block in results["segments"].items():
            lines.extend(
                [
                    "",
                    f"## 固定分段 {segment}",
                    "",
                    md_table(
                        [
                            [
                                c,
                                *[
                                    block["rows"][c][k]
                                    for k in ("N", "P", "K", "TP", "Recall", "Precision", "AP")
                                ],
                            ]
                            for c in config["cells"]
                        ],
                        ["组", "N", "P", "K", "TP", "Recall", "Precision", "AP"],
                    ),
                    "",
                    f"Δ0TP={block['effects']['delta0_TP']}，"
                    f"Δ1TP={block['effects']['delta1_TP']}，J={block['effects']['J']}，"
                    f"I={block['effects']['I']}。",
                ]
            )
        lines.extend(
            [
                "",
                "三个分段各自分配ceil(3%N)容量，分段捕获差不要求加总为主窗口捕获差；"
                "三个固定分段全部报告，没有删除无正例step或挑选结果。",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## 停止与可用证据",
                "",
                f"外部停止原因：{resource.get('stop_reason')}。"
                f"入口错误：{pipeline.get('error', native.get('error', '见启动记录'))}。",
                "",
                f"已完成训练项：{pipeline.get('completed_cells', [])}。"
                "部分项和失败项保留，不以幸存组形成完整六组科学结论。"
                "若为资源停止，只表示本预算内未完成，不能解释为研究假设被否定。",
            ]
        )
    lines.extend(
        [
            "",
            "## 完整成本与执行记录",
            "",
            md_table(
                [
                    [key, value]
                    for key, value in costs.get("shared", {}).items()
                    if isinstance(value, (int, float, str, bool))
                ],
                ["共享环节（只计一次）", "记录值/秒"],
            ),
            "",
            md_table(
                [
                    [
                        c,
                        costs.get("preprocessing", {}).get(c, {}).get("fit_seconds"),
                        costs.get("training", {}).get(c, {}).get("resource", {}).get("seconds"),
                        counts["workers"].get(c, {}).get("optimizer_seconds"),
                        counts["workers"].get(c, {}).get("save_seconds"),
                        counts["workers"].get(c, {}).get("reload_seconds"),
                        costs.get("review", {}).get(c, {}).get("resource", {}).get("seconds"),
                        counts["predictions"].get(c, {}).get("native_prediction_seconds"),
                    ]
                    for c in config["cells"]
                ],
                [
                    "组",
                    "预处理fit秒",
                    "训练worker启动至退出秒",
                    "模型fit函数秒",
                    "保存秒",
                    "重载秒",
                    "回顾worker启动至退出秒",
                    "回顾预测函数秒",
                ],
            ),
            "",
            md_table(
                [
                    ["监督采样峰值RSS/GiB", resource.get("sampled_peak_rss", 0) / 2**30],
                    ["采样产物峰值/GiB", resource.get("max_new_directory_bytes", 0) / 2**30],
                    ["最大实际采样间隔/秒", resource.get("max_sample_interval")],
                    ["观察累计CPU/秒", resource.get("observed_cumulative_cpu")],
                    ["受控入口启动至退出/秒", resource.get("seconds")],
                    ["入口启动可用内存/GiB", startup.get("available_memory", 0) / 2**30],
                    ["入口启动D盘可用/GiB", startup.get("D_free", 0) / 2**30],
                ],
                ["外部监督", "记录值"],
            ),
            "",
            "全外部入口起止、交付写入和退出另见audit/ENTRY_CLOCK.json。"
            "子分项属于总墙钟的嵌套组成，不能全部相加充当总耗时。矩阵转换、输入输出字节、"
            "实际调用数、每50ms目标采样的实际间隔及失败记录见audit/COSTS.json、"
            "DATA_READS.jsonl、各worker和RESOURCE流水。短命进程、启动握手、退出清理及采样间隙"
            "可漏掉瞬时峰值或部分CPU；RSS求和可能重复计共享页。OS IO计数不等于物理扇区读量。",
            "",
            (
                "本次公共特征投影遍历完整原CSV，HS没有减少该字节遍历；"
                if any(x["operation"] == "public_feature_projection" for x in data_reads)
                else "本次没有已完成的真实公共CSV投影，不声称发生了完整原CSV特征读取；"
            )
            + "没有冷缓存对照，不宣称总体成本节省。人工及实际共享准备成本均保留。",
            "",
            "## 下一步投入判断",
            "",
            (
                "本轮回答了固定C07、单一已暴露BankSim环境中的条件性描述问题；目前只允许保留为开发证据。"
                "不自动增加42次选型或统计回放。若要进一步投入，需另行审查可反驳的开发外问题、"
                "合格评价安排及实际决策依据，不能仅因I非零就升级论文贡献。"
                if complete
                else "停止本次执行；没有完整六组结果，暂不作进一步科学投入结论。"
                "本轮不修改规则或依赖继续补跑，恢复需要新的明确版本与预算决定。"
            ),
            "",
            "未做新调参、早停、权重、校准、合并重拟合、bootstrap、显著性、非劣/等效、"
            "历史保留/删除建议、独立最终验证或算法创新声明；未修改论文、旧协议和原暂不执行记录。"
            f"保护记录逐项哈希一致：{final['protected_unchanged']}。",
            "",
            "本轮到此停止。",
        ]
    )
    (ROOT / "六组回顾开发结果与下一步投入判断.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if costs:
        cost_rows = []
        for cell in config["cells"]:
            row = {
                "cell": cell,
                "preprocessor_fit_s": costs.get("preprocessing", {})
                .get(cell, {})
                .get("fit_seconds"),
            }
            row.update(
                {
                    key: counts["workers"].get(cell, {}).get(key)
                    for key in (
                        "optimizer_seconds",
                        "save_seconds",
                        "reload_seconds",
                        "check_before_prediction_seconds",
                        "check_after_prediction_seconds",
                        "model_loads",
                        "prediction_calls",
                    )
                }
            )
            row.update(
                {
                    f"review_{key}": counts["predictions"].get(cell, {}).get(key)
                    for key in (
                        "model_load_seconds",
                        "native_prediction_seconds",
                        "model_loads",
                        "prediction_calls",
                    )
                }
            )
            cost_rows.append(row)
        with (ROOT / "results/执行成本与调用.csv").open(
            "x", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(cost_rows[0]))
            writer.writeheader()
            writer.writerows(cost_rows)
    return final
