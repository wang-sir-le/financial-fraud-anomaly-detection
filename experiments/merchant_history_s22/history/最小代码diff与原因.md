# 最小代码diff与原因

本次在新run_02恢复，run_01保持只读。完整补丁见[audit/MINIMAL_CODE_DIFF.patch](audit/MINIMAL_CODE_DIFF.patch)，原run73项文件哈希见config/OLD_ATTEMPT_SNAPSHOT.json。

| 文件 | 改动及原因 |
| --- | --- |
| phase1.py | 一次规范化生成并绑定源ID后，已带身份记录直接调用build_history；HS/HL均验原顺序、反转及seed20261006打乱；用一对一ID映射核对公开内容、特征和来源。保留失败断言的实质约束，并增加十项有意错误的拒绝检查与业务重复保留检查。从头执行原其余接入检查。 |
| identity_check.py（新增） | 人工验收比较器：行数、整数ID、唯一性及集合、逐ID公开内容、输出完整性、原1e-12特征精度、精确来源及同商户严格过去范围。拒绝重新编号、重复/遗漏ID、特征和来源错配；不按无序特征集合比较。 |
| launch.py | 新ROOT双重校验和新的开发起点；清单哈希等大工作后核对预算；报告、状态、清单与退出码共享尾部判断；最终状态写入后再检查，超限粘性降级，不仅记录Boolean。 |
| tail_budget.py（新增） | 小型共享预算决定函数。人工验正常、尾部墙钟/交付墙钟/产物/RSS超限及流程未完成时的状态/退出码；不改变任何上限或监督框架。 |
| delivery.py | 结果生成时先标记预算待终检，阶段二未启动时明确NOT_STARTED；按实际流水显示四项资源检查PASS/FAIL/NOT_RUN；仅在真实读取记录存在时描述相应读取；保留旧尝试成本。 |

native_batch.py、native_common.py、source_io.py、model_worker.py、pipeline.py、entry.py及supervision.py七文件逐字未改。生产normalize_public已有ID时会保留并检查唯一性，没有发现覆盖已有身份的缺陷；因此不修改生产规范化。原CSV行位置身份仍只在第一次投影规范化边界生成，并列SHA规则不变。

原EXECUTION_CONFIG每一个既有字段值及C07参数字节均一致；只新增新ROOT、只读恢复来源和修复范围元数据。新config的保护清单增加旧run全文件，未复制旧通过状态、锁、日志或结果。所有写入位置由新代码文件ROOT派生，正式入口还核对ROOT等于run_02且不等于run_01。

静态检查属于开发，不是新原生验收通过。正式恢复入口只启动一次；全部新人工接入通过才允许六次项目模型fit和六次项目预处理fit。入口外修复成本、旧人工6次规则fit、旧一次入口及旧开发/运行成本独立保留，不虚构科学额度返还。

科学定义、资源和停止条件按修复范围与计时记录执行，不改论文，不自动追加尝试。

