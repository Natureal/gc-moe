# EP64 ECO 工作文档

本 session 的 EP64 工作文档：实验方法、ECO 现状、优化选项、实测计划。

- 只用 EP64 集群；只改本仓库（`/home/nanjun.cp/ep64_exp/gc-moe`，分支 `eco_ep64`）。
- `/data0/home/nanjun.cp/gc-moe` 及其 AGENTS.md 属于另一个 session（EP32），不读、不改。
- 工具链在 `~/work/mpo/ep64`。那里的 AGENTS.md 是历史流水账（混有 EP32 记录），EP64 结论以本文件为准。
- 最后更新：2026-11-15。
- **EP32 实验数据（2026-11-15 用户指定）**：real1500 零复用分段流 `~/work/mpo/data/real1500_vllm_dsv41/r32/seg1024/`（`stream40_seg.py`：全部 1483 decode track 切成 ≤1024 token 段、打乱、每 token 只用一次；32×32 槽 MTP-4 T=128；预热 50 步丢弃；历史窗口 [0,1000) → `hist_stats.npy`，EPLB/eco2/c16/c32 放置只用它；测量窗口 [1000,1595) 共 595 步 = 23800 call → `calls.npy`，第 c 个 call 对应层 c%40，段用尽于 1594 步，所以不是计划的 650 步）。block 峰值表：`r32_blkpeak2.py <dir>` → `<dir>/blkpeak.json`。旧 `meas_calls`/`stats_hist` 和 `data/ep32` 只作历史对照。EP32 T=128 下 gc-moe 对 P448-576 实际选 bm64。
- **EP32 默认数据集（2026-11-15 用户指定，以后默认用它）= 按请求分开的版本**：`r32/seg1024_disj/`。请求按 token 量 70/30 分层切成两组（`requests_stats.npy` / `requests_meas.npy`，没有重叠），各自跑零复用分段流：统计组 1000 步 → `hist_stats.npy`，测量组预热 50 步 + 340 步 = 13600 call → `calls.npy`。漂移（TV）0.068，seg1024 是 0.014；EPLB 用历史放置比事后最优多 +0.42 块、token 不均衡多 +0.056，seg1024 只有 +0.02 / +0.008（即对 EPLB 过于有利）。已知偏差：测量组只有 45 个长请求，每个平均同时占约 18 个槽，每步只有 183 个不同请求（线上是 1024 个），会放大请求自身的专家偏好，对 EPLB 不利。

## 1. 目标与口径

- 目标：EP64 decode 下 ECO 达到或超过 EPLB。同时压低 token/block 不均衡度（目标 < 1.05）、planner 耗时（目标 ≤ 12 µs）、每 GPU 活跃副本峰值和 block 峰值。
- 场景：DSV4.1-Flash（E=384，topk=6，H=5120，I=2304），64 rank，T=128 token/rank。routing 用 `routing_off.npy`（[1200, 64, 128, 6]，B 组，8 层轮换）。
- 对比配置只有 4 个：`eplb448 / eplb512 / eco448 / eco512`。448 = 每 GPU +1 slot，512 = +2 slot（416 不能被 64 整除）。
- 主指标 `kernel_us_est`：40-call CUDA graph 回放的平均每 call 时间减去 copy 图时间（bench 直接输出）。同时记 `p50`。
- 辅助指标：
  - token/block 不均衡度（max/mean）
  - 每 GPU block 峰值、活跃副本峰值
  - planner 耗时（`plan_start → plan_published`）
  - plan 等待（stall）
- 噪声：旧任务 3 次重复的 spread ≤ 0.5 µs，差异 < 1 µs 一律视为噪声。

## 2. 环境

### 2.1 集群（task `..._L20B_f444fa3f.inference`）

- 16 节点 × 4 L20B（aarch64）。每卡 152 SM：dispatch 16 / compute 136；ECO 时 compute 让出 2 SM 给 plan cluster。64 卡经 NVLink 互联（symm_mem P2P）。
- 访问规则：
  - 只用 `ssh e02-sg-cc54rhpcz03-<SN>-workdir`，需设 `WT_AUTH_SKIP_NODE_SITES=1`；不碰宿主机。
  - 节点上不跑 AI CLI。
  - 容器层会被重置，集群侧文件只放 `$W` 下。
- rank r 在 part `r // 4` 上：

| part | SN | IP | part | SN | IP |
|---|---|---|---|---|---|
| 0 | a0007 | 10.44.2.1（master，http 25280） | 8 | a0016 | 10.44.44.33 |
| 1 | a0005 | 10.44.59.232 | 9 | a0015 | 10.44.16.254 |
| 2 | a0004 | 10.44.27.22 | 10 | a0002 | 10.44.31.38 |
| 3 | a0003 | 10.44.7.46 | 11 | a0013 | 10.44.13.72 |
| 4 | a0009 | 10.44.68.15 | 12 | a0006 | 10.44.40.102 |
| 5 | a0011 | 10.44.66.228 | 13 | a0014 | 10.44.66.235 |
| 6 | a0008 | 10.44.4.7 | 14 | a0018 | 10.44.31.122 |
| 7 | a0017 | 10.44.75.55 | 15 | a0001 | 10.44.16.126 |

### 2.2 远端布局（`$W` = 各节点 login 目录 `.../workdir`）

- `$W/gcenv/site`：CuTe DSL 4.8.1。
  - 当前是 cu12 IR + cu12 runtime，11-08 10:25 后由另一 session 重装。
  - 包必须完整（`cutlass/` 下有 `cute/`、`base_dsl/` 等），否则报 `Symbols not found` 或 `No module named cutlass.cute`。
  - 不要用 `sgpkg/` 里的 cu13 libs wheel 覆盖它。
- `$W/gcenv/env.sh`：
  - 默认 PYTHONPATH 指向 `sgpkg/gc-moe-eco`，不要依赖这个默认值。
  - `CUTE_DSL_LIBS` 指向 cu12 runtime。
  - `GC_MOE_ECO_CACHE_DIR=$W/cache/eco`。
- `$W/sgpkg/` 下的包（md5 为 `gc_moe/impls/sm100_fp8_fp4_gc_moe.py` 的前 8 位）：

| 目录 | 内容 | md5 | 用途 |
|---|---|---|---|
| gc-moe-eco | r5 + 另一 session 的 DIAG 补丁（强制 row-tag 轮询，3 处超时 trap 改为 pass），11-08 12:13 原地修改 | a829580b | 不再使用 |
| gc-moe-r5b | HEAD 40e5e64，不含未提交改动 | 35db7c5c | 对照组 |
| gc-moe-nw | r5 + `warp_wait_count_u64` 回退 | 7c5786ab | 二分排查用 |
| gc-moe-late | r5 + plan 晚发射 | 7c5786ab | 二分排查用 |
| gcbench-ep64 | bench 与工具脚本 | — | 使用 |

- `$W/sgpkg/data/ep64` 内容：
  - `routing_off.npy`、`routing_eplb{448,512,576}.npy`
  - `eco{448,512,576}_metas{,_phy2log}.npy`
  - `eco448id_metas{,_phy2log}.npy`（identity 对照）
  - `hist_A.npy`、`N_t128.npy`
  - 本地副本在 `~/work/mpo/data/ep64`，没有 identity 文件。
- 日志：每个节点各一份，路径 `$W/runs/<tag>_node<k>.log`。

### 2.3 本地工具（`~/work/mpo/ep64`）

- 集群脚本：
  - `cluster.sh`：提供 `ep64_ssh <k> <cmd>`。
  - `ep64_deploy.sh <tar> <dest>`：经 part0 的 http 扇出并解包，dest 会被整个替换。
  - `ep64_launch.sh <nnodes> <tag> <args>`：`@D` 展开为 `$W/sgpkg/data`；用 env `SG_ENV`、`PORT`。
  - `ep64_queue.py <queue.json> <out.jsonl>`：串行执行并做 GPU 空闲检查；用 env `QUEUE_PORT`、`QUEUE_PREFIX`。
  - 结果放在 `results/`。
- bench `pkg/gcbench/gc_eco_bench.py`：
  - eco：`--mode eco --eco-meta @D/ep64/eco<P>_metas.npy --warm --routing @D/ep64/routing_off.npy`
  - eplb：`--mode plain --num-experts <P> --routing @D/ep64/routing_eplb<P>.npy`
  - native：`--mode plain --num-experts 384 --routing @D/ep64/routing_off.npy`
  - 计时：`--chunks 30 --passes 2 --warmup-passes 1`（40-call graph，一次 run 约 3700 个 MoE call）
  - 正确性：`--check --check-steps 40`（40 个 eager call，ECO 与 plain 逐 rank 比对输出）
  - graph trace：env `GC_MOE_ECO_TRACE=1` 加计时参数，输出 `graph_trace_us` 和 `per_call_us`。
  - eager trace：`--trace 16 [--trace-dump <path>]`，输出逐 slot 时间线 `trace_us`。汇总只含 slot < 32，其余看 dump。
  - `--block-m <bm>`：两种模式都钉死 block M。只钉 ECO 的话用 env `GC_MOE_ECO_BLOCK_M`（16–128）。
  - `--dump-plan`：check 模式下输出第一个 call 的每 rank quota 负载。
- 离线工具：
  - `pkg/gcbench/ep64_stats2.py`：用 Python 参考 planner `unit_ref.UnitPlanner` 重放，输出每 rank 的 token/block/active 和代价模型时间。
  - `unit_role_ep64.py`：单 GPU 模拟 R=64 的 plan 角色，做 bit-exact 对拍。

## 3. 实验规程（必须遵守）

1. **同时只跑一个队列。** 启动 `ep64_queue.py` 前确认本地没有队列进程、节点上没有 torchrun。现有的 GPU 空闲检查（util=0 且 mem<2000MiB）挡不住队列重叠。
2. **每个结果都能追溯到包。** 队列条目的 env 里显式写 PYTHONPATH 指向哪个包；部署后校验 16 节点 md5 一致；结果记录写上包名和 md5。
3. **先过正确性门槛。** 顺序是 `--check --check-steps 40` 通过 → 计时 → trace。
4. **对比条件一致。** routing、block M、graph-len/chunks/passes/warmup 全部相同；至少重复 3 次并交错执行；报中位数和 spread。
5. **计时和 trace 分开跑。** trace 有开销，只用来拆分时间，不用来报性能。
6. **失败先分诊，再下结论。**
   - 基础设施类：`EADDRINUSE`（端口被占）、`Symbols not found` / `ModuleNotFound`（DSL 包）、ssh 或 launch 失败。作废重跑，不算 kernel 结果。
   - kernel 类：某个 rank 先报 `ILLEGAL_INSTRUCTION`（或其他首发错误）；其余 rank 报 `combine readiness timeout ... peer=X`，60 s 后 `trap`，表现为 `LAUNCH_FAILED`。X 就是首发 rank。
   - dmesg：Xid 13 是 SM 异常；Xid 43 是 context 被拆除，属于次生。
   - 定位步骤：在 node0 日志里统计 `readiness timeout ... peer=` 找到首发 rank X → 去 part `X//4` 看该节点日志和 torchrun 的 "Root Cause"。
7. **概率性故障的判定。** 一次干净的失败就能证明 bug 存在；要判定"已修复"，至少 3 次计时 run 零失败。
8. **污染的 run 一律作废。** 包括队列重叠、端口冲突、环境替换期间跑的 run，并在记录里注明。

## 4. ECO 现状（2026-11-08）

### 4.1 代码

当前代码 = HEAD `40e5e64` + 未提交的约 1400 行（11 个文件），部署时称为 r5。

架构：
- 角色分工：
  - role 1 dispatch：路由、source-index、远端拉取。
  - role 2 compute：GEMM + scheduler。
  - role 3：combine。
  - role 4 plan：每个 rank 各算一份，结果在所有 rank 上一致，决定副本之间怎么切 token。
  - role 5 helper（可选）：plan 结束后回收它占的 SM。
- SM 划分：dispatch 16；compute 136，ECO 时 role 2 用 134，plan 占 compute 分区里的一个 2-SM cluster。
- planner：
  - 576 线程 = 16 个 plan warp + 2 个 index warp。
  - 目标：最慢 rank 的 wb·blocks + wa·active + tokens 最小。
  - EP64 轮数（`eco_unit_config`）：P448 为 warm 3 / cold 6 / two-hop 2；P512 为 warm 2 / cold 6 / two-hop 2。
  - wb、wa 按 block M 取最近的拟合值；bm192 没有拟合值，借用 128 的 293/52。
- block M（gc-moe 启发式）：ECO 按逻辑专家数 384 选，得 **bm192**；EPLB 按物理槽数选，448/512 都得 **bm128**。
- 发射与跨 call 同步：
  - plan 最先发射，顺序 4 → 1 → 2 →（5）→ 3；split 模式下 plan 以 2-CTA cluster 启动。
  - 跨 call 靠 `recv_parity` 双 bank + rank barrier，没有尾部屏障。

未提交改动与开关（括号内为默认值）：

| 改动 | 开关（默认） | 状态 |
|---|---|---|
| split-count 两 CTA plan（producer CTA 计数并推送 count 行，另一个 CTA 跑 planner） | `GC_MOE_ECO_SPLIT_COUNTS`（1） | **EP64 计时必崩的头号嫌疑**；EP32 上的收益在噪声内 |
| role1 拉取双缓冲流水线（plain 模式也生效） | `GC_MOE_PULL_STAGES`（2）/ `GC_MOE_PULL_PIPE`（1） | 关掉仍崩，不是唯一原因 |
| scheduler 单拷贝计数改用 `warp_wait_count_u64` | 无 | 回退仍崩，不是唯一原因 |
| plan 最先发射 | 无 | 回退仍崩，不是唯一原因 |
| trace 与 eco 解耦（plain 也能 trace） | `GC_MOE_ECO_TRACE`（0） | 可用 |
| route-prep / rows-after-pub | `GC_MOE_ECO_ROUTE_PREP` / `GC_MOE_ECO_ROWS_AFTER_PUB`（0） | 依赖 split，未验证 |
| 推测式 quota（输出不正确，只用于计时） | `GC_MOE_ECO_SPEC`（0） | 只作上界探针 |
| helper 回收 plan 的 SM | `GC_MOE_ECO_HELPER`（0） | EP64 未验证；EP32 上持平或变差 |
| 其他覆盖开关 | `GC_MOE_ECO_UNIT_CFG`（覆盖 wb,wa,warm,cold,hop2）、`GC_MOE_ECO_OBJECTIVE`、`GC_MOE_ECO_BLOCK_M` | |

### 4.2 正确性与稳定性

- `--check`：r5 与 r5base 都 bit-exact。
- 计时稳定性（一次 run 约 3700 call）：

| 变体 | split | 计时结果 |
|---|---|---|
| r5base（HEAD，没有 split 代码） | — | 1/1 通过（112.98 µs） |
| 干净 r5（nw / late 包） | 开 | 0/3 |
| r5+DIAG：默认 / PIPE=0 / STAGES=1 / e448 / id448 / SPEC×2 | 开 | 0/7 |
| r5+DIAG：`SPLIT_COUNTS=0` | 关 | 3/3 通过（111.14 / 111.53 / 111.71）；另有 1 次被污染，但没崩 |
| r5+DIAG：eager check 2/4/8/40 步 | 开 | 8/8 通过 |

- 故障形态：
  - 每次是随机一个 rank 先出错（观察到 rank 40 / 9 / 44 / node0 上的 rank）。
  - 能看清的两例里，首发错误是 `ILLEGAL_INSTRUCTION`；其余 rank 60 s 后 readiness 超时并 trap。
  - 每个 call 的出错概率很低：几十个 eager call 通常不触发，一次计时 run 必然触发。
- 已排除为"唯一原因"的项：pull 流水线、`warp_wait_count_u64`、plan 最先发射、plan→quota 依赖（SPEC 下仍崩）、真实副本布局（identity 布局也崩）。
- 注意：EP32 上单 CTA 的 m3 build（eco416id）也出现过约 50% 的 flaky illegal instruction。所以 split 可能只是放大器，不一定是唯一根因，需要用 coredump 定位到具体指令。
- 修正：之前把 Xid 108（远端写到未映射地址）当作根因证据，这是过度解读。它更可能是次生故障：某个 rank 死掉后，其他 rank 还在往它的内存里写。

### 4.3 性能（当前任务上仅有的有效数据）

| 配置 | 包 | kernel_us_est | 备注 |
|---|---|---|---|
| eplb512 | r5（DIAG 之前） | 90.35 / 90.34 | 另一 session 所测；EPLB 路径也包含 pull 流水线 |
| eco512，split=0 | r5+DIAG | 111.14 / 111.53 / 111.71 | 默认 bm192 |
| eco512 | r5base | 112.98 | 只跑了 1 次 |

- ECO512 比 EPLB512 慢约 21 µs（+23%）；r5 比 HEAD 只快约 1.5 µs。
- 还缺：eplb448、eco448、bm128 对照、任何 trace 拆分、identity 固定开销、在线 plan 质量。

### 4.4 历史数据（旧任务 / 旧 build，只作假设来源，不能直接比较）

- 旧任务（11-03/05，build 2f064ae，plan 占 1 个 dispatch SM）：
  - native384 180.95（bm192）；EPLB 448/512/640/896 分别为 89.46 / 90.15 / 89.25 / 99.90。
  - 离线不均衡度：native token 4.77，block 峰值 22.7；EPLB token 1.22 / 1.24 / 1.23 / 1.20，block 峰值 12.02 / 12.48 / 15.99 / 22.73，活跃副本峰值 7 / 8 / 10 / 14。
- 旧 ECO（c32 metas，bm192）：eco448 101.93，eco512 107.38。eco448 用 bm128 为 98.2，比 bm192 快 3.7 µs，但仍比 eplb448（89.7）慢 8.5 µs。
- 旧 identity 固定开销约 13.6 µs（eco448id 194.4 vs plain 181.0–181.7），其中约 5.9 µs 来自让出一个 dispatch SM。现在 plan 已改为占 compute 分区。
- 旧 P640 trace：plan_published 在 32–34 µs，而 role1 约在 34 µs 需要 plan，说明 auction 轮数已经落在关键路径上。
- EP6（6 卡）：ECO 比 EPLB 快 6–21 µs。说明机制本身是健康的，EP64 的开销来自规模。
- EP32（本 session 早期）：identity 固定开销约为 0。eager 下 eco512 比 eplb512 早约 21 µs，但 graph 回放下只剩 0–2 µs，即 ECO 的串行优势被跨 call 重叠吸收了。

## 5. 优化选项（数据状态与触发条件）

| # | 选项 | 现有依据 | 需要的数据（第 6 节） | 决策规则 |
|---|---|---|---|---|
| O1 | split 故障：修复，或 EP64 默认 split=0 | 4.2 的证据 | Phase 3 coredump | 修好之前性能测量一律用 split=0；修好后做 A/B 再定默认值 |
| O2 | ECO 的 block M 从 192 改为 128（或 96） | 旧 build 上 bm128 快 3.7 µs；planner 权重表里没有 192 | Phase 1 计时：eco vs eco_b128 | 快 ≥ 1 µs → 改 ECO 的 block M 选法，之后都用新 bm |
| O3 | planner 耗时（auction 轮数、kernel 本身、目标函数加开工下限） | 离线：Tp* 只占最慢 rank 时间的 31–41% | O_plan（Phase 1 trace）；离线 warm0/lag 质量（第 7 节） | O_plan < 0.5 µs/call → 不再优化 planner；≥ 0.5 µs → 见 6.5 |
| O4 | 固定机制开销（count 交换、copy-list 查找、副本索引写、role3 额外工作） | 旧值 13.6 µs，包含已去掉的 dispatch SM 损失 | O_pre / O_pull / O_combine（Phase 1 trace）；identity 交叉验证（Phase 3） | 单项 ≥ 1 µs → 按 6.1 表打对应项 |
| O5 | 放置与 plan 质量 | 离线（第 7 节）：P448 plan 距下界 1.2%，P512 距下界 7% | T_ora − 下界（Phase 1 标定后换算 µs） | ≥ 1 µs → 改进 plan（P512）或 placement（P448） |
| O6 | 跨 call 耦合 | EP32 上 eager 优势在 graph 下被吸收 | O_xcall（Phase 1 trace） | ≥ 1 µs → 查 call 之间的依赖 |
| O7 | role1 拉取 / 每 slot 机械开销、pull 流水线价值 | 旧估计 4–20 µs（P448→P640） | O_pull（Phase 1 trace）；条件 A/B PIPE=0/1 | ≥ 1 µs → 优化拉取 |
| O8 | plan 占 2 个 compute SM；helper、route-prep、rows-after-pub | EP32 上无收益 | O_sm（Phase 1）；helper（Phase 3） | O_sm ≥ 1 µs → helper 回收或换 plan 的位置 |
| O9 | 结构性：count 交换从 O(R²) 压到 O(slots) 等 | 无 | O_pre（Phase 1 trace） | 条件触发 |
| 上限 | ECO 到底能不能赢 EPLB | 离线模型：448 −7.6%，512 −14.5% | G = T_eplb − T_ora（Phase 1） | G < 去不掉的开销 → 如实报告：EP64 decode T=128 下 ECO 赢不了 EPLB |

## 6. 实测计划（目标：ECO 做到"几乎零额外开销"还差哪些）

### 6.1 拆分方法

- **零开销 ECO 的定义**：每个 rank 的负载和 ECO 逐 call plan 完全相同，但不跑任何 ECO 机制（count 交换、plan、副本 index 扇出、plan 等待、quota 读取、plan 占用的 2 SM）。
- **oracle 控制组**：`pkg/gcbench/ep64_oracle.py` 用 bit-exact 参考 planner（每层 warm 状态，两遍重放取第二遍）离线算出每个 call 的 quota，写成 plain 模式的物理槽 routing `ora<P>b<bm>.npy`。`--mode plain --num-experts P --routing @D/ep64/ora<P>b<bm>.npy --block-m <bm>` 的时间就是零开销 ECO 的时间 T_ora。
  - GPU 上 warm 状态的历史和离线重放不完全相同（bench 开头多跑了几遍 chunk 0），所以逐 call quota 不保证逐位相同，但统计上等价；gate 里用冷启动首 call 逐位核对 GPU planner 与参考 planner。
- **三个量**：
  - 机制开销 O = T_eco − T_ora：完美实现就是把 O 压到 0。
  - 均衡收益 G = T_eplb − T_ora：零开销时 ECO 比 EPLB 快多少。
  - plan 质量余量 = T_ora − T(下界)：离线模型下界，用 6.4 的标定换算成 µs。
- **O 按阶段拆**（同负载下 eco 对 oracle 的 trace 时间点差）：

| 分项 | 测法 | 对应的优化 |
|---|---|---|
| O_pre：rank barrier 之前（count 行、副本 index 扇出、copy-list 查找） | barrier_done 之差 | count 行压缩、扇出优化 |
| O_plan：等 plan | graph trace 的 role1/role2 plan stall 累计；eager trace 的首次等待时刻 vs plan_published | 加速 plan、目标函数加开工下限 |
| O_pull：role1 拉取（副本 peel、phy2log） | role1_end 之差减去 O_plan 中 role1 的部分 | peel / 查表优化 |
| O_sm：plan 占 2 个 compute SM | ora vs ora + `GC_MOE_DBG_COMPUTE_MINUS=2` | helper 回收、换 plan 的位置 |
| O_combine：role3（quota 读取、副本输出回传） | (role3_end − role2_end) 之差 | role3 副本路径 |
| O_xcall：跨 call 耦合 | (eco − ora) 在 graph 每 call 时间与 eager 端到端之间的差 | call 之间的依赖 |

- role2 本身：ECO 与 oracle 的 compute 负载相同，role2_end 之差应约等于 O_sm 加上 role2 的 plan 等待；明显更大说明副本任务的排布有额外代价。

### 6.2 离线分析（不占集群；第一轮已完成，见第 7 节）

`ep64_oracle.py <data> <out> <unit_ref.py> [--routing] [P:bm ...]`，每个 (P, bm)：
- plan 变体：full（默认轮数）、warm0（只做 warm 起点，不做 auction）、lag（上一步同层 plan 的比例套到本 call 计数，按 unit 对齐）、eplb、native。
- 下界：max(单副本负载最重的 rank, (单副本总量 + 每个副本专家只放一份) / R)。
- 副本 token 占比 x；plan 时间窗：接收副本的 rank r 最晚在 W_r = L − Q_r 开始副本工作（L 为最慢 rank 时间，Q_r 为它的副本工作），Tp* = min_r W_r。

### 6.3 集群 Phase 1（本轮，52 次 run，队列 `results/q_p1.json` → `results/p1.jsonl`）

所有 ECO 条目带 `GC_MOE_ECO_SPLIT_COUNTS=0`；全部用包 `gc-moe-r5c`（r5 + 默认关闭的 `GC_MOE_DBG_COMPUTE_MINUS`）。

1. **门槛**（失败即停队列）：
   - `plan448`、`plan512`：`--check --check-steps 1 --dump-plan`，GPU 冷启动首 call 的每 rank quota 负载必须与参考 planner 逐位一致。
   - `chk448`、`chk512`：`--check --check-steps 40`；`chk512_b128`：再加 `--block-m 128`。
2. **计时矩阵**（各 3 次，交错）：
   - `eplb448`、`eplb512`（gc-moe 默认 bm128）
   - `eco448`、`eco512`（默认 bm192）；`eco448_b128`、`eco512_b128`（`--block-m 128`）
   - `ora448_b128`、`ora512_b128`、`ora448_b192`、`ora512_b192`
   - `nat384` ×2（标定用）
3. **SM 预留**：`ora512_b128_cm2` ×3（`GC_MOE_DBG_COMPUTE_MINUS=2`）。
4. **trace**（bm128，各 1 次）：
   - graph trace（`GC_MOE_ECO_TRACE=1` + 计时参数）：eplb448、eplb512、eco448_b128、eco512_b128、ora448_b128、ora512_b128
   - eager trace（`--trace 16 --trace-dump`，dump 在各节点 `$W/runs/etr_*`）：同上 6 个
   - 若计时显示 ECO 在 bm192 下更快，Phase 2 补 bm192 的 trace。

### 6.4 Phase 1 的产出

- 每个 P：T_eplb、T_eco（两种 bm）、T_ora（两种 bm）→ G、O、block M 的选择。
- 标定：用 nat384、eplb448/512、ora×4 的 kernel_us 对离线模型最慢 rank 时间做线性拟合（同 bm 内），得到每单位多少 ns，把第 7 节的离线结论（plan 余量、Tp*）换算成 µs。
- O 的阶段拆分（6.1 的表）。

### 6.5 决策规则

| 测得 | 阈值 | 动作 |
|---|---|---|
| eco_b128 比 eco（bm192）快 | ≥ 1 µs | ECO 默认 block M 改为按物理槽选（或 EP64 固定 128） |
| O_plan | ≥ 0.5 µs/call | P448 先试 warm0（离线只差 1.6%）；P512 需要 auction（warm0 差 6.3%），改为加速 kernel 或给目标函数加开工下限 |
| O_sm | ≥ 1 µs | helper 回收（`GC_MOE_ECO_HELPER=1`）或把 plan 放到别的空闲资源 |
| O_pre / O_pull / O_combine | 各 ≥ 1 µs | 6.1 表里对应的机制优化 |
| O_xcall | ≥ 1 µs | 查 call 之间的依赖 |
| T_ora − T(下界) | ≥ 1 µs | plan 质量还有空间：P512 调轮数 / 改进 auction；P448 只能改 placement |
| G | < O 中去不掉的部分 | 如实报告：EP64 decode T=128 下 ECO 赢不了 EPLB |

### 6.6 Phase 2/3（按 Phase 1 结果触发）

- bm192 的 trace（若需要）。
- plan 时延探针：eco512 `GC_MOE_ECO_UNIT_CFG=293,52,0,6,0`（warm0）×3，加离线同配置生成的 oracle ×3。
- identity 交叉验证：eco448id vs nat384。
- helper：`GC_MOE_ECO_HELPER=1`（若 O_sm 明显）。
- split-count 根因（独立支线）：eco512 split 开 ×2，加 1 次 coredump，用 cuda-gdb 看首发 rank 的 kernel / PC / SASS。coredump 会影响性能，只用来诊断。env：
  - `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1`、`CUDA_COREDUMP_SHOW_PROGRESS=1`、`CUDA_COREDUMP_FILE=$W/cores/core_%h.%p.%t`
  - `CUDA_COREDUMP_GENERATION_FLAGS=skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory`

### 6.7 工具（已完成）

- `ep64_queue.py`：文件锁（同时只能有一个队列）；每个条目必须写 `pkg`，启动时校验该包在 16 节点逐文件 md5 一致，摘要写入每条结果；每次 run 前要求 64 卡空闲且没有 torchrun；失败自动分诊；`gate` 条目失败即停；记录起止时间。
- `ep64_triage.py <tag>`：首发 rank、错误类别（infra / kernel）、readiness 超时指向的 peer、各节点 torchrun "Root Cause" 时间、相关节点 Xid。
- `GC_MOE_DBG_COMPUTE_MINUS=<偶数>`（`gc_moe/jit_kernels/impls/sm100_fp8_fp4_gc_moe.py`，只对 plain 生效，默认 0）。

## 7. 结果记录

### 2026-11-08 离线 plan 质量（`~/work/mpo/ep64/results/oracle_local/oracle_stats.json`，1200 call）

模型时间 = wb·blocks + wa·active + tokens。bm192 没有拟合权重（借用 128 的 293/52），所以只在同一 bm 内比较；下表都是 bm128。

| 配置 | 最慢 rank | 时间不均衡 | token 不均衡 | block 峰值 | 活跃峰值 |
|---|---|---|---|---|---|
| eplb448 | 4805 | 1.169 | 1.222 | 12.00 | 7 |
| eco448 full | 4442（−7.6%） | 1.127 | 1.203 | 11.00 | 7 |
| eco448 warm0 | 4514 | 1.143 | 1.212 | 11.13 | 7 |
| eco448 lag | 4578 | 1.162 | 1.228 | 11.30 | 7 |
| 下界 448 | 4390 | | | | |
| eplb512 | 5007 | 1.162 | 1.241 | 12.48 | 8 |
| eco512 full | 4283（−14.5%） | 1.083 | 1.199 | 10.52 | 8 |
| eco512 warm0 | 4553 | 1.144 | 1.235 | 11.13 | 8 |
| eco512 lag | 4577 | 1.159 | 1.238 | 11.25 | 8 |
| 下界 512 | 4003 | | | | |

- P448：plan 距下界 1.2%。下界由单副本负载最重的 rank 决定（这部分不能挪），时间不均衡度的下界约 1.11，达不到 1.05；再提升要改 placement，不是 plan。
- P512：plan 距下界 7%（下界对应的不均衡约 1.01）；auction 轮数有用（warm0 差 6.3%）。
- 滞后 plan（lag）比 full 差 3.1%（448）/ 6.9%（512），不值得用质量换掉 plan 时延。
- token 不均衡度即使在最优 plan 下也约 1.2：目标函数优化的是 block 主导的时间（wb=293 个 token 当量），token 均衡不是 EP64 T=128 下的决定性指标。
- 副本 token 占比 x = 45.5%（448）/ 61.0%（512）→ 至少路由到一个副本专家的 token 占 97.4% / 99.6%，几乎所有 token 的 combine 都要等副本尾巴。
- plan 时间窗 Tp*（从 compute 开始算，最紧的接收 rank）：448 平均为 L 的 40.5%（p10 1557 单位），512 为 30.8%（p10 1030 单位）；接收副本的 rank 占 88% / 98%。换算成 µs 待 Phase 1 标定。

### 2026-11-08 Phase 1 启动（队列 `p1`，`results/p1.jsonl`，QUEUE_PORT 25500）

- 部署：`sgpkg/gc-moe-r5c`（impls 7c5786ab，jit impls fc6d61c07e73 = r5 + `GC_MOE_DBG_COMPUTE_MINUS`），`sgpkg/gcbench-ep64`（含 `ep64_oracle.py`）；16 节点 md5 一致。
- oracle routing 在 part0 生成（59 s），quota 与本地 x86 生成的逐位相同；`ora{448,512}b{128,192}.npy` 扇出到 16 节点，md5 一致。
- 门槛：`plan448`、`plan512` 的 GPU 冷启动首 call 每 rank quota 负载与参考 planner 逐位一致；`chk448`（40 步）bit-exact。
- 队列开销：run 结束后偶有 GPU 残留 1% util，post 检查判为不干净会多一轮 kill + 等待（结果不受影响）。

### 2026-11-08 Phase 1 中途结果（rep 0 完成；单次 run，待 3 次重复确认）

- 计量：部分 run 有少数异常 chunk 把均值拉高（mean−copy 与 p50−copy 相差最多 22 µs），p50−copy 稳定（eplb448 两次 88.93 / 88.52）→ 暂以 **p50−copy** 为准。
- p50−copy（µs）：eplb448 88.7 / eplb512 90.0；eco448 97.1、eco512 102.6（bm192）；eco448_b128 95.1、eco512_b128 101.3；ora448_b128 87.3、ora512_b128 87.0；ora448_b192 99.4、ora512_b192 90.4；ora512_b128_cm2 87.3；nat384 184.3。
- 拆分（bm128）：G = +1.4（448）/ +3.0（512）；O = +7.8 / +14.3；O_sm ≈ +0.3（可忽略）；bm192→128：ECO 快 1.4–1.9，oracle 快 3.4–12 → bm192 不该用。
- 标定（4 个点）：约 4 ns/单位；模型差 × 4 ns 与实测 G 吻合（1.45 vs 1.4，2.9 vs 3.0）。
- plan 在关键路径上：graph trace 中每 call 全部 67 个 role2 scheduler 都等 plan，平均 4.1 µs（448）/ 5.8 µs（512），最坏 rank 7.1 / 9.6；role1 192 个 warp 平均等 2.5 / 3.7 µs。
- planner 时间线（eager trace，rank 均值）：counts 到齐 → 静态表就绪 +3.1/3.2 → auction 轮 +10.4/10.2 → 发布 +1.4/1.9，共约 14.9 / 15.3 µs；role2 首次等待比发布早 2.9 / 6.4 µs。
- 初步推论：去掉 auction（warm0）的质量损失按标定只有约 0.3（448）/ 1.1 µs（512），而 role2 等待 4–6 µs → 便宜的 plan 很可能净赚，待 Phase 2 的 `GC_MOE_ECO_UNIT_CFG=293,52,0,6,0` 探针确认。O 中剩余约 3–8 µs 待按 trace 归因。

### 2026-11-08 Phase 1 暂停（用户要求）：oracle vs EPLB

- 已完成 38/52 次（rep 0 全部 + rep 1 的 9 个计时配置）；未跑：ora512_b192_r1、nat384_r1、cm2_r1 与整个 rep 2（共 14 次）。集群已清空。
- p50−copy（各 2 次，spread ≤ 0.6 µs）：eplb448 88.72 / ora448_b128 87.33；eplb512 89.99 / ora512_b128 87.31；eco448_b128 95.39；eco512_b128 101.27。
- 零开销收益 G：448 为 1.40 µs（1.6%），512 为 2.68 µs（3.0%）。
- 标定（bm128，4 点，残差 ±0.3 µs）：T ≈ 70.9 µs + 3.77 ns × 最慢 rank 模型时间。模型预测 G = 1.37 / 2.73 µs，与实测一致：plan 的均衡改善被完整兑现。
- 收益小的原因：随均衡变化的只有最慢 rank 的计算（EPLB 约 18–19 µs）；plan 把它降 7.6% / 14.5%；完美均衡也只再多 0.19 / 1.06 µs（G 上限 1.57 / 3.79 µs）。
- 对比机制开销 O（bm128）：8.06 / 13.96 µs → ECO 净慢 6.7 / 11.3 µs。要打平需把 O 压到 G 以下（砍掉 >80%）。

### 2026-11-08 离线：批大小缩放（同层 k 个 call 合并成一个，T = 128·k；bm128；按 3.77 ns/单位换算）

| P512 每卡 T | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| 时间不均衡 EPLB / ECO | 1.16 / 1.08 | 1.19 / 1.06 | 1.21 / 1.05 | 1.22 / 1.04 |
| 每卡 block 数 | 11 | 16 | 28 | 52 |
| 零开销收益 G（只算计算） | 2.7 µs | 4.6 µs | 8.2 µs | 15.7 µs |
| 收 token 峰值（dispatch 量）ECO 相对 EPLB | −3.4% | −8.9% | −12.9% | −14.7% |

- P448：G 为 1.4 / 2.3 / 4.2 / 8.8 µs；收 token 峰值 −1.6% / −6.4% / −8.4% / −9.1%。EPLB 的收 token 不均衡在所有 T 下都约 1.21–1.24（静态 placement 与流量不匹配）。
- 局限：G 是 EPLB 与 oracle 的计算差，dispatch/combine 的公共增长在差值中抵消，但没有算进：总时间增大后的相对稀释；dispatch 超过计算成为瓶颈后的情况（大 T 时 ECO 对收 token 峰值也有改善，T=128 时几乎没有）；ECO 每 token 的机制开销（副本 index 扇出、copy-list 查找、peel）随 T 增长；标定只在 T=128 拟合。要用 T=512/1024 的实测确认。

### 2026-11-08 批大小实测（`results/pT.jsonl`；同层 k 个 call 合并成每卡 T = 128·k；bm128；各 2 次，p50−copy 中位数，µs）

所有门槛（T=512/1024 的 chk448、chk512，8 步）均 bit-exact。

| T | P | EPLB | oracle | ECO | G = EPLB−oracle | O = ECO−oracle | ECO − EPLB |
|---|---|---|---|---|---|---|---|
| 128 | 448 | 88.72 | 87.33 | 95.39 | +1.40（1.6%） | +8.06 | +6.67 |
| 128 | 512 | 89.99 | 87.31 | 101.27 | +2.68（3.0%） | +13.97 | +11.28 |
| 512 | 448 | 199.19 | 184.78 | 189.56 | +14.41（7.2%） | +4.78 | −9.63 |
| 512 | 512 | 198.55 | 180.29 | 190.61 | +18.26（9.2%） | +10.31 | −7.94 |
| 1024 | 448 | 341.75 | 316.12 | 323.29 | +25.63（7.5%） | +7.17 | −18.46 |
| 1024 | 512 | 348.99 | 310.23 | 317.89 | +38.76（11.1%） | +7.66 | −31.10 |

- EPLB 用 gc-moe 默认 block M（T≥512 时为 240）：222.32（T512）、367.27（T1024），比 bm128 慢 18–24 µs → 基线一律用 bm128。
- eplb448_T512 两次相差约 8 µs（195.3 / 203.1），其余配置两次相差 ≤ 1 µs。
- 按 ECO − EPLB 线性插值，打平点约在每卡 T ≈ 290（P448）/ 350（P512），未实测。

eager trace（P512，barrier 之后，rank 均值，µs）：

| T | dispatch EPLB / oracle / ECO | 计算结束 | combine 尾巴 | plan（计数到发布） |
|---|---|---|---|---|
| 128 | 25.7 / 25.6 / 30.4 | 48.2 / 45.9 / 51.6 | 19.3 / 18.4 / 24.0 | 15.3（role2 早等 6.4） |
| 512 | 107.0 / 104.9 / 109.6 | 131.7 / 128.6 / 133.5 | 42.7 / 25.7 / 29.0 | 19.5（被盖住） |
| 1024 | 211.2 / 211.3 / 216.3 | 240.9 / 240.3 / 246.2 | 68.3 / 39.4 / 41.5 | 23.3（被盖住） |

- 瓶颈转到 dispatch：dispatch 占计算窗口 53% → 81% → 88%，计算只比 dispatch 晚结束约 22 / 25 / 30 µs。
- 均衡收益集中在 combine 尾巴（等最慢 rank）：EPLB 19 → 43 → 68 µs，oracle 18 → 26 → 39，ECO 24 → 29 → 42。
- 实测 G 是只算计算的离线模型的 2.2–3 倍（T512：14.4 / 18.3 vs 4.2 / 8.2；T1024：25.6 / 38.8 vs 8.8 / 15.7）：均衡同时减少了最忙 rank 的 dispatch 量和输出回写。
- 大 T 下剩余开销 O 主要在 dispatch 侧：ECO 的 dispatch 比 oracle 晚约 5 µs，三档 T 下基本不变（副本 token 要等单副本拉完、plan 就绪后才拉，另有每 token 的查表 / peel）；plan 本身在 T≥512 时已被完全盖住。
- 下一步候选：planner 代价模型加入收 token（dispatch）项并在 T=512/1024 重新拟合；role1 副本拉取提前 / 交错；单独检查 gc-moe 默认 block M（240）在 EP64 上是否普遍不如 128；T=128 decode 按 T 自适应开关 ECO。

## 8. 四个方向（2026-11-08 夜间，自主执行）

目标：完成下面 4 个方向并给出全部结果。代码改动只进本树，打成新包 `gc-moe-r6`；`gc-moe-r5c` 保持不动作为基线。

| 方向 | 内容 | 做法 |
|---|---|---|
| D1 | planner 代价模型加入收 token（dispatch）项，在大 T 下重新拟合 | 离线扫 (wb, wa) 缩放系数 s（等价于加大 token 权重）→ 选候选；用 `GC_MOE_ECO_UNIT_CFG` 在线测 + 对应 oracle；最后在 `eco_unit_config` 里实现随 T 变化的权重 |
| D2 | role1 副本拉取优化，砍掉 dispatch 侧约 5 µs | 先诊断（trace 中 role1 的 plan 等待、identity 对照 eco448id vs nat384 @T512），再按诊断改 role1 |
| D3 | 补测 T=256 坐实打平点，实现按 T 自动开关 ECO | T=256 的 eplb/oracle/eco；静态切分回退（ECO placement + EPLB 式切分）性能；API 层按每卡 token 数选择 ECO 或静态切分 |
| D4 | gc-moe 默认 block M（大 T 选 240）问题 | EPLB512 在各 T 下扫 bm；ECO 默认 bm 对照；按实测修正选择规则（ECO 按物理槽数选） |

代做清单（顺序执行，结果逐项追加到本节）：
1. 准备：T=256 routing 与 oracle（merge=2），part0 生成后扇出。
2. D1 离线：权重缩放扫描（T=128/256/512/1024）→ 候选 + 对应 oracle。
3. D3 离线：静态切分回退的 routing 生成。
4. D2 诊断：从已有 trace 读 role1 的 plan 等待。
5. 集群队列 Q1：D3 的 T=256 三组 + trace；D4 的 bm 扫描；D1 的权重候选（eco + oracle）；D2 的 identity @T512；D3 的静态切分。
6. 实现 D4、D1、D3、D2 的代码改动 → `gc-moe-r6`。
7. 验证：门槛 + 最终矩阵（T=128/256/512/1024，EPLB vs ECO-r6）。
8. 汇总 4 个方向的全部结果。

### 8.1 进度（Q1）

- D2 诊断（已有 eager trace，barrier 后、rank 均值）：ECO 的 role1 比 oracle 晚结束 4.2–4.8 / 4.3–4.7 / 5.0 µs（T=128/512/1024），三档基本不变。T=128 时 role1 首个 warp 在 plan 发布前 0.5–2.9 µs 就开始等；T≥512 时 role1 首次需要 plan 在发布后 16–64 µs，根本不等。→ 大 T 下这约 5 µs 不是等 plan，而是 ECO 拉取路径自身的固定开销（每个 warp 进入副本专家时读 64 源计数的 volatile load 等是嫌疑），需 identity 对照确认。
- D1 离线（`results/d1_screen.json`）：权重缩放 s（wb·s, wa·s）在 token 峰值与 block 峰值之间取舍。P512：T=128 时 s=1 → token −3.4% / block −15.7%，s=0.25 → −7.7% / −12.8%，s=0 → −11.0% / −9.3%；T=256 时 s=0.25 → −11.6% / −14.1%（s=1：−8.9% / −16.4%）；T≥512 时各 s 相差 ≤ 1.5%。P448 各 T 下相差都很小。→ 在线测 s=0.25（`GC_MOE_ECO_UNIT_CFG=73,13,<nr>,6,2`）@T=128/256/512。
- D3 离线：ECO placement 上的静态切分（sta）与 EPLB 相当（T=256 P512：时间峰值 8007 vs EPLB 8043）。
- Q1（`results/q_q1.json` → `results/q1.jsonl`，59 次，QUEUE_PORT 25800）：T=256 的 eplb/eco/oracle（×2）+ trace；静态切分 sta @T=128/256（×2）；D1 的 s=0.25 eco + oracle（×2）；identity id448 vs nat384 @T512（×2）+ trace；EPLB512 bm 扫描（T=128：64/96/192；T=256：96/192/240；T=512：96/192；T=1024：192）；ECO 默认 bm @T=256/512/1024。
- 代码（本树，未提交）：D2 候选 = `_eco_table_count` 非 spec 路径改为 L1 可缓存的 `ld_global_u32`（新 helper，`ptx/ld_st.py`）；D3 = `apis/gc_moe.py` 中 `GC_MOE_ECO_MIN_TOKENS`（默认 0 不启用）：每卡 token 数低于阈值时按 `rank mod copies` 静态映射（`_eco_static_map`，从 EcoMeta copies 表构建，首次调用须 eager）后走 plain 路径。离线：静态切分 sta（按位置轮转）与 source-affinity stb（整源固定副本）都与 EPLB 相当（P448 时间峰值 4807 / 4812 vs 4805；P512 4969 / 5039 vs 5007）。
- 包 `gc-moe-r6a` = r5c + D2 + D3（impls fef32129d836）；D2 的 A/B = r6a vs r5c。
- 运行状态（夜间）：Q1（`results/q1.jsonl`，59 次，r5c）进行中；Q2（`results/q_q2.json` → `results/q2.jsonl`，27 次，QUEUE_PORT 25900）由 `/tmp/chain_q2.sh` 在 Q1 结束后自动启动：r6a 门槛（chk512 40 步、chk512_T512、auto512 check）→ D2 A/B（r5c vs r6a 的 eco448/512 @T=128/512，各 2 次交错）→ D3 auto448/512（r6a，`GC_MOE_ECO_MIN_TOKENS=100000`）vs eplb448/512（r5c）各 2 次。汇总脚本：`~/work/mpo/ep64/q1_summary.py`（读 p1/pT/q1/q2）。
- 待办：Q1 出数后定 D4 规则（ECO 改按物理槽选 bm；按扫描结果给 L20B/EP64 加上限或按 SM 并行度选 bm）与 D1 规则（`eco_unit_config` 按每卡 token 数缩放 wb/wa）→ 打包 `gc-moe-r6`（= r6a + D1 + D4）→ Q3 最终矩阵：T ∈ {128,256,512,1024} × P ∈ {448,512}，ECO-r6 默认设置 vs EPLB，各 2 次 + 门槛。
- Q1 中途（T=256，第 1 次）：P448 eplb 123.98 / oracle 119.86 / eco 124.81（G +4.1，O +5.0，ECO−EPLB +0.8，≈打平）；P512 125.32 / 118.10 / 130.30（G +7.2，O +12.2，ECO−EPLB +5.0）。
- Q1 中途（第 1 次）：静态切分 sta 在 T=128 与 EPLB 相当（448：88.82 vs 88.72；512：89.46 vs 89.99），T=256 略快于 EPLB（−1.8 / −2.1 µs），比 ECO 快 2.6 / 7.1 µs → 回退方案可用。D1 s=0.25：T=256 的 ECO −0.05 / −0.60 µs，T=512 的 ECO −0.51 µs（均在噪声内，待第 2 次）。
- Q1 中途 D1（s=0.25，第 1 次）：T=128 P512 ECO 101.27→100.48（−0.8），oracle 87.31→86.58（−0.7）；T=256 ECO −0.05（448）/ −0.60（512），oracle −1.14 / −0.50；T=512 P512 ECO −0.51，oracle +3.86（单次，疑似噪声）。
- Q1：`id448_T512_r0` 失败，node0 日志为 `DistStoreError: wait timeout after 600000ms`（CUDASymmetricMemory rendezvous，有 rank 没到场）→ 属基础设施类，等分诊结果与 rep 1 重试；队列继续。
- 分诊结果：`id448_T512_r0` 的首发是 a0003（node 3）上 rank 12–15 `FileNotFoundError: eco448id_metas.npy`（该节点缺 identity metas），其余 rank 10 分钟后 rendezvous 超时。已从 part0 把 `eco448id_metas{,_phy2log}.npy` 扇出到全部节点并核对 md5（3a9b457b81cd / bee99d37d4e7）。rep 1 与 identity trace 会正常重跑。教训：部署后要对所有用到的数据文件做 16 节点存在性 + md5 检查（`ep64_triage.py` 的 INFRA 列表应加上 `FileNotFoundError`、`DistStoreError`）。
- Q1 中途 D4（EPLB512，p50−copy，单次）：T=128：bm64 98.10 / bm96 **88.87** / bm128 89.99 / bm192 92.14；T=256：bm96 **123.15** / bm128 125.32 / bm192 126.46 / bm240 150.38；T=512：bm128 198.55 / bm240 222.32；T=1024：bm128 348.99 / bm240 367.27（其余待测）。gc-moe 启发式在 T=128/256 选 128、T≥512 选 240，都不是最优；bm96 在 T≤256 最好。
- Q1 D4 扫描补全（EPLB512，单次）：T=512：bm96 **194.99** / bm128 198.55 / bm192 200.38 / bm240 222.32；T=1024：bm128 348.99 / bm192 **341.29** / bm240 367.27（bm96 未测）。→ 最优 bm：T≤512 为 96，T=1024 为 192；启发式（128 / 128 / 240 / 240）分别多花 1.1 / 2.2 / 27.3 / 26.0 µs。
- D4 代码（本树）：`apis/gc_moe.py` 中 ECO 改按物理槽数选 block M（不再按逻辑专家数）；`jit_kernels/heuristics/gc_moe.py` 的 `get_block_m_for_gc_moe` 增加实测覆盖表 `GC_MOE_BLOCK_M_TABLE='<每槽期望 token 上限>:<bm>,...'`（EP64 L20B 建议 `400:96,100000:192`，未设时行为不变）。
- D1 代码（本树）：`eco/plan.py` 的 `eco_unit_config` 增加 `tokens` 参数与可选 `GC_MOE_ECO_WSCALE='<每卡最少 token>:<s>'`（达到阈值时 wb、wa 乘 s；默认不启用），调用处传 `config.tokens`。是否默认开启取决于 Q1 rep 1 的 s=0.25 结果（rep 0：ECO 仅 −0.05 ~ −0.8 µs）。
- 下一步（Q1、Q2 完成后）：打包 `gc-moe-r6` = 当前树（D2 缓存读 + D3 自动开关 + D4 物理槽选 bm/覆盖表 + D1 可选缩放）→ 部署并 16 节点 md5 校验 → Q3 最终矩阵：T ∈ {128,256,512,1024} × P ∈ {448,512}，每格三组各 2 次：`eplbN`（r6，`GC_MOE_BLOCK_M_TABLE=400:96,100000:192`）、`ecoN`（r6，同表 + split=0）、`ecoA`（r6，同表 + split=0 + `GC_MOE_ECO_MIN_TOKENS=<打平点>`）；门槛：r6 下 chk512（T=128，40 步）、chk512_T512、auto 的 check、bm96 下的 ECO check。
- 包 `gc-moe-r6`（已部署 16 节点）= 当前树：impls fef32129d836、apis 738d59fad016、plan c28e1b088df2、heuristics 2ab4d3322f8c。
- Q3（`results/q_q3.json` → `results/q3.jsonl`，50 次，QUEUE_PORT 26000，全部 r6 + `GC_MOE_BLOCK_M_TABLE=400:96,100000:192`）由 `/tmp/chain_q3.sh` 在 Q1、Q2 结束后自动启动：门槛 g_chk512_T128（40 步）/ g_chk512_T512 / g_chk448_T1024 / g_auto512_T128 → 每个 T ∈ {128,256,512,1024}、P ∈ {448,512} 的 `eplbN`（EPLB + 新 bm 表）与 `ecoN`（ECO r6 + 新 bm 表）各 2 次；`ecoW512`（+ `GC_MOE_ECO_WSCALE=256:0.25`）@T=256/512/1024；`ecoA`（+ `GC_MOE_ECO_MIN_TOKENS=300`，即 T<300 走静态回退）@T=128/256。
- 汇总：Q1 / Q2 / Q3 全部完成后，用 `q1_summary.py`（Q1、Q2）与 Q3 汇总，把 4 个方向的结论写进本节 8.2。

### 8.2 Q1 结果（r5c，p50−copy 中位数，µs；多数 2 次）

- D3 打平点：T=256 时 P448 ECO−EPLB = +0.81（≈打平），P512 +4.98；T=512 时 −9.63 / −7.94 → 打平点约 T≈270（P448）/ ≈350（P512）。回退阈值取 `GC_MOE_ECO_MIN_TOKENS=300`。
- D3 静态回退（ECO placement + 按位置轮转，plain 路径）：T=128：88.88 / 89.50（EPLB 88.72 / 89.99，持平）；T=256：122.38 / 123.13（比 EPLB 快 1.7 / 2.4，比 ECO 快 2.5 / 7.4）。
- D1（s=0.25）：ECO 在 T=128/256/512 均略快 0.2–0.8 µs（P512 T128 −0.73，T256 −0.23 / −0.82，T512 −0.25）；oracle 同方向（T512 一次异常 +3.86）→ 收益小但方向一致。
- D4：EPLB512 最优 bm：T≤512 为 96（比 128 快 1.1 / 2.4 / 3.6），T=1024 为 192（比 128 快 7.7）；gc-moe 启发式的 240 在 T=256/512/1024 慢 24.8 / 23.8 / 18.3（相对 128）。ECO 现行默认（按逻辑专家选 → 192/192/240）比 bm128 慢 +2.4 / +4.2 / +18.4 → 证实应改为按物理槽 + 实测表。
- D2：identity（eco448id，副本全为同 rank 被动副本）@T512 的 eager trace：dispatch 103.6 vs native 104.9，计算结束 127.8 vs 129.0 → 没有真实跨 rank 副本时 ECO 路径不比 native 慢；真实 ECO 的约 5 µs dispatch 开销来自跨 rank 副本拉取本身（T=256：ECO 55.6 vs oracle 51.8）。
- D2 identity 计时 @T512（bm128，各 1 次有效）：见上行 id448 − nat384（ECO 机制在无跨 rank 副本、负载 = native 时的固定开销）。
== D2: identity @T512 (bm128): id448 - nat384 =   +8.00 |  556.79  548.79
- Q2 中途：r6a 三个门槛全过（chk512 40 步、chk512_T512、auto512 的 check 均 bit-exact）。D2 A/B（第 1 次，r5c → r6a）：eco448 T128 96.50 → 94.45（−2.05），eco512 T128 101.12 → 101.02（−0.10），eco448 T512 189.75 → 188.88（−0.87）。
- Q2 最终（p50−copy 中位数，各 2 次）：
  - D2 eco448_T128: r5c 95.82 -> r6a 94.31 (-1.51) n=2
  - D2 eco512_T128: r5c 101.09 -> r6a 101.00 (-0.09) n=2
  - D2 eco448_T512: r5c 190.30 -> r6a 188.78 (-1.52) n=2
  - D2 eco512_T512: r5c 190.71 -> r6a 192.32 (+1.61) n=2
  - D3 auto448 T128 94.21 vs eplb 88.82 (+5.39)
  - D3 auto512 T128 94.69 vs eplb 89.85 (+4.84)
- 解读：D2（副本计数改为可缓存读）没有稳定收益（−1.5 ~ +1.6 µs，噪声量级）→ 约 5 µs 的 dispatch 开销不在这条读路径上；identity 对照也显示无跨 rank 副本时没有 dispatch 延迟 → 开销来自真实副本拉取（副本 token 在单副本全部拉完后才开始拉，且要跨 rank peel），需要改拉取顺序 / 交错才能动它（未实现）。
- D3 API 回退在 T=128 比 EPLB 慢 4.8–5.4 µs，但离线 sta/stb 与 EPLB 持平、Q1 实测 sta 与 EPLB 持平 → 差额基本是 API 里 torch gather + copy 映射（graph 内 2–3 个小 kernel）的代价；它仍比 ECO 快 0.1（448）/ 6.3 µs（512）。要拿满收益需把映射并进上游 gating/topk kernel 或 role1 路由循环。
- Q3 汇总脚本：`~/work/mpo/ep64/q3_summary.py`（r6 新 bm 表下的 eplbN / ecoN / ecoW / ecoA，对比 r5c bm128 的 EPLB）。
- Q3 中途（rep 0，r6 + bm 表，单次，p50−copy）：T=128：eplbN 87.67 / 89.20（bm 表让 EPLB 快 1.1 / 0.8），ecoN 93.91 / 100.98，ecoA（回退）91.91 / 93.14；T=256：eplbN 122.58 / 123.29，ecoN 121.93 / 126.41（P448 ECO 已快 0.65），ecoW512 125.91，ecoA 126.61 / 127.37（T=256 回退反而更慢 → 阈值应降到约 200）；T=512：eplbN 202.04 / 195.30，ecoN 192.31 / 184.29（−9.7 / −11.0），ecoW512 207.36（单次异常）；T=1024：eplbN 335.69 / 342.06，ecoN 330.36 / 315.38（−5.3 / −26.7），ecoW512 314.65。

### 8.3 最终结果（Q3：r6 + `GC_MOE_BLOCK_M_TABLE=400:96,100000:192`，p50−copy 中位数，各 2 次，µs）

| T | P | EPLB bm128（r5c） | EPLB 新 bm 表 | ECO-r6 | ECO+WSCALE | 回退（MIN_TOKENS=300） | ECO−EPLB新 |
|---|---|---|---|---|---|---|---|
| 128 | 448 | 88.72 | 87.83 | 93.81 | — | 92.09 | +5.98 |
| 128 | 512 | 89.99 | 89.04 | 100.60 | — | 92.95 | +11.56 |
| 256 | 448 | 124.04 | 122.90 | 121.83 | — | 126.69 | −1.07 |
| 256 | 512 | 125.56 | 123.26 | 126.53 | 126.03 | 127.37 | +3.27 |
| 512 | 448 | 199.19 | 199.75 | 192.04 | — | — | −7.70 |
| 512 | 512 | 198.55 | 195.57 | 184.19 | 195.44 | — | −11.38 |
| 1024 | 448 | 341.75 | 336.11 | 325.38 | — | — | −10.73 |
| 1024 | 512 | 348.99 | 341.60 | 315.44 | 314.73 | — | −26.17 |

四个方向结论：
- D4（block M）：实测表让 EPLB 比 bm128 快 0.9–7.4 µs（T=512 P448 持平），比 gc-moe 默认 bm240 快 18–26 µs；ECO 改按物理槽选 bm 后，T=512 P512 从 190.61 降到 184.19（−6.4）。建议 EP64 L20B 默认启用该表。
- D1（dispatch 加权）：离线只在 T≤256 有取舍空间；在线 s=0.25 仅 −0.2 ~ −0.8 µs（r5c）；r6 下 T=256/1024 −0.5 / −0.7，T=512 +11.3（变差）→ 不默认开启，保留 `GC_MOE_ECO_WSCALE` 开关。
- D2（role1 副本拉取）：副本计数改可缓存读无稳定收益（−1.5 ~ +1.6）；identity 表明约 5 µs 来自真实跨 rank 副本拉取（单副本拉完后才拉副本 + peel），需改拉取顺序/交错才可能消掉（未实现）；identity 下 ECO 固定开销 @T512 为 +8.0 µs。
- D3（打平点与自动开关）：r5c bm128 下打平点 T≈270（P448）/ ≈350（P512）；r6 下 P448 在 T=256 已快 1.07，P512 约 T≈300。纯静态切分与 EPLB 持平，但 API 回退（torch gather 映射在 graph 内）多约 4–5 µs：T=128 比 ECO 快 1.7 / 7.6，仍比 EPLB 慢 4.3 / 3.9；T=256 回退反而比 ECO 慢 → 阈值应降到约 200，且映射应并入上游 gating 或 role1 路由。

### 8.4 匹配 bm 的零开销上界（oracle，Q4）

- 新 bm 表下各配置实际用的 block M（每槽期望 token = T·64·6/P）：T=128 / 256 都是 96；T=512 时 P448 为 192（期望 439 > 400）、P512 为 96；T=1024 都是 192。
- 已在 part0 生成对应 oracle routing（plan 的 unit 与该 bm 一致）：`ora448b96`、`ora512b96`、`ora448b96_T256`、`ora512b96_T256`、`ora448b192_T512`、`ora512b96_T512`、`ora448b192_T1024`、`ora512b192_T1024`（md5 依次 5064bcb8468d 3d5fec45ac0f 189e50108cda 609c5ef83090 69435244a649 1be677465169 d24d8291e929 93d8f71a2ecb）。
- Q4（`results/q_q4.json` → `results/q4.jsonl`，16 次，r6 + 同一 bm 表，plain 模式）：`oraN<P>_T<T>` 各 2 次。上界 = oraN；ECO 剩余开销 = ECO-r6 − oraN；零开销收益 = EPLB新 − oraN。
- Q4 结果（16/16 通过；r6 + bm 表；p50−copy 中位数，各 2 次；上界 spread ≤ 0.5 µs，只有 oraN512_T512 为 2.58）：

| T | P | bm | EPLB新 | ECO-r6 | 上界 oraN | ECO−EPLB | G = EPLB−上界 | O = ECO−上界 | 已兑现 (EPLB−ECO)/G |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 448 | 96 | 87.83 | 93.81 | 85.72 | +5.98 | +2.11 | +8.09 | <0 |
| 128 | 512 | 96 | 89.04 | 100.60 | 84.17 | +11.56 | +4.87 | +16.43 | <0 |
| 256 | 448 | 96 | 122.90 | 121.83 | 117.32 | −1.07 | +5.58 | +4.51 | 19% |
| 256 | 512 | 96 | 123.26 | 126.53 | 114.90 | +3.27 | +8.36 | +11.63 | <0 |
| 512 | 448 | 192 | 199.75 | 192.04 | 187.48 | −7.70 | +12.27 | +4.56 | 63% |
| 512 | 512 | 96 | 195.57 | 184.19 | 177.72 | −11.38 | +17.85 | +6.47 | 64% |
| 1024 | 448 | 192 | 336.11 | 325.38 | 316.24 | −10.73 | +19.87 | +9.14 | 54% |
| 1024 | 512 | 192 | 341.60 | 315.44 | 309.35 | −26.17 | +32.25 | +6.09 | 81% |

- O 的组成（eager trace，barrier 之后，bm128/r5c，与上表 bm 不完全匹配，只作近似）：dispatch（role1）+3.8 ~ +5.0（各 T 基本不变；T=128 时含 role1 等 plan）；role1 结束后的计算 −0.4 ~ +0.9（≈0）；combine 尾巴 +3.6 / +5.6（T=128）、+4.9（T=256 P512）、+0.9 / +3.3（T=512）、+2.0 / +2.1（T=1024）。eager trace 里 role1_start 含 host 发射抖动，barrier 前与端到端差值不可用。
- 要赢 EPLB 需 O < G：T=128 需砍掉 ≥ 6.0（P448）/ 11.6 µs（P512）；T=256 P512 需砍掉 ≥ 3.3 µs。

### 8.5 订正：dispatch 顺序与 combine 尾巴（逐 rank trace 分析，2026-11-09）

数据：14 组 eager trace 的逐 rank dump（`$W/runs/etr*_rank<r>.npy`，本地 `/tmp/etr`），bm128 / r5c，每 rank 16 个 call，barrier 后时间取中位数。脚本 `/tmp/etr_skew.py`、`/tmp/etr_rank.py`。

- dispatch 模型：role1 是拉取式。每 rank 16 个 dispatch SM × 12 warp（bm ≥ 96）= 192 个 warp，在本 rank 全部接收 token 的拼接列表上按 warp 跨步（token w, w+192, …），每个 warp 双缓冲，同时最多约 384 个 token 在途。ECO 只是把副本槽排在单副本槽之后，同一条并行拉取流；plan 及时到达时，这个顺序本身不增加时间。之前"副本 token 要等单副本拉完才拉"作为开销原因的说法不准确。
- combine 不查副本：专家侧在 FC2 epilogue 把输出直接写进源 rank 的 combine buffer（[topk 槽][token]）；`_role3_combine` 只读本地内存，plain 与 ECO 代码相同。ECO 专有的只有 role3 清理部分的 `_eco_slot_total`，它在 warp 0–3 上、发出 combine_ready 之后运行，不阻塞 reduce。实测"最慢 rank 的 role2 结束 → role3_end"ECO 与 oracle 相同（11.2–20.2 vs 11.9–20.6 µs）。"combine 尾巴"变长完全来自 role2 结束时间的跨 rank 偏差（max − mean）：T=128 P512 15.2 vs 5.4，T=256 P512 12.9 vs 7.4；T ≥ 512 基本相同。
- 偏差来源：副本多的 rank 等 plan。按 rank 上有 token 的副本槽数分组（T=128 P512，0 / 1 / 2 / ≥3 个）：role1 结束延迟 +1.5 / +2.7 / +5.1 / +6.6 µs，其中 role1 等 plan 0.1 / 1.5 / 3.7 / 4.8 µs；role2 结束延迟 +0.7 / +2.7 / +5.2 / +9.1 µs。ECO 把副本工作分给单副本负载轻的 rank，它们单副本拉得早、先撞上 plan 未就绪。≥3 副本槽 rank 的 plan 等待：T=128 4.5 / 4.8，T=256 2.9，T=512 5.1（P448）/ 1.7，T=1024 0.7 / 0.3。
- 扣掉等 plan 后还剩一块与副本槽数无关、随 T 增长的均匀延迟：+1.2 ~ 2.1（T=128）→ +4 ~ 6 µs（T=1024）。嫌疑：plan 占用的 2 个 compute SM（role2 用 134 个，ring 回压把 dispatch 结束和计算进度绑在一起）。待验证：T=1024 跑 oracle + `GC_MOE_DBG_COMPUTE_MINUS=2`。

### 8.6 投机拉取（按上一 step 同层 plan 先拉，本次 plan 后补拉）与均衡指标表（2026-11-09，离线）

- 方法：`~/work/mpo/ep64/supp_pull.py`。参考 planner（与 GPU planner 逐位一致）的逐 call plan（bm128，warm 链第二遍），call i 用 call i−8（同层上一 step）的 plan 先拉；映射到本次 token 序列（源 rank 优先的固定顺序）有两种：绝对位置（截到 N）或按比例缩放。补拉 = 本次 plan 分给自己但没拉到的 token；浪费 = 拉了但本次 plan 分给别人的 token。
- 结果（补拉 / 浪费占副本 token 的比例，按比例映射；绝对映射约多 1.5–2 个百分点）：

| T | P | 副本 token 占全部 | 补拉 | 浪费 | 补拉占全部 token | 最重 rank 的补拉 | 现在最重 rank 在 plan 之后要拉的副本 token |
|---|---|---|---|---|---|---|---|
| 128 | 448 | 45.5% | 8.2% | 8.2% | 3.7% | 170 tok（该 rank 的 20.9%） | 719 tok（82.5%） |
| 128 | 512 | 61.0% | 9.1% | 9.1% | 5.6% | 214 tok（26.2%） | 782 tok（87.7%） |
| 256 | 448 | 45.5% | 5.8% | 5.8% | 2.7% | 215 tok（13.5%） | 1363 tok（83.1%） |
| 256 | 512 | 61.0% | 7.4% | 7.4% | 4.5% | 285 tok（17.9%） | 1479 tok（87.9%） |
| 512 | 448 | 45.5% | 3.8% | 3.8% | 1.7% | 261 tok（8.3%） | 2647 tok（83.4%） |
| 512 | 512 | 61.0% | 5.6% | 5.6% | 3.4% | 371 tok（11.8%） | 2854 tok（88.6%） |
| 1024 | 448 | 45.5% | 2.1% | 2.1% | 1.0% | 303 tok（4.9%） | 5212 tok（83.4%） |
| 1024 | 512 | 61.0% | 3.8% | 3.8% | 2.3% | 467 tok（7.5%） | 5618 tok（88.8%） |

- 均衡指标表：`~/work/mpo/ep64/balance_table.py` → `results/balance_table.json`（EPLB 的槽负载按 T=128 routing 合并；ECO 为 oracle quota；指标逐 call 在 64 rank 上取 max/mean 或 max，再对 call 取均值）。激活槽峰值在所有设定下 EPLB 与 ECO 都等于每卡槽数（7 / 8），只有均值不同。

### 8.7 成本函数回归（2026-11-09，`~/work/mpo/ep64/cost_fit.py` → `results/cost_fit.out`；稳健性 `/tmp/cost_robust.py`）

- 数据：49 组 plain 模式实测（无 ECO 机制：EPLB / oracle / 静态切分 / s=0.25 oracle；T=128–1024；bm 96/128/192；p50−copy 多次中位数）。特征：每个 call 每个 rank 的 token 数、block 数（按该 run 的 bm）、激活槽数。
- 模型：y = c_T + mean_call max_r(wb(bm)·blocks + wa·active + wt·tokens)，wb(bm) = b0 + b1·bm；拟合方法为固定关键 rank 后做最小二乘、迭代，多起点。
- 结果：b0 = 227 ns/block，b1 = 12.9 ns/（block·行）→ wb = 1468 / 1881 / 2709 ns（bm96/128/192）；wa = 0；wt = 14.1 ns/token；换成 token 单位：wb = 104 / 134 / 192，wa = 0（现行 planner：335 @96，293 @128/192；wa 38 / 52）。截距 c_T = 54.3 / 61.9 / 80.8 / 117.7 µs。RMSE 1.90 µs，最大残差 5.1，留一法 RMSE 2.44。
- 可信度低：bootstrap 90% 区间 wb128 819–3101 ns、wt 3.6–26.1 ns，比值 wb128/wt 33–838；去掉可疑点后 wb128/wt 从 134 变为 61。wa 无法辨识：关键 rank 的激活槽数在所有配置里只在 6.44–8.00 之间变化。不考虑 bm 时 wb 拟合为负值（同一 T 下 block 与 token 高度共线）。按 T 分别拟合的 token 单价（block 单价共享）：0 / 6.2 / 14.3 / 14.3 ns（T=128/256/512/1024）→ token（dispatch）项随 T 变重。
- 样本外：bm240 实测比预测慢 19–23 µs、bm64 慢 9.9 µs（dispatch 只有 8 个 warp）、native 慢 17–40 µs → 线性模型只在 bm 96–192、不均衡 ≤ 1.25 的区间成立。
- 结论：现有观测数据定不下成本函数；需要做受控实验（合成 routing，让一个 rank 成为唯一关键 rank，一次只改 token / block / 激活槽中的一个）。

### 8.8 成本函数受控实验（Q5，2026-11-09）

- 合成 routing：`pkg/gcbench/syn_routing.py`（各节点用固定种子自行生成到 `$W/sgpkg/data/ep64/syn/`，16 节点合并 md5 e7de605d4310，33 个文件）。plain 模式 P=512（每卡 8 槽），每个文件 80 个 call，每个 call 的槽计数完全相同；token 顺序每个 call 打乱。rank 0 为唯一关键 rank，组成按设计；rank 1–63 每槽固定背景（T=128：90 / 槽，T=512：360 / 槽，约等于真实均值）；多余的 token-topk 置 −1（gc-moe 计数、source-index、combine 均跳过 ≥ P 的路由）。
- 设计（三条轴，每次只动一个量）：T=128 bm128：token 1040–2048 @16 block / 8 槽；block 12/14/16/18 @1536 token；激活槽 4/6/8 @1536 token / 12 block。T=128 bm96：token 1200/1560/1920 @20 block；block 16–22 @1536；激活槽 4 vs 8 @1536 / 16 block。T=512 bm128：token 4200/4660/5120 @40 block；block 36/39/42 @4600；激活槽 4 vs 8 @4800 / 40 block。T=512 bm192：token 3900/4640/5376 @28 block；block 25/28/31 @4800；激活槽 4 vs 8 @4800 / 28 block。
- 队列 `results/q_q5.json` → `results/q5.jsonl`（gc-moe-r6，`--block-m` 显式指定，`--chunks 2 --passes 15 --warmup-passes 2`，QUEUE_PORT 26200）：1 个门槛 + 33 个变体 × 2 次。
- Q5 中途（T=128 bm128，rank 0 约为背景 2 倍）：token 轴 1040→2048 斜率约 30 ns/token；block 轴（1536 token，12/14/16 block）不增反降（109.23 / 108.86 / 108.04）→ rank 0 此时受 dispatch 限制、block 不在关键路径上。为覆盖真实 plan 的"接近均衡"区间，追加近均衡组（axis `nb_*`，rank 0 为背景的 1.0–1.4 倍）：T=128 bm128：token 720–1024 @8 block；block 8/10/12/14 @960 token；激活槽 4 vs 8 @960 / 8 block。T=512 bm128：token 3200–4096 @32 block；block 29/32/35 @3600；激活槽 4 vs 8 @3600 / 32 block。17 个变体（50 个文件，16 节点 md5 一致），Q6（`results/q_q6.json`，34 次，QUEUE_PORT 26300）由 `/tmp/chain_q6.sh` 在 Q5 之后自动启动。
- Q5 结果（重载区间，rank 0 为背景的 1.4–2.8 倍；66/67 次有效，重复 spread 中位数 0.20 µs）：

| T | bm | token 单价 | block 单价 | 激活槽单价 | token 单位 wb / wa |
|---|---|---|---|---|---|
| 128 | 128 | 29.7 ns（se 1.9） | 0.10 µs（se 0.20） | 0.43 µs（se 0.08） | 3.5 / 14.5 |
| 128 | 96 | 26.0 ns（se 2.4） | 0.40 µs（se 0.13） | 0.29 µs | 15.5 / 11.1 |
| 512 | 128 | 31.8 ns（se 1.3） | 0.24 µs（se 0.06） | 0.25 µs | 7.4 / 7.8 |
| 512 | 192 | 32.2 ns（se 0.5） | 0.17 µs（se 0.02） | 0.36 µs | 5.2 / 11.3 |

  现行 planner 为 wb 293–335、wa 38–52（token 单位）。重载区间下关键 rank 受 dispatch（入口带宽）限制：token 约 30 ns/个，block 与激活槽都便宜。
- Q6（近均衡，34 次）+ Q7（补测噪声点，18 次）后的最终结果（`q5_summary.py`；共 119 次有效，各变体 ≥ 2 次取中位数；两个单次离群点 212.75 / 113.1 已被补测压掉）：

| 区间 | T | bm | token 单价 | block 单价 | 激活槽单价 | token 单位 wb / wa |
|---|---|---|---|---|---|---|
| 重载 | 128 | 128 | 29.7 ns（se 1.9） | 0.10 µs（se 0.20） | 0.43 µs（se 0.08） | 3.5 / 14.5 |
| 重载 | 128 | 96 | 26.0 ns（se 1.0） | 0.40 µs（se 0.13） | 0.29 µs | 15.5 / 11.1 |
| 重载 | 512 | 128 | 31.8 ns（se 1.3） | 0.24 µs（se 0.06） | 0.25 µs | 7.4 / 7.8 |
| 重载 | 512 | 192 | 32.2 ns（se 0.5） | 0.17 µs（se 0.02） | 0.35 µs | 5.2 / 10.8 |
| 近均衡 | 128 | 128 | 21.5 ns（se 4.2） | 0.52 µs（se 0.26，非单调） | 0.44 µs | 24 / 20 |
| 近均衡 | 512 | 128 | 36.0 ns（se 0.5） | ≈0（−0.08） | 0.21 µs | ≈0 / 5.9 |

  近均衡 T=128 的 token 轴有拐点：rank 0 比背景多 ≤ 10% 时几乎不涨（720→800：+0.19 µs），超过后 31–37 ns/token（max 是"软"的，其他 rank 同时接近关键）。
- 验证（`cost_validate.py`，49 组真实 plain 配置，只拟合各 T 截距）：现行 planner 成本函数 RMSE 7.5–9.3 µs（最大 19–22）；只用 token（30 ns）2.24；token 30 ns + block 0.1 µs + 激活槽 0.4 µs 2.31；回归（样本内）1.90。
- 结论：EP64 L20B 上关键 rank 受 dispatch 入口带宽限制，成本函数以 token 为主：t_r ≈ 30 ns·tokens + 0.1–0.5 µs·blocks + 0.2–0.45 µs·active（token 单位约 wb ≈ 10、wa ≈ 12），现行 planner（wb 293–335、wa 38–52）把 block 高估约 20–30 倍。
- 下一步：planner 改用新权重（如 wb 10、wa 12）→ 生成新权重的 oracle routing，实测零开销收益 G（预期 T=128 P512 token 峰值从 −3.4% 变为约 −11%，多约 2 µs）→ 再测 ECO。

### 8.9 新成本权重（wb 10，wa 12）验证（Q8，2026-11-09）

- 离线（T=128 bm96）：token 峰值 EPLB 938.5 / 953.4 → 旧权重 ECO 895.7 / 897.0 → 新权重 ECO 870.1 / 845.6（P448 / P512）；token 不均衡 1.222 / 1.241 → 1.166 / 1.168 → 1.133 / 1.101；block 峰值略升（13.33 → 13.39，12.36 → 13.12）。按新成本函数（30 ns/token + 0.3 µs/block + 0.35 µs/槽）估算的关键 rank 时间：EPLB 34.69 / 35.78 → 旧 32.87 / 32.67 → 新 32.25 / 31.41 µs。
- `ep64_oracle.py` 新增 `--weights=wb,wa`；新权重 oracle `ora<P>b<bm>c10x12<sfx>.npy`（8 个，与 Q4 同 bm，16 节点 md5 一致）。ECO 用 `GC_MOE_ECO_UNIT_CFG=10,12,<nr>,6,2`（P448 nr=3，P512 nr=2）。
- Q8（`results/q_q8.json` → `results/q8.jsonl`，65 次，QUEUE_PORT 26500）：门槛 gW_chk512_T128（新权重 ECO --check）→ 8 个配置 × {oraW, oraN, ecoW, ecoN} × 2 次交错。
- Q8 中途（rep 0，T=128–512，单次）：零开销上界 oraW − oraN = +0.52 / +0.20 / −0.04 / +0.88 / −0.12 / +0.26 µs（新权重的 oracle 没有变快）；ECO ecoW − ecoN = −0.53 / −1.21 / −0.67 / −1.72 / −0.23 / −0.87 µs（新权重的 ECO 稳定快约 0.9 µs，来自机制开销 O 下降）。离线模型原本预测上界能快 0.6–1.3 µs → 疑似"软最大值"：接近均衡时只压最慢 rank 不省时间。待验证脚本 `cost_agg.py`（max / top-k 均值 / log-sum-exp / p 范数，对比 oraW 与 oraN 的预测差和实测差）。
- Q8 已按用户要求暂停（2026-11-09）：31/65 次完成（rep 0 中 15/16 个配置；ecoW/ecoN512_T1024 与全部 rep 1 未跑）；本地队列已停，16 节点无残留进程、GPU 空闲。rep 0（单次）：oraW − oraN = +0.52 / +0.20 / −0.04 / +0.88 / −0.12 / +0.26 / +0.16 µs（上界没变快）；ecoW − ecoN = −0.53 / −1.21 / −0.67 / −1.72 / −0.23 / −0.87 / +1.16 µs（T=1024 P448 为 +1.16）。恢复：重跑 `results/q_q8.json` 中缺的条目；软最大值验证脚本 `cost_agg.py` 已写好未运行。

### 8.10 小 batch（T=32/48/64）最慢 rank 的决定因素（Q9，2026-11-09）

- 判断（实测前）：gc-moe 默认 bm 为 32（P512 T=32）或 64，bm < 96 时 dispatch 每 SM 只有 8 个 warp；每卡约 190–380 个 token-expert 对，每个激活槽约 1 个 block。预计最慢 rank 的时间 ≈ 固定延迟（barrier、count 交换、dispatch 往返、combine 信号，各 rank 相同）+ 激活专家数 × 一份 FP4 权重（约 17.7 MB）的 HBM 读取；token / block 数影响小，均衡几乎无用，复制专家（槽多）反而可能更慢。bench 每个 call 复用同一套权重，L2 可能低估权重读取。
- 数据：`pkg/gcbench/small_t.py` → `$W/sgpkg/data/ep64/small/`（16 节点合并 md5 29be72f6c817）：真实 routing 截取每源前 T 个 token（routing_off / eplb448 / eplb512 的 `_t32/_t48/_t64`）；T=64 均匀合成负载（每个 rank 相同组成，bm64）：a8t48 / a8t24 / a8t8 / a4t96 / a4t48 / a2t192 / a1t8（a=激活槽数，t=每槽 token）。
- Q9（`results/q_q9.json` → `results/q9.jsonl`，22 次，QUEUE_PORT 26600，各 1 次）：T=32/48/64 × {native384, eplb448, eplb512, eco512}（默认 bm）+ 7 个均匀负载 + 3 个 eager trace（eplb512 T=32/64，eco512 T=64）。
- Q9 结果（22/22，各 1 次，p50−copy，µs；默认 bm）：T=32：native384 91.78 / eplb448 65.39 / eplb512 70.30 / eco512 87.84；T=48：112.66 / 70.79 / 71.67 / 86.72；T=64：114.02 / 75.97 / 76.46 / 91.14。T=64 均匀负载（每个 rank 相同，bm64）：a1t8 49.81（底座）；a4t48 58.46；a2t192 66.26；a4t96 68.04；a8t8 63.70；a8t24 69.45；a8t48 70.27 → 同 8 槽 8 block 时 token 64→384 只 +6.6 µs；同 192 token 时 4 槽 4 block → 8 槽 8 block +11 µs；每个 block 约 2 µs 量级，token 很便宜。eager trace：eplb512 T=32 dispatch 9.5 / 计算结束 33.5 / combine 尾巴 15.9；T=64 16.5 / 37.5 / 16.8；eco512 T=64 21.5 / 41.4 / 23.3。→ 小 T 下关键路径是固定开销（约 50 µs）+ 每个 block 的 GEMM（受权重 / 延迟限制），token 与 dispatch 次要；极端不均衡（native）仍很伤（热 rank 的 block 多）；ECO 机制开销 +15–17.5 µs。

### 8.11 生产数据（EP32，DeepSeek-V4.1-Flash，每卡每次 MoE call 192–256 token）分析（2026-11-09，未实测）

- 生产：EPLB 关 R0：SP step 17.109 ms / moe_lb 2.464；EPLB R0：16.168 / 1.617；R64：16.624 / 1.187；R128：15.978 / 1.070。
- 每卡平均收 6T = 1152–1536 个 token-expert 对，与 EP64 的 T=192–256 相同 → 处于 EP64 实测的"dispatch 主导"区间（关键 rank 约 30 ns/token），不是小 T 的 block 主导区间。
- gc-moe 默认 bm（EP32）：R0 为 bm128（T=192）/ bm192（T=256），R64、R128 为 bm96 / bm128，都是 12 个 dispatch warp → 之前"bm64 + 8 warp 跳档"的假设不成立；bm 差异按 EP64 扫描只有 1–3 µs，且偏向 R64/R128 更快。
- 任何"a·moe_lb + b·每卡槽数"的平滑模型都无法同时给出 R64 > R0 与 R128 < R0（要求 b > 0.215a 且 b < 0.137a）→ R64 需复测或另查原因。
- EP32 的特点：每卡 12–16 个专家、各层权重不同（生产不会像 bench 那样复用权重命中 L2）→ 每个激活专家约 17.7 MB FP4 权重的 HBM 读取可能进入关键路径。粗算：R0→R128 的 token 收益约 (1.617−1.070)×6T×30 ns ≈ 19–25 µs/层；实测每步只快 0.19 ms → 若 MoE 层数约 40，每卡多出的 4 个激活专家每个约 3.6–5 µs，与 HBM 读权重同量级 → 与"R128 用 dispatch 换来了权重读取"一致。
- ECO 的机会：R128 式的 token 均衡 + R0 式的激活数（只为让 rank 变热的专家激活副本）；收益在通信（热 rank 的 dispatch 入口 + combine 等最慢 rank），代价在计算（新激活专家的权重读取）。成本函数应按两种资源建模：t_r ≈ max(tokens·t_tok, active·t_w + blocks·t_blk) + 尾巴；bench 测到的激活代价（0.3–0.45 µs）因权重复用而偏低，需用每 call 轮换权重的 bench 重测。

### 8.12 计划：真实 DSV4.1-Flash 权重 + 完整 40 层，重测 eco512 vs eplb512（T=128，待确认）

- 现成资源：checkpoint `/mnt/hf3fs/3fs/models/DeepSeek-V4.1-Flash`（开发机可读，476 GB；40 层全是 MoE，384 路由专家 + 1 共享，topk 6；路由专家 FP4：w1/w3 [2304, 2560] int8、w2 [5120, 1152] int8，E8M0 scale 每 32 元素一个；每个 expert-layer 约 18.8 MB，全部路由专家约 289 GB）。集群节点访问不到 /mnt/hf3fs，每节点本地盘约 1 TB 空闲。40 层真实请求轨迹 `~/work/mpo/data/real1/{A,B}_tracks40.npz`（[tokens, 40, 6]）。生成器 `pkg/gcbench/gen_ep.py`（目前 L=8，用 rtp_eplb/policy.py 与 eco.py）。
- gc-moe 权重接口：`transform_weights_for_gc_moe` 的 L1 输入是 [gate; up] 按行拼接（内部自己交错），SF 走 `deep_gemm.transform_sf_into_required_layout`；checkpoint 需 w1‖w3 拼成 [4608, 2560]，E8M0 转 float scale。FP4 nibble 顺序需验证（参照 bundle41 里 rtp_llm dsv41 的加载代码）。
- 权重传输调查（2026-11-09）：集群节点 `/mnt/fuse/oss/nanjun.cp` = OSS（oss://shark-console/nanjun.cp/）的 fuse 挂载，16 节点共享（part0 写、part1 数秒内可读；刚写完立即回读会 I/O error）。本机没有这个挂载；本机 ossutil 配的是别的账号/endpoint（oss-cn-hangzhou-zmf-internal），访问 shark-console 返回 AccessDenied（上海 endpoint 报"bucket 不属于你"）。本机经 ssh 网关到集群只有约 0.2 MB/s（32 MB 用 156 s），289 GB 不可行；本机 hf3fs 读 669 MB/s。集群节点能访问 huggingface.co / hf-mirror.com（200），HF 上有公开仓库 `deepseek-ai/DeepSeek-V4.1-Flash`（private=false）→ 建议在集群上直接下载（只下含路由专家的 shard）到 `/mnt/fuse/oss/nanjun.cp/models/` 或各节点本地盘。测试文件已删除。

### 8.12 L2 复用验证（2026-11-09 19:4x，受阻）

- 共享工具目录里的 `pkg/gcbench/gc_eco_bench.py` 在 19:22 被另一个 session（EP32）改过：已加入 `--weight-sets K`（第 i 个 call 用第 i mod K 套权重，plain 模式会自检"各套输出不同、同套重复逐位一致"），md5 2617382c74de。该 session 正在自己的 EP32 集群（前缀 e02-sg-i5c4uouzh0h，与本集群不冲突）跑同类 L2 实验（`queue_l2.json` → `results/ep32_l2.jsonl`）。本 session 未读其结果。
- 打算直接复用 `--weight-sets`（快照 /tmp/gcbench-k.tgz），先跑快速对照：EPLB512 / oracle512 T=128、EPLB512 T=256、受控实验的 act4 与 blk12（8 槽）、T=64 均匀负载 a4t48 与 a8t24，各 K=1 / K=8 相邻交错。
- 部署失败：16 个节点全部报 `wt: could not select one pouch workdir container ... Matching containers seen: <none>`，重试后变为 `kex_exchange_identification: Connection closed` → EP64 任务的容器不可达（疑似任务被停止或重启），需用户确认任务状态。

### 8.13 EP8（B300 主机 .113）基于 benchmarks/benchmark_gc_moe.py 的测试（2026-11-09）

- 新脚本 `benchmarks/benchmark_gc_moe_eco.py`：沿用 `benchmark_gc_moe.py` 的计时（单个 call 抓 graph、每窗口回放 inner_iters 次、窗口前清 L2 + rank barrier、各 rank 中位数取最慢 rank），支持 plain / eplb / ora / eco 四种模式、真实 routing、`--cold-l2`（graph 内每次回放前 memset 2×L2，并扣掉单独计时的清空开销）、`--check`（ora / eco 输出与 plain 逐位比较）。权重按逻辑专家生成，副本权重相同。
- 主机：.111 忙；.112 有他人进程；.114 ssh 公钥被拒；.115 的 GPU7 在 CUDA 下不可用（"No CUDA GPUs are available"），只有 7 卡；.113 8 卡空闲，环境整体复制到 `~/ep8env`（conda310、gcenv、dg341_smx、gc-moe-r6、数据），不碰该机已有文件。运行脚本 `~/ep8run.sh`，结果 `~/ep8res/`。
- 数据：EP8 redund（40 层；EPLB placement 来自不重叠的历史窗口）；新增 eplb384（只重排、零冗余，对应生产 R0，`gen_redund.py "384"`）；oracle `data/ep8/ora{400,416,448}b32.npy`（T=128）与 `data/ep8_T256/ora*b64.npy`（T=256），`ep64_oracle.py` 加了 `ORACLE_LAYERS=40` 与 R≤32 的轮数。
- 冒烟（T=128，2 个 call，warm L2）：plain384 172.3 / eco448 176.2 / ora448 173.3 / eplb448 187.5 µs；eco448 与 ora448 输出和 plain 逐位一致。
- 完整矩阵 `~/ep8_matrix.sh`：T ∈ {128, 256} × {warm, cold L2} × {正序, 倒序}，每次 11 个模式（plain384、eplb384/400/416/448、ora400/416/448、eco400/416/448）× 16 个 call。
- EP8 矩阵结果（.113，各 16 个 call，正序 / 倒序均值，最慢 rank 中位数，µs）：
  - T=128 warm：plain384 175.3｜eplb384 174.8、eplb400 178.3、eplb416 179.3、eplb448 187.4｜ora400 174.4、ora416 174.1、ora448 173.6｜eco400 174.7、eco416 175.3、eco448 175.8。cold：188.0｜186.6 / 190.3 / 192.5 / 199.6｜187.1 / 185.9 / 186.0｜188.7 / 189.0 / 190.7。
  - T=256 warm：186.2｜186.8 / 188.5 / 189.8 / 197.7｜186.1 / 185.1 / 184.8｜187.8 / 189.2 / 188.9。cold：198.8｜198.6 / 200.9 / 202.9 / 209.9｜198.1 / 196.7 / 196.6｜203.5 / 202.6 / 204.2。
  - cold − warm：plain / eplb / ora 约 +11.6 ~ +13.2，ECO +13.4 ~ +15.7；与槽数无关（eplb448 +12.2 vs plain +12.7）→ EP8 每卡权重 0.85–1 GB 远大于 L2，warm 模式本来就几乎没有权重命中；+12 µs 的常数更像是 memset 留下的脏行回写，不是权重缺失。EP64 P512（每卡 141 MB ≈ L2）上的影响仍需单独测；清 L2 应改为读而非写。
  - 结论：EP8 不均衡小（native token 不均衡 1.36），零开销上界只比 plain 快 0.1–1.7 µs；EPLB 冗余越多越慢（全部副本激活），eco 基本持平 plain，比同 placement 的 EPLB 快 3.6–11.6 µs（warm T=128），但比 oracle 慢 1–4 µs（cold T=256 达 5–7.6 µs）。

### 8.14 第一步离线模拟：完全均匀负载离现有 oracle 多远（2026-11-10，集群不可用）

- 方法（`/tmp/uniform_sim.py`）：每 rank 成本 30 ns·token + 0.3 µs·block + 0.35 µs·激活槽，五种聚合（max、前 4 / 前 8 均值、log-sum-exp τ=1/2 µs）分别对 60+ 组实测拟合（RMSE 2.9–3.9 µs）；均匀负载 = 每个 call 所有 rank 都取该 call 的均值 token / block / 激活数（bm96）。
- 预测（eplb / 旧 oracle / 新 oracle / 均匀，µs）：硬 max 下 T128 P448 89.2 / 87.4 / 86.8 / 83.6，P512 90.2 / 87.3 / 86.0 / 83.7；T256 P448 123.5 / 118.8 / 118.3 / 112.6，P512 125.0 / 117.5 / 116.1 / 112.8。前 k 均值相近，log-sum-exp τ=2 的差距更大（约 8–10 µs）。
- 注意：模型预测新 oracle 比旧的快 0.6–1.5 µs，实测却持平或略慢 → 接近均衡时模型高估均衡收益，均匀负载的收益应视为上限：T=128 约 2–3 µs（约 3%），T=256 约 3–6 µs（约 3–5%）。需上 GPU 用均匀合成负载确认（每 rank 768 token，8 或 12 个 block，8 个激活槽）。
- T=192 补充（`/tmp/uniform_t192.py`；每个 call = 一个 T=128 call + 下一个同层 call 的前 64 token；EPLB 静态路由截取，oracle 用参考 planner bm96 新旧权重重算；截距在 T128 与 T256 之间线性插值，无实测点）：硬 max 下 eplb / 旧 oracle / 新 oracle / 均匀 = P448 106.2 / 103.2 / 102.6 / 98.1（剩余 4.5），P512 107.4 / 102.3 / 101.2 / 98.3（剩余 2.9）；前 k 均值剩余 3.0–5.2，log-sum-exp τ=2 为 9.2–9.7。token 不均衡：eplb 1.215 / 1.234，旧 oracle 1.142 / 1.123，新 oracle 1.122 / 1.081。

### 8.15 EP32 版离线预测（2026-11-10，`/tmp/uniform_ep32.py` → `/tmp/uniform_ep32.out`）

- 数据 `data/ep32`（1200 call × 32 rank × T=128，8 层；EPLB 416/448/512 + ECO metas；T=192/256 由同层 call 拼接）；新增 EPLB R0（384 槽只重排，同一 policy + hist_A）。bm96（新表下 EP32 T≤256 都是 96）。斜率沿用 EP64 拟合，没有 EP32 实测 → 只给"比自身完全均匀负载多出的 µs"。
- 硬 max（µs，T=128 / 192 / 256）：native384 44 / 66 / 89；EPLB R0 25 / 38 / 51；eplb416–512 3.3–3.4 / 4.9–5.0 / 6.3–6.5；旧 oracle 3.4–3.9 / 4.0–4.2 / 4.4–4.6；新 oracle P416 2.5 / 3.1 / 3.8，P448 1.8 / 2.1 / 2.3，P512 1.6 / 1.7 / 1.9。前 k 均值结论相同；log-sum-exp τ=2 对所有配置都多出约 7.5–9（软 max 的常数项）。
- token 不均衡：native 2.81；EPLB R0 2.04；eplb416–512 约 1.13；旧 oracle 1.10–1.19；新 oracle 1.08–1.10（T128）→ 1.05–1.08（T256）。
- 结论：EP32 上最大的空间在零冗余 EPLB（每层 25–51 µs，随 T 增长）；一旦有 +32 冗余，EPLB 只剩 3–6.5 µs；新权重 oracle 再减半到约 2 µs。生产 R0 的 moe_lb 为 1.617，本数据 EPLB R0 为 2.04（placement 来自 A 组历史、测的是 B 组流量），量级需用生产 routing 校准。

### 8.16 EP8 模拟 EP64 的校准（2026-11-10，.113）

- 生成器 `pkg/gcbench/emul_ep8.py`（数据 `data/emu`）：每个 call 取 EP64 中按成本（30 ns·token + 0.3 µs·block + 0.35 µs·槽）最重的 8 个 rank，把它们的每槽负载放到 8 张卡上；均匀组为该 call 的均值组成；同一配置三组共用源端 T（T=128→152，T=256→304，多余项 −1）。plain 模式，bm96，16 个 call，正序 / 倒序各 1 次。运行脚本 `~/emu_run.sh`，结果 `~/ep8res/emu_*.json`。
- 结果（µs）：T128 P448 eplb 79.07 / ora 79.04 / uni 76.40；T128 P512 81.16 / 77.61 / 76.16；T256 P448 108.90 / 104.50 / 99.03；T256 P512 109.77 / 100.56 / 98.79。
- 校准（模拟的 eplb−ora 对比 EP64 实测）：+0.03 vs +2.11（偏差 2.1）、+3.55 vs +4.87、+4.40 vs +5.58、+9.21 vs +8.36 → 4 组中 3 组误差 ≤ 1.3 µs，整体略偏小，T128 P448 对不上。
- 完美均衡的 gap（模拟）：eplb−uni = 2.7 / 5.0 / 9.9 / 11.0；旧 oracle−uni = 2.6 / 1.5 / 5.5 / 1.8。
- 第二轮（2026-11-10）：`pkg/gcbench/emul2.py` 已生成 EP64 top-4 变体（`emu_*_{eplb,ora}4.npy`）与 EP32 模拟数据（`e32_T{128,192,256}_{eplb384,eplb448,ora448,uni448,eplb512,ora512,uni512,uni384}.npy`，源端 T = 328 / 488 / 648），已同步到 .113 `~/ep8env/data/emu`；运行脚本 `~/emu_run2.sh`。
- 未跑成：启动时 .113 的 8 张卡已被另一用户（luohaoc…）的 rtp_llm 任务占用（每卡 130–150 GB）；本 session 的进程已全部停止，未留结果。其他主机：.112 满载、.111 忙、.114 公钥被拒、.115 GPU7 不可用。待 .113 空闲后重跑 `~/emu_run2.sh`。
- 第二轮完成（2026-11-10 13:02–13:17，.111 空闲 8 卡，`~/ep8watch/watch.sh` 发现后自动同步环境运行；结果 .111 `~/ep8res/`）。
  - EP64 校准（4 次：.113 两次 + .111 两次；最重 8 rank / 最重 4 rank + 均值）：eplb−ora = +1.18 / +3.77 / +4.57 / +10.30（实测 +2.11 / +4.87 / +5.58 / +8.36）；top-4 变体 +1.86 / +4.91 / +6.67 / +9.09（误差 −0.25 / +0.04 / +1.09 / +0.73）→ top-4 变体更准，误差 ≤ 1.1 µs。EPLB−均匀 = 3.5 / 4.5 / 9.9 / 11.6，旧 oracle−均匀 = 2.3 / 0.8 / 5.4 / 1.3。
  - EP32（2 次，最重 8 rank）：T128 / 192 / 256：EPLB R0 114.95 / 135.37 / 160.33 vs 均匀 88.85 / 96.24 / 107.98（gap 26.1 / 39.1 / 52.4，与离线预测 25 / 38 / 51 吻合）；EPLB448 92.38 / 104.15 / 116.63 vs 均匀 88.09 / 98.53 / 109.08（4.3 / 5.6 / 7.6），新 oracle448 90.45 / 100.38 / 112.31（2.4 / 1.9 / 3.2）；EPLB512 93.98 / 106.87 / 121.15 vs 均匀 86.84 / 98.64 / 112.10（7.1 / 8.2 / 9.1），新 oracle512 89.35 / 99.87 / 111.84（2.5 / 1.2 / −0.3）。
- EP64 最新模拟（2026-11-10 13:29–13:36，.113，top-4 方法，`emul3.py` / `~/ep8watch/emu3_run.sh`，结果 .113 `~/ep8res3/`，2 次正序 / 倒序）：EPLB / 旧 oracle / 新 oracle / 均匀（µs）：T128 P448 78.71 / 78.73 / 79.36 / 75.11；P512 80.31 / 77.35 / 77.27 / 75.42；T192 P448 94.02 / 90.18 / 90.10 / 85.96；P512 94.78 / 88.53 / 89.37 / 86.09；T256 P448 108.25 / 104.10 / 103.86 / 99.34；P512 109.77 / 100.77 / 100.85 / 99.50。EPLB−均匀 3.6 / 4.9 / 8.1 / 8.7 / 8.9 / 10.3；新 oracle−均匀 P448 4.3 / 4.1 / 4.5（受 placement 限制），P512 1.9 / 3.3 / 1.4；新旧权重 oracle 相差 ≤ 0.9 µs。
- 均匀组细分（emu4，`emul4.py`：uniB 最少 block + 均值激活，uniBa6 最少 block + 6 激活，uniBa1 最少 block + 均值+1 激活）。.113 上 13:54 起另一用户（yanxi.w）的 rtp_llm 任务占卡 → 13:54 之后的 run（T256_P512_r0 后半段及全部 r1）作废；脚本处于等待状态，空闲后自动续跑。只采用 r0 中 13:54 前完成的结果。
- emu4 复测（.113，r1 干净；r0 每轮最后一个 arm 偶发 140–187 µs 异常，组成正常，r1 倒序时正常 → 测量异常，作废）：均匀(oracle 均值 block) vs 最少 block：T128 77.7/76.5 vs 70.5/69.2；T192 85.3/85.9 vs 80.0/79.9；T256 98.4/98.8 vs 94.7/94.3。激活 6 / 均值 / 均值+1 差 ≤ 1 µs。
- 改动：`GC_MOE_ROUTING_WARPS` 覆盖 dispatch warp 数（heuristics/gc_moe.py）；role1 三处 launch 的线程数改用 `self.dispatch_warps * 32`（原来写死 `384 if block_m >= 96 else 256`，导致覆盖时 launch 报 INVALID_VALUE）。默认行为不变。bm<96 配 12 warp 的输出未做 bit-exact 校验。
- emu5 bm × warp 扫描（.111，emu3 负载，2 次）：EPLB / 新 oracle / 均匀 / 最少 block（µs）
  - T128 P448：bm96w12 81.3 / 79.2 / 77.5 / 69.2；bm64w12 87.4 / 84.8 / 82.4 / 80.0；bm64w8 89.3 / 86.6 / 82.2 / 79.8；bm32w12 112.5 / 107.8 / 103.1 / 92.9
  - T128 P512：88.4 / 80.1 / 77.6 / 69.1；88.1 / 83.3 / 82.4 / 79.0；91.8 / 83.6 / 82.5 / 79.7；113.4 / 105.6 / 103.3 / 92.7
  - T192 P448：94.9 / 91.7 / 87.1 / 81.0；103.2 / 99.7 / 97.2 / 91.0；106.4 / 101.8 / 99.0 / 93.6；142.3 / 135.3 / 127.2 / 120.4
  - T192 P512：96.8 / 89.5 / 87.5 / 81.6；104.6 / 99.2 / 97.4 / 91.8；108.4 / 99.7 / 98.2 / 94.0；144.7 / 131.7 / 125.9 / 122.0
  - T256 P448：109.9 / 105.6 / 101.1 / 96.2；121.3 / 116.4 / 109.0 / 106.0；125.6 / 119.2 / 110.8 / 107.7；174.6 / 164.9 / 152.8 / 147.3
  - T256 P512：111.4 / 102.4 / 100.6 / 95.6；124.0 / 113.5 / 110.8 / 103.9；126.9 / 115.8 / 113.2 / 108.1；179.0 / 159.8 / 151.3 / 147.3
  - 结论：小 bm 全面更慢（bm64 +5–12，bm32 +25–50 µs），12 warp 只比 8 warp 快 0–4 µs；时间不随 padding 行数下降，而随 block 数上升 → 推测每个 block 都要重读一遍该专家权重（约 17.7 MB，约 2 µs），block 数 ≈ 权重读取次数。
- emu6 权重读取假设检验（.114 本地，3 次，spread ≤ 1.9 µs；每卡相同，8 槽；a4 = 4 专家等分，a8 = 8 专家等分）中位数 µs：
  - 768 token：bm96 a4（8 block）68.34 / a8（8 block）70.59；bm128 a4（8）72.77 / a8（8）73.02；bm192 a4（4 block）72.96 / a8（8）76.98
  - 1536 token：bm96 a4（16）94.49 / a8（16）95.17；bm128 a4（12）96.12 / a8（16）100.03；bm192 a4（8）96.14 / a8（8）97.84
  - 结论：假设"block 数 = 权重读取次数、大 tile 少读权重就更快"不成立（bm192 4 个 block 比 bm96 8 个 block 慢 4.6 µs）；bm96 在同 token 下最快；同一 bm 下多 block、多激活专家都会变慢（bm192 4→8 block +4 µs；bm96 同 8 block 时 4→8 专家 +2.3 µs）。"最少 block"组的优势来自 bm96 下不满 block 更少，真实 routing 下这部分由冷专家决定，规划算法拿不到。
- EP32 R32（P416，每卡 13 槽）补测（`emul7.py`，最重 8 rank，.111，2 次，EPLB448 同场作锚点：92.52 / 104.94 / 116.76，与此前 92.38 / 104.15 / 116.63 相差 ≤ 0.8 µs）：EPLB / 新 oracle / 均匀 = T128 92.39 / 89.86 / 87.54，T192 102.93 / 101.66 / 97.88，T256 116.55 / 114.81 / 108.79；EPLB−均匀 4.9 / 5.1 / 7.8，oracle−均匀 2.3 / 3.8 / 6.0；token 不均衡 EPLB 1.137 / 1.129 / 1.124，oracle 1.098 / 1.086 / 1.078。

### 8.17 EP32 实测准备（2026-11-10，临时借用 EP32 集群 task `L20B_31f2113b`，8 节点 × 4 L20B）

- 集群：part0–7 = a0015/a0003/a0011/a0007/a0002/a0004/a0018/a0008（容器 IP 10.44.42.103/32.34/17.99/46.245/24.97/19.250/17.151/1.75），`ep32_cluster.sh` 可访问，32 卡空闲。
- 只用本 session 的目录 `$W/ep64s/`（pkg = 本仓库当前树 + `pkg/gcbench`；data；cache）；另一 session 的 `$W/sgpkg` 只读（复用其 `gcenv`、`sgpkg/dg341`，以及与本地逐字节相同的 `sgpkg/data/ep32` 源文件）。
- 本地→集群上传很慢（约 100 KB/s，1.3 GB tar 卡在 30%），改为只上传 46 MB（eplb448 routing、eco448 metas、oracle quotas、`build_remote.py`），在每个节点本地重建 `$W/ep64s/data`（31 个文件），7 个抽检文件 md5 与本地一致。
- 数据：`~/work/mpo/data/ep32x`（`pkg/gcbench/ep32_prep.py`）：routing_off / routing_eplb{384,416,448,512}，T=128 / 192（call + 下一同层 call 前 64 token）/ 256；oracle-W = 参考 planner（GC_MOE_ECO_UNIT_CFG=10,12,2,6,0，bm96）的 `ora{P}b96c10x12[_T*].npy`。ECO 的 phy2log 与 EPLB 是同一 placement（每个 rank 专家集合相同，仅 rank 内槽序被 permute_local 重排）。`ep64_oracle.py` 新增 `--tail=n`（merge 时最后一个 call 只取前 n 个 token）。
- 离线统计（bm96，`offline_stats_bm96.json`）：token 不均衡 T128/192/256：eplb384 2.19，eplb416 1.14，eplb448 1.13，eplb512 1.11→1.09；oraW416 1.09→1.06，oraW448 1.08→1.05，oraW512 1.09→1.05。最慢卡 block（T128）：eplb 17.7–18.1（R0 24.7），oraW 16.9–17.3，平均 15.3。
- 队列：`results/q_ep32real.json`（4 个 ECO gate + 30 配置 × 2 次 + K=1 对照，共 66 次），执行器 `ep32_queue.py`（ep64_queue 的 EP32 版，路径指向 `$W/ep64s`）。oracle-F（最大流取整）未做。
- oracle-F（`pkg/gcbench/oracle_flow.py`，2–8 ms/call）：每 call 二分 + 整数最大流求最小可达最大 rank 负载 L*，再在不超过 L* 的前提下合并小片（少激活）、余数对齐到 block。EP32 bm96 T128/192/256：token 不均衡 R32 1.064/1.061/1.059，R64 1.019/1.019/1.018，R128 1.000；最慢卡 block R128 18.4/22.1/25.9（oracle-W 17.3/20.8/25.0），激活峰值 R128 15.5（oracle-W 15.2–15.9）。变体 Fb48（合并/对齐允许超出 L* 48 token）：token 1.03–1.12，block 峰值 17.7/21.4/25.3，不优于 oracle-W，未纳入。集群上 `ora{P}F[_T*].npy` 已在 8 节点重建（md5 与本地一致），队列增至 39 条 / 84 次。
- token 与 block 不能同时最优的原因分析（`/tmp/joint_cause.py`，EP32 T128，24 call）：要求 token 上限 = 均值且 block 上限 = 真实下界；理想 placement（每个专家在每卡都有副本）下对齐切分的启发式也只到 token 1.084；冗余从 R128 加到 R256 不改善"block 取下界时的 token 不均衡"（J+0：1.113→1.131），T 翻倍后减半（T256：1.05–1.06）→ 主因是 block 粒度（96 token）相对每卡负载（768 = 8 个 block）太粗：一张卡要补 x 个 token 又不能多开 block，只能整块（96）补，误差 ±48（约 6%）。R32 另受 placement 限制（token 最优只到 1.06）。
- 候选算法离线对比（`/tmp/eval_plans.py`，30 call；rank 成本 = token + 55·block + 12·激活，×30 ns 换成 µs；最慢 rank 成本）：新增 oracle-L（`pkg/gcbench/oracle_ls.py`，从 W/F/J 中最好的出发，局部搜索按降序成本向量字典序改进）。
  - T128：EPLB / W / F / J+0 / L = R32 60.4 / 57.9 / 58.5 / 58.3 / 57.4；R64 60.8 / 57.1 / 58.4 / 57.1 / 56.4；R128 62.8 / 57.2 / 58.9 / 57.5 / 55.8；R256 67.2 / — / 60.3 / 57.7 / 55.9
  - T256：R32 98.5 / 94.4 / 95.3 / 94.5 / 94.0；R64 99.6 / 92.9 / 94.3 / 93.2 / 92.0；R128 102.6 / 93.3 / 94.1 / 92.4 / 91.3；R256 107.6 / — / 95.4 / 93.0 / 91.4
  - L 相对 EPLB：T128 −3.0 / −4.4 / −7.0 / −11.3，T256 −4.5 / −7.6 / −11.3 / −16.2 µs（R32/64/128/256；注意 L 直接优化这个成本模型，指标对它有利）。在线规划的上限在 R64–R128 饱和（T128 L：57.4 / 56.4 / 55.8 / 55.9）。
- token / block 取舍的离线诊断（EP32 真实 routing，bm96，`/tmp/front.py`、`/tmp/oracle_joint.py`、`/tmp/joint_ls.py`，placement 存 `/tmp/front_placements.npz`）：
  - 算法：J+dB = 每卡 block 上限 B = 下界 + dB，副本切片对齐 96（每专家只留一片零头），二分 L + 整数最大流；JL = J 之后再做局部搜索（只多改进 0.4–2%）。
  - T128，格式 token 不均衡 / 最慢卡 block：EPLB R128 静态 1.098/17.98；F 1.000/18.18；J+0 1.124/15.72（block 下界 15.65）；J+1 1.072/16.65；JL+1 1.066/16.65。R256（eplb640）、R64+64 热专家随机副本、R128+128 热随机副本的前沿与 R128 几乎相同（J+1 1.061–1.088 / 16.65–16.68）；均匀随机副本或在 R0 上加随机副本更差（R0+128 热随机 token 下界 1.255）。
  - T256：EPLB R128 1.139/28.00；F 1.000/25.97；J+0 1.045/23.70（= 下界）；J+1 1.032/24.70。
  - 结论：R128 起 token 下界 = 1.000、block 下界也各自可达，但两者不能同时达到；加冗余（R256）或随机加副本都不移动前沿 → 限制来自 block 粒度（T128 时每卡约 15 个 block 只装 768 token，平均每 block 约 50 token，冷专家零头不可拆），而非冗余或 placement。EPLB placement 本身已足够；随机副本不优于 EPLB。
- 只切热专家（`/tmp/hotsplit.py` H(k)：最热 k 个副本专家按 token 精确切分，其余副本专家整份按 token 贪心放）：R128 T128 token / block 峰值 / 激活峰值：H4 1.069/17.85/14.6，H8 1.049/17.70/14.6，H16 1.031/17.67/14.6，全切 1.000/18.12/15.1 → token 好、激活少，但 block 峰值不降（整份放置只看 token）。
- 组合 J + 热专家微调（`/tmp/hotpolish.py`：先 J 对齐分配整块平衡 block，再只对最热 k 个专家做非对齐挪动，block 上限不变）：R128 T128 J+0+hot16 1.089/15.72（block 贴下界），J+1+hot16 1.045/16.65；R64 T128 J+1+hot16 1.054/16.67；R128 T256 J+0+hot16 1.031/23.70，J+1+hot16 1.019/24.70。对比 oracle-W 1.087/17.49、oracle-F 1.000/18.18、EPLB 1.098/17.98 → 前沿明显前移。要点：多 block 的是非对齐切分，整块（96 的倍数）挪动不增加 block。
- 按热度选切分专家（用户思路）离线对比（EP32 R128 T128，格式 token 不均衡 / 最慢卡 block / 最慢卡激活）：oracle-H(k)（`/tmp/oracle_hot.py`：非最热 k 个的副本专家整体放到负载最低的副本，最热 k 个按 token 精确最大流切分）H8 1.047/17.70/14.6，H16 1.031/17.67/14.6，H32 1.013/17.73/14.8；给整体放置加 block 上限（`/tmp/oracle_hotb.py`）结果不变；HJ（`/tmp/oracle_hj.py`：最热 k 个按 96 对齐切）HJ32+0 1.090/16.03/14.7，HJ32+1 1.078/16.63/14.8。对照：EPLB 1.098/17.98/16.0，W 1.085/17.49/15.2，F 1.000/18.18/15.6，J+0 1.124/15.72/15.2，J+1 1.066/16.65/15.2。R64 T128：H16 1.088/17.80，H32 1.025/17.60；T256 R128：H16 1.048/25.98，H32 1.017/25.85。
- 结论：只切最热专家能在 token 很平（1.01–1.03）的同时少激活约 1 个副本、block 比 F 少约 0.5；但仍在同一条 token / block 取舍曲线上，block 降不到 J 的水平（最慢卡多出的 block 来自热专家切片和单副本冷专家，与中等专家放哪无关）。
- EP32 R256（eplb640，每卡 20 槽）模拟（`emul8.py`，最重 8 rank，.114，2 次；T256 两次差约 5 µs，噪声大）：
  - 离线 token / 最慢卡 block / 最慢卡激活（T128）：新 oracle R256 1.091/17.96/16.2，R128 1.084/17.45/15.3，R64 1.084/17.25/14.0；EPLB R256 1.124/20.67/20.0，R128 1.139/18.95/16.0。
  - 实测（µs，T128/192/256）：新 oracle R64 89.29/98.79/114.33，R128 89.79/99.16/114.63，R256 91.10/99.50/114.02；EPLB R128 93.03/106.19/121.40，R256 99.02/107.12/124.76。
  - 结论：R256 不比 R128 好（T128 慢 1.3 µs，其余在噪声内）；同场 R64 与 R128 相差 ≤ 0.5 µs，此前"R128 比 R64 好"（相差 0.5–1 µs，不同场次）在噪声内。EPLB 冗余越多越慢。
- EPLB R32 / R64 补测（.111，2 次，与 R256 场次不同主机；同场锚点 EPLB R128 / 新 oracle R64）：.111 原始值 T128/192/256：EPLB R32 90.61/102.29/116.50，R64 92.70/105.06/117.11，R128 95.07/106.86/120.94，新 oracle R64 90.84/99.85/112.95。两台主机锚点差（.114−.111）：T128 −2.0/−1.6，T192 −0.7/−1.1，T256 +0.5/+1.4。同场比较：EPLB R32 在 T128 与新 oracle R64 持平（−0.2），T192 / T256 慢 2.4 / 3.6 µs；EPLB 冗余越多越慢（R32 < R64 < R128）。
- 新 oracle R32 同场测（.111，2 次，`~/ep8watch/emu10_run.sh`）：T128/192/256：EPLB R32 90.34/102.35/117.36，新 oracle R32 89.06/100.94/115.58，EPLB R64 92.18/104.87/117.99，新 oracle R64 89.84/100.15/112.71。新 oracle R32 比 EPLB R32 快 1.3/1.4/1.8 µs；T128 时 R32 与 R64 的 oracle 相差 0.8（R32 快），T192 / T256 时 R64 快 0.8 / 2.9（R32 受 placement 限制：离线 token 不均衡 1.098/1.086/1.078，连续最优约 1.08）。

### 8.18 EP64 优化下界实测准备（2026-11-10）

- 新 EP64 集群（16 节点，物理 IP 10.44.175.138–171，容器 IP 见用户消息，part0 容器 IP 10.44.2.78）：`wt containers <ip>` 暂时返回无容器（WebIDE 启动中），主机名 / SN 未知，尚不能访问。
- 本地数据 `~/work/mpo/data/ep64x`（`ep32_prep.py <src> <out> 64 448,512,576`，已通用化 R 与 P）：routing_off / routing_eplb{384,448,512,576}，T=128/192/256；oracle：W（`ep64_oracle.py --weights=10,12`）与 `pkg/gcbench/oracle_variants.py`（F / J<d> / H<k>，任意 R）生成的 F、J0、J1、H16、H32 quotas。
- 离线（bm96，token 不均衡 / 最慢卡 block / 最慢卡激活；`offline_variants.json`）T128：EPLB R0 3.861/34.03/6，R64 1.191/13.97/7，R128 1.207/14.63/8，R192 1.197/15.18/9；R64：W 1.170/13.45，F 1.133/13.82，J0 1.182/13.27，J1 1.174/13.67；R128：W 1.091/13.17，F 1.009/13.76，J0 1.088/12.50，J1 1.078/13.05；R192：W 1.095/13.14，F 1.000/14.07，J0 1.087/11.93，J1 1.063/12.92（block 下界均值 11.88）。H16/H32 在 EP64 上差（每卡槽少），不纳入。T256 R128：W 1.055/21.17，F 1.006/21.88，J0 1.046/20.15，J1 1.038/20.99（下界 19.74）。
- 结论：EP64 上 J0 / J1 同时优于 W；R64 受 placement 限制（F 只能到 1.13）；R192 能让 J0 贴住 block 下界、F 到 1.000。
- 新 EP64 集群接入（task `L20B_88e0d6dd`，2026-11-10 20:05）：物理 IP / 容器主机名直连都失败；用 SN：`wt session start --container workdir e02-sg-bvv4rizlu05-<SN>` 后 `ssh <SN>-workdir`。part0–15 = a0003 a0014 a0005 a0006 a0018 a0007 a0015 a0010 a0017 a0002 a0008 a0001 a0011 a0016 a0013 a0004（容器 IP 见 `cluster_n.sh`；与平台列表顺序不同）。脚本：`cluster_n.sh`、`ep64n_deploy.sh`（part0 http :25280 扇出）、`ep64n_launch.sh`、`ep64n_queue.py`。
- 环境：容器无 deep_gemm → part0 用 gcc-toolset-12 + CUDA 13.2 从 dg341b.tgz 编译后扇出（`$W/ep64s/dg341`，_C md5 623fc1a3e0e3）；CuTe DSL 4.8.1 aarch64 cu12 wheels 装到 `$W/gcenv/site`。上传约 85–200 KB/s。数据在各节点由 `build_ep64.py` 重建（61 个文件，抽检 md5 与本地一致；EPLB routing 用 `eplb_regen.py` 按 gen 规则重建，本地验证逐字节一致）。
- 队列 `results/q_ep64lb.json`（4 gate + 18 配置 × 3 T × 2 次 + K=1 对照 = 114 次），20:45 启动，日志 `results/ep64lb.log`，结果 `results/ep64lb.jsonl`；ECO gate 448/512 已通过（逐位一致）。
- 21:30 `lb_eco576_T128_r0` 计时失败：rank 22 首发 illegal memory access（rank 13 unspecified launch failure），其余 rank combine readiness timeout。gate（40 次 eager check）通过 → 与此前 EP64 ECO 的低概率故障同类（本次 SPLIT_COUNTS=0），仅 ECO R192 出现，EPLB / oracle 无故障。
- 队列完成（114 条，23:50 前后；p50 中位数，2 次）：
  - T128：EPLB R0/R64/R128/R192 166.56/93.91/95.69/97.80；oracle R128 W 91.29、F 92.41、J0 92.00、J1 92.02；ECO R64/R128/R192 99.83/106.31/117.10（R192 2 次中 1 次 kernel 故障）。
  - T192：EPLB 218.94/112.44/113.64/114.57；oracle R128 W 106.91、F 107.33、J0 107.19、J1 107.12（R192 F 107.36）；ECO 114.54/120.06/129.67。
  - T256：EPLB 271.66/130.38/130.97/131.69；oracle R128 W 122.17、F 122.25、J0 122.03、J1 122.26（R192 F 122.17）；ECO 129.50/132.41/142.61。K=1 权重 EPLB R128 T256 129.59/130.44（K=8 130.97）。
  - 结论：EP64 下界 ≈ oracle R128（各变体相差 ≤ 0.5 µs）；比最好的 EPLB（R64）快 2.6 / 5.5 / 8.2 µs（T128/192/256）。ECO 机制开销 R64 7.9/6.4/5.4、R128 15.0/13.2/10.2 µs；ECO 仅在 T256 R64 比最好 EPLB 快 0.9 µs。

### 8.19 ECO 开销拆分（新 EP64 集群 trace，2026-11-11；`trace_breakdown.py`，dump 在 `/tmp/ep64tr`，结果 `results/ep64tr.jsonl`）

- 配置：ECO 与 oracle-W（同样每槽负载、无 ECO 机制）各跑 eager trace（32 call，按各 rank 的 barrier_done 对齐）和 graph trace。
- graph p50（ECO / oracle / 差）：R128 T128 110.21/91.90/+18.3；R64 T128 102.77/93.13/+9.6；R128 T256 136.45/123.17/+13.3。
- 拆分（µs）：barrier 之前 +2.3/+1.4/+1.9（graph barrier_done）；barrier 之后到最慢 rank 结束（eager）+10.9/+3.3/+5.5；剩余仅 graph 出现的跨 call 部分 ≈ +5.1/+4.9/+5.9（三个配置都约 5 µs，来源未定位）。
- barrier 之后的差来自关键 rank（副本多的 rank）的 role2 晚结束：R128 T128 关键 rank role2_end +12.7（其他 rank +3），role3（combine）本身不变；所有 rank role1_end +2.6–4.1。plan：auction 在 barrier 后 2.6–3.3 µs 结束、4.1–4.9 µs 发布；role1 等 plan 的跨度中位数 7–9 µs；role2 最后一个副本槽的 plan 行在 barrier 后 23.6 µs（R128 T128）/ 24.8（R64）/ 8.4（R128 T256）才看到。graph 模式 role1 每 warp 等 plan 2.0–3.6 µs，role2 每 CTA 2.7–4.6 µs。
- 新增 trace（只在 GC_MOE_ECO_TRACE=1 时编译进去）：ring 字段 8 role2_end、9 role3_end、10 plan kernel 退出、11 call 周期（本 call 与上一 call 的 role1 start 之差）；bench 的 ring 读取原按 64×8 解析（实际每 call 16 字段，旧 per_call_us 错位），已改为 64×16。
- graph ring（R128 T128，2 次一致；所有 rank 均值，ECO / oracle）：barrier 17.2/14.8，role1_end 48.6/39.9，role3_start 74.1/65.0（最慢 rank 88.9/71.3），role3_end 95.2/77.5，周期 110.4/92.3；role3_end 到下一 call 开始的间隔两者都约 15 µs → 之前推测的"graph 独有约 5 µs 跨 call 开销"不存在，ECO 多出的 +17.7 µs 全在 call 内部，主要是最慢 rank 的 role2 晚结束 +17.6 µs。plan：输入齐 8.6、发布 23.4（barrier 后约 6 µs，planner 从输入到发布约 15 µs）、plan kernel 退出 23.9；role1 看到 plan 平均 27.2、最慢 rank 37.9（barrier 后 10 / 21 µs）。
- 发现 graph 回放中下一 call 的 role1 会在上一 call 的 role2 结束前开始（role2_end 字段出现回绕的负值），说明 call 之间本来就有重叠；role2_end 字段需按 call 序号而非全局 slot 33/24 记录才可用。
- 现有 `GC_MOE_ECO_SPEC=1` 探针（副本按上一 call 的 plan 拉取、不等 plan，输出错误）在新 EP64 上不是上界：p50 R128 T128 120.02（正常 ECO 106.25），R64 T128 117.16（99.61），R128 T256 181.97（132.54）。原因推测：过期区间折回本 call 顺序、超界 token 落到 rank 0 的同一地址形成热点，探针本身有病理开销，不能用来评估"预取"收益。
- 下一步先测缩短 planner：GC_MOE_ECO_UNIT_CFG 的 auction 轮数（base / 1 轮无 two-hop / 0 轮只用 warm 起点），队列 `results/q_ep64rounds.json`。
- auction 轮数测试（`results/ep64rounds.jsonl`，p50 两次均值；base = 现行 warm 2–3 轮 + two-hop 2 轮，r1h0 = 1 轮无 two-hop，r0h0 = 只用 warm 起点）：R128 T128 106.62 / 102.00 / 101.98；R64 T128 99.71 / 96.38 / 96.42；R128 T256 132.78 / 127.84 / 127.85。少轮数快 3.3–5 µs，1 轮与 0 轮无差别。对照同集群 oracle-W（91.29 / 91.91 / 122.17）：机制开销降为 10.7 / 4.5 / 5.7 µs（原 15.3 / 7.8 / 10.6）。对比最好的 EPLB（T128 R64 93.91，T256 R64 130.38）：T128 仍慢 2.5 µs（ECO R64），T256 快 2.5 µs（ECO R128）。
- 默认改动（2026-11-11）：`eco_unit_config` 在 EP64（num_ranks > 32，time 目标）改为 wb,wa = 10,12，auction 1 轮 warm、无 two-hop（`(96, 10, 12, 1, 6, 0)`）；`ep64_oracle.unit_config` 同步。gate 4 项（R64/R128/R192 T128、R128 T256，各 64 步）逐位一致。默认路径（无环境变量）p50：R128 T128 102.39，R64 T128 96.07，R128 T256 128.21（旧默认 106.6 / 99.7 / 132.8）。
- 踩坑：`ep64n_deploy.sh` 原来按文件大小判断 part0 上的 tar 是否最新，tar 大小恰好相同导致旧代码未更新（df / df2 队列的 "default" 测的是旧代码，作废）；已改为 md5 比较。part0 上 28005/28006 被平台 fuse_daemon 常驻占用，QUEUE_PORT 不要落在这两个端口（df3 有两次 EADDRINUSE）。

### 8.20 同层 spec 实验：预取方向证伪（2026-11-11，`results/ep64same.jsonl`）

- 目的：让 `GC_MOE_ECO_SPEC=1` 探针的"上一次 plan"变成合法的同层 plan，从而给"按上一次同层 plan 预取副本"一个干净的上界。方法：`routing_same.npy` = 同一层 routing 重复 1200 个 call（shape (1200,64,128,6)，md5 3f3af8100310，16 节点就位）。plan 逐 call 收敛到定点，spec 读到的过期 quota 基本正确。
- 结果（bm96，权重 10/12，split=0，chunks 10）：正常 ECO kernel_us 119.02 / p50 119.89；spec 117.78 / p50 123.39。ring（µs，38 call，均值/最慢 rank）：period 115.6/120.1 vs **121.7/126.5**；role1_end 44.2/70.5 vs 45.6/74.9；role3_start 68.6/94.0 vs 70.1/100.4；role3_end 100.5/103.6 vs 106.6/110.3。plan：输入齐 8.8、发布 16.1；r1_plan_seen 均值 35.8、最慢 66.5。
- **结论：预取证伪，不再实施。** 三个原因：
  1. **拉取侧根本不在等 plan**：plan 16.1 µs 发布，而 warp 平均 35.8 µs 才到副本区（忙着拉单副本 token）——plan 早到约 20 µs，没有可省的空等。
  2. **spec 反而慢约 6 µs**：planner 平衡的是本 call 的 quota，过期 quota 按定义更不平衡 → 关键 rank 拉取得更晚（+4.4），顺延到 role3（+6.7）。同层 (start,q) 完全一致率只有约 65%，错配的 35% 槽直接把负载挪到错的 rank。
  3. **正确的暂存+补拉实现也救不回来**：它只能修复第 2 点，却要付出命中行的本地拷贝（约 90% × 426 token/rank 的搬移）和逐行校验的代价；预取只改变拉取顺序、不减少拉取量，而拉取引擎是吞吐受限的——收益上限≈去掉 plan 等待的约 1 µs。
- 推论：role1 侧剩余的 ECO−EPLB 差 ≈ barrier 前约 2.4 µs（计数交换、表发布）+ 副本路径每 token 的额外读（quota/start 加载、phys2log 查表）约 2–3 µs；role2/3 尾部约 +4 µs（关键 rank 副本块在 plan 发布后才调度）。下一步候选：① 副本块在 role2 里优先调度 / plan 更早发布；② barrier 前计数表瘦身或与 routing 重叠；③ 副本拉取的 quota+start 合并为一次加载、phys2log 缓存。
- 备注：同层 workload 本身更重（period ~116 vs 轮换 routing ~107），combine 占大头；不能与正常 workload 跨比较，只有同场 eco-vs-spec 有效。

### 8.21 副本计数并入 native recv_count 行（方案 B，2026-11-11 实现，待上机验证）

- 动机：ECO 对 EPLB 的剩余差距 ≈ barrier 前 ~2.5 µs + 副本拉取路径每槽入口读 planner 打包表（10-bit、带 epoch tag）的 ~2–3 µs + role2/3 传导 ~2.5 µs。副本槽入口那 64 个源计数在语义上就是 native `expert_recv_count` 的同构数据（源 rank 发往该槽逻辑专家的 token 数），不需要以 planner 表的形式存在。
- 实现（三个文件）：
  1. `gc_moe/eco/eco_unit_kernel.cu`：新增 `fill_replica_counts`（kernel 头部 lambda）。index warp（warp 16–17，split 下 planner CTA 的闲置 warp；非 split 下 send_rows 后空闲）每 lane 认领一个源 rank，轮询本地 count 矩阵该源行的 tag 后，把本 rank 副本槽 `[num_fixed, kS)` 的 phy2log 专家计数写成 native `expert_recv_count` u64 行（`gen*(R*kS) + src*kS + l`，双 generation）。不写 `expert_recv_count_sum`（role3 的 quota 语义不变）。调用点：split 路径 index warp 提前 return 处、非 split 路径 index warp return 处。
  2. `gc_moe/eco/plan.py`：defines 新增 `OFF_RECV_COUNT`。
  3. `gc_moe/impls/sm100_fp8_fp4_gc_moe.py`：两处 pull 点（`_pull_pipeline` 与非流水 fallback）删掉 `_eco_table_count` 分支，副本和 native 统一走 `ld_global_u64(expert_recv_count[src][slot])`；删 `_eco_table_count` 方法、`eco_counts_bytes` 属性、`phy2log_smem` 参数与 smem 写循环（`copies_pk` 分配缩回 `route_experts*4`）。`jit_kernels` 注释同步更新。
- 正确性论证：transform 的写入以 count 矩阵到达为界，而 plan 发布本身就以全矩阵为门槛——transform ⊆ 矩阵到齐+ε ≪ plan_ready（~16 µs）；role1 的读取全部发生在 plan acquire 之后（acquire 使 L1 失效，读到 L2）。被动槽 quota=0 永不进入 pull，行内容无读者；finalize 清零覆盖所有槽，与 native 行同生命周期。spec 路径不再有 tag 轮询（探针可接受）。
- 本地验证：nvcc 编译 sm_100a 通过（split 与非 split 两变体）；ast 语法检查通过。
- **集群恢复经过**：`cluster.sh` 的 cc54rhpcz03 job 容器集体消失后，实际存活的是 `cluster_n.sh`（bvv4rizlu05，task 88e0d6dd，part0=a0003）。工具链对应 `ep64n_*`：`ep64n_deploy.sh` 全 repo tar 到 `$W/ep64s/pkg/gc-moe`、`ep64n_queue.py`（PREFIX 硬编码，包路径 `$W/ep64s/pkg/<pkg>`）。**老 `ep64_queue.py`/`ep64_deploy.sh` 指向死掉的 cc54rhpcz03，勿再用。**
- 上机验证（rc2 包，digest 8f4f3f0c）：eco512/eco448 `--check` 逐位一致；非 trace 计时 R128 T128 kernel 96.2/96.5/96.6/97.4、p50 102.03–102.26（旧包 98.7/102.39，噪声内偏好）；T256 kernel 121.4–121.7、p50 127.6。
- **trace 异常**：graph ring 显示 pull 相位一致变慢——rc2/rc3 三次 r1_end 均值 51.6–52.1、period +1.9 µs（旧基线 nd_trg512 r1_end 46.4、period 105.2；同代码旧样本 trg2 为 49.0–49.2 但 planner 配置不同 plan=24 vs 18）。bar→plan_seen +1.5、seen→r1_end +3.5、role2 副本输入等待（ring f15）+3.3、period +2 µs。
- **bisect（rc4 包）**：加 `GC_MOE_ECO_RECV_COUNTS` 开关——0=副本回读 planner 打包表（`ld_volatile_u32`+tag 轮询，忠实复现旧路径；transform 写入仍跑，属冗余但正确）、1=读 native recv_count 行（B 原样）、2=同一 recv_count 行但 `ld_volatile_u64`。开关在 impls 的 `_pull_pipeline` 与 fallback 两处；`_eco_table_count`/`eco_counts_bytes`/phy2log 的 meta 直读都恢复了。结果（交错两轮）：**flag=0 p50 107.18/107.25、r1_end 48.3–48.7 ≈ 旧基线；flag=1 p50 109.11/109.15、r1_end 51.8 ≈ rc2/3**——transform 写/poll 无罪，**慢在 recv_count 读取路径本身**。flag=2（同地址 volatile 读）用于区分"区域慢"还是"L1 缓存行为慢"。
- **flag=2 结果（rc5）**：p50 109.60、r1_end 52.0 ≈ flag=1——排除 L1 缓存行为，**慢在 `expert_recv_count` 区域本身**（与 routing 远端写洪流、per-token `l1_full_count` 远端原子共享 L2 热区）。
- **flag=3 修复（rc6/rc7，默认模式，方案 B 终态）**：transform 改写目标到 ECO 区内新增的 `eco_recv_counts` 专属区（`layout/gc_moe.py` 的 `eco_recv_counts_offset`：紧随 count 表 + route offsets 之后，[gen][src][slot] u64 与 native 同构，2 gens × R × kS × 8B = 8KB@P512）；pull 用 `ld_global_u64` 单发读取。kernel 写目标随 `ECO_RECV_MODE` define（plan.py `recv_mode` 参数，lru_cache 键内；jit_kernels 与 impls 同读 `GC_MOE_ECO_RECV_COUNTS` env，**默认 3**）。结果（graph ring 中位，交错验证）：**flag=3 p50 105.86/106.37、r1_end 44.5/44.9、r2_wait 4.7/5.0——比 flag=0 的 r1_end 47.4–47.6 还快约 2.5 µs**（省去逐源 tag 轮询与 phy2log meta 直读），flag=0 对照本轮 106.71/47.8。生产路径定为 flag=3：u64 统一格式、无打包解码、无 tag 轮询、单次普通加载；mode 0–2 留作 debug 开关。
- **rc7 终态验证（默认 mode 3，无 env）**：chk512/chk448 逐位一致；非 trace p50 T128 101.89/102.36（kern 96.76/97.38）、T256 127.98/128.05——与旧包噪声内持平；默认 trace p50 106.23、r1_end 46.2。方案 B 落地完成：副本计数由 planner 物化为 u64 行，pull 走统一 ld_global 路径，比原 packed 表路径略快。

### 8.22 归因实验：publish 截止约束 + pull 残差定位（2026-11-11，q_attr1.json）

- **目的**：分离 ECO−oracle-W 剩余 ~4.7 µs pull 残差的归属；检验 publish/barrier 优化候选。
- **4 个 trace run**（bm96 P512 T128，同包 rc7 默认 mode 3，交错）：
  - `ap_r0`：`GC_MOE_ECO_UNIT_CFG=10,12,0,6,0`（0 warm 轮）→ p50 105.78；**publish 提前 4 µs（13.9 vs 17.9–18.1）但 plan_seen/r1_end/period 全部不变**
  - `ap_r2`：`UNIT_CFG=10,12,2,6,0`（+1 warm 轮）→ p50 107.89；**publish 推到 ~21 µs，亏 ~2 µs**
  - `ap_ora`：oracle-W plain 同负载 → p50 93.35；barrier 14.9、pull 时长 24.9、r1_end 39.8、period 92.1
  - `ap_pub`：`GC_MOE_ECO_ROWS_AFTER_PUB=1` → p50 115.81；plan_input 9.1→19.9、publish 17.9→28.3、r1_end 44.5→52.7；**barrier 纹丝不动（16.0 vs 15.9）**
- **结论**：
  1. **publish 有 ~19–20 µs 的截止约束**：早于它零收益（r0），晚于它 ~1:1 亏（r2，约束源是 role2 副本调度边界而非 role1 plan_seen）。**发布压缩（提前发布）这一优化方向已排除**；只需保证 publish 不超截止。
  2. **barrier 前 +1.2 µs ≈ 副本 index 扇出固有成本**：ap_pub 把 count 行推送推迟 10 µs，barrier 完全不动 → egress 竞争对 barrier 的贡献 ≈ 0；残余基本是扇出的额外远端写，**结构性不可收**。
  3. **pull 残差 +3.7 µs 是每 rank 固定相位开销而非 per-token 税**：ECO pull 时长 28.5–28.7 vs oracle 24.9；64/64 rank 全慢（+1.5~+5.8，中位 +4.0），且 delta 与 rank 负载**负相关**（corr −0.46）。per-token 路径副本/native 逐位同构、drain 无 eco 分支——已排除代码差异，需要仪器定位。
- **下一步仪器（probe 包 gcm-eco-probe.tgz，impls md5 83f24676）**：
  - `GC_MOE_ECO_RECV_COUNTS=4`：副本入口跳过计数行加载、常量计数——**输出错误，仅计时探针**
  - ring 字段 4（原空闲）：`_pull_pipeline` 内 `rep_entry_ns`（`entry_t0`→首次 pull 发射边界的累计副本入口 setup ns，`eco_trace` 门控）
- **事故与恢复**：部署 probe 时误漏 DEST 参数。`ep64n_deploy.sh` 里 `rm -rf $W/$DEST` 为空会展开成 `rm -rf $W/`——但该命令在 `&&` 链上、被 md5 校验门控，part0 上 tar scp 失败故 rm **未执行**；`ep64s/` 实际消失于 ~20:42 的容器层重置（`.sub2api-webide`/`runs` 等均 20:42 重建；tgz、`gcenv/` 在持久层幸存）。**教训：`ep64n_deploy.sh` 调用必须显式给出非空 DEST（布局：repo tar → `ep64s/pkg/gc-moe`），容器随时可能重置，跑实验前要先核对各节点 `ep64s` 完整性。**
- **恢复清单**（part0，其余 15 节点未受影响）：`ep64s/pkg`（=rc7, md5 3ce30a0a ✓）、`dg341`、`tr`、`cache` 从 part1 tar 中转恢复；`ep64s/data` 仅恢复实验所需 npy（routing_off/_T256/_T192、eplb512、ora512b96c10x12、eco{448,512,576}_metas{,_phy2log}）；`gcenv/` 未受损；`ep64s/env`/`small`/build.log 为安装残留未恢复。`$W` 持久层 vs 容器层的边界不明——**重要产物必须在 `$W` 外留本地副本**。

### 8.23 单副本下限 placement（eco512sf，2026-11-12，显著收益）

- **动机**：idx10 trace 分析确认 role2 尾部挂在"副本任务排最后 + quota 门控"上；副本专家的 canonical 槽也等 plan，真正无门控的只有 count==1 专家的槽。**关键指标 = 每 rank 单副本专家 token 量（ungated work）**，只要它覆盖 plan 窗（~19–25 µs ≈ 5–6 个 bm96 块）GEMM 就不饿。
- **生成器**：`pkg/gcbench/gen_eco_floor.py`（在节点上跑，数据目录作参数）。两阶段：
  - Phase A：合并最冷的副本专家（其多余槽改给"该 rank 缺席 + 每副本负载最高 + nc<32"的已复制专家），直到 U 总量够 τ=0.40 floor；
  - Phase B：欠载 rank 拿副本槽与富余 rank 的单副本专家交换，直到 min U ≥ τ·768；
  - 最后 `permute_local` + `EcoLayout.meta` 产出 `eco512sf_metas{,_phy2log}.npy`（16 节点 md5 f9798700）。
- **离线验证**（参考 planner 重放，bm96）：tokens peak 852.9→844.9、imb 1.111→1.100；blocks peak 13.27→13.18、imb 1.155→1.147；cost-model time 1066→1054。**block 峰值不降反升**（合并冷副本减少碎片）。实测 ungated tokens/rank：min 70（9%）→ **316（41%）**，ungated 块 min 2.5→5.4。rep_experts/layer 降到 31–93。
- **上机结果**（q_sf.json，同包 idx10 代码，bm96 T128，SPLIT_COUNTS=0）：chk 逐位一致；**计时 est 89.31/89.01、p50 ~94.2**（同代码旧布局 97.66/97.25、p50 ~103；oracle 86.3；eplb512 历史 ~90.35）——**−8.3 µs，且 ECO 首次快过 EPLB512、距 oracle 仅 ~3 µs**。sf_t3 est=119 为污染离群（p50 94.26 正常），作废。
- **机制验证**（sf_f1 ring）：role2_stall_per_wait 2.2–2.7 → **1.3 µs**（GEMM 等 plan 基本消失）；r3e 均值 91.6 → 85.1（straggler 尾巴收掉）；plan_published 18.9 不变；r1e ~49 持平（远端 canonical 读的 ~+5 µs 固定开销仍在）。
- **推论**：placement 层下限约束是目前性价比最高的优化——零运行时成本、顺带改善 block 峰值。剩余 ~3 µs 差距来自远端 canonical index 读的 fabric 开销，方向 = source-partition 或 pull 路径优化，不在本方案范围。

### 8.24 副本行本地化实验全灭：远端读毒性定性（2026-11-12，idx11–idx14）

- **目标**：消掉 sf 之后剩余的 ~9 µs pull 窗差距（vs oracle），让副本槽 index 读本地化。
- **实现**：plan kernel 的闲置 index warp 在 rank barrier（`eco_idx_ready`）后把 canonical source-index 行拷进本 rank 副本槽自己的行（32 KB/槽），`eco_reps_ready[rank]` 发布；pull 统一读本地行。
- **实测**（bm96 T128，eco512sf metas，SPLIT_COUNTS=0）：

| 变体 | kernel_us_est | 结论 |
|---|---|---|
| idx11 车道式 `ld.volatile.v4` 远端读 | 8973 | role1_end ~4.7 ms，灾难 |
| idx11 改普通 `src[v]`（c1_t1） | 8982 | 同样灾难——**与 volatile 无关** |
| idx12 零填充探针（无远端读，输出错） | 127.3 | flag 机制本身只值 ~38 µs 开销 |
| idx13 TMA 中转（远端→smem→本地，逐位正确） | 426.3 / 426.5 / 434.2 | 好转但仍 5× 于基线 |
| idx14 远端 `prefetch.global.L2` | 1368 / 1384（另 1 次 combine timeout） | prefetch 也是远端读，同毒 |

- **定性结论**：pull 窗口内任何**批量/投机性远端读**都毒药化——唯一不毒的模式是 idx10 的逐 token 4B 远端读（与真实工作交错、每个都是必需）。行复制方向的收益上限 < 成本：4 MB/call 的行复制流量挤在 pull 窗里，无论读侧怎么实现都被 canonical rank 链路竞争放大。
- **历史对照**：更早的 post-barrier push/pull 扇出也均差于基线（162 / 119 vs 105 µs）。至此 **pull 远端读、push 扇出、车道拷贝、TMA 中转、L2 prefetch 五个方向全部证伪**。
- **当前状态**：代码回滚至 idx10 部署版（89.0/89.3 µs，优于 eplb512 ~90.4，距 oracle ~3 µs）。剩余差距 = 远端 canonical index 读的 fabric 固有税，只有结构性方案（source-partition）或更低副本 token 占比的 placement 能再压；后者与 block 峰值目标冲突（副本必须集中在热专家才能控峰，rep~55% 接近下限）。

### 8.25 源侧扇出（plan kernel index warp 重放 routes → 远端写副本行）证伪 + 关键上限发现（2026-11-12，idx15–idx16）

- **设计**：plan kernel 的两个 index warp 在等 count 的空窗里重放 `routes[]`，把每个 index entry 直接远端写进该专家所有活跃非 canonical 副本的行（per-expert `fan_cursor` 定位，段 = (slot×R+src)），`__threadfence_system` + `index_sync` 后向全部 64 peer 的 `eco_reps_ready[src]` release 写 epoch+1；pull 侧在 `_eco_load_plan` 等齐 64 个标记后统一读本地行。零 pull 起步延迟、零远端读、保留全部在线 quota 灵活性。
- **实测**（bm96 T128，eco512sf metas，SPLIT_COUNTS=0，FANOUT=1 全量写 / =2 只发 flag）：

| run | mode | kernel_us_est | 说明 |
|---|---|---|---|
| fo_t1/t2/t3（idx15） | 写 | 108.32 / 107.83 / 107.0 | +19 µs vs 基线 |
| fo_f1 trace（idx15） | 写 | 111.19 | pull 窗 43.9 vs sf 33.2 µs，全 rank 均匀 +10.7 |
| f2_chk / f2_t2（idx16） | 写 | failed ×2 | combine readiness timeout（随机 rank LAUNCH_FAILED，与历史 flaky 崩溃同形态） |
| f2_t1（idx16） | 写 | 110.57 | 复现 |
| **m2_t1 / m2_t2（idx16）** | **flags-only** | **89.84 / 91.72** | **≈ 基线——本设计天花板** |

- **两条结论**：
  1. **远端写也有事务税**：~840 个散落 4B 远端写/rank 在 pull 窗内消耗 ≈ +20 µs（≈23 ns/写），均匀摊在所有 rank——写方向不毒死但绝不免费，dispatch 窗口内 egress 与 pull 读请求/写共存时被排队放大。
  2. **上限 = 基线，方向无收益**：flags-only（本地副本行读 + 协议，零写流量）只到 ~90 µs ≈ idx10 基线 → **远端 canonical 逐 token 4B 读本来就几乎免费**（与 payload 拉取交错重叠充分），sf 距 oracle 的 ~3 µs 不在它们身上。任何给副本行供数的机制最优只能打平基线，有成本即亏。
- **附带发现**：m2（无写）2/2 过、f2（有写）3 次里 2 次撞已知概率性崩溃——样本太小不能定罪，但与 idx15（有写）5/5 过并存；留档。
- **当前状态**：3 文件回滚至 `gc-moe` 基线包逐字节一致（kernel f1e1fa09 / impls c2a518d8 / plan.py 07b65ff1）。**副本行本地化方向全部关闭**（扇出读/写两侧共 7 种实现实测全灭或打平）；sf 之后的 ~3 µs 归因应重查：候选为 plan 等待尾部、quota 装载、combine 端，而非远端 index 读。

### 8.26 pull 循环相位劈分 + canon 预载（2026-11-12，EP8 本地验证通过——pipeline 路径）

- **背景归因**（ix10_f3b vs ap_ora ring，40 call 均值）：pull 窗 33.8 vs 25.1 µs 差 +8.7，且与副本 token 数基本无关（rep 209–381 跨 rank pull 只差 ~1 µs）。按 `pull = a·one + b·rep` 重分解：**固定槽 token ~+9ns/个（~4.2 µs）+ 副本 token 专属 ~+15ns/个（~4.5 µs）**。已排除项：远端 index 读带宽（m2）、count 行（rc4）、plan 门控（reps ~26 µs < 实际需要 ~32 µs）。
- **实现**（impls/sm100_fp8_fp4_gc_moe.py，未编译验证）：
  1. `_pull_pipeline` 单 `while` 劈成 `for eco_phase in range_constexpr(2)`——phase-0 循环 `while current_expert < eco_num_fixed`，const 特化出与 plain 逐 bit 相同的瘦循环体；phase-1 副本段。`token_idx >= eco_fixed_tokens` 每迭代门控整段消失。
  2. 相位间无条件 `_eco_load_plan` + `advance_expert` 全量刷新（副本 counts 此前为 0，walk 到 local_experts 后从 -1 重走）。
  3. **canon 解析上移每 call 序幕**：`eco_canons` rmem 打包 `(le<<16)|gslot`（phy2log+copies 两次依赖加载从每副本槽边界 ~2 次全局读降为一次 shuffle）；slot 边界一次 `cached_num_tokens` 解包。
  4. **统一 index 基址 `eco_idx_ptr`**：每槽一次 `make_ptr(mapped_peer_base(canon_rank)+size_off+canon_slot*N*cap*4)`（peer 自映射覆盖本地 canon），per-token 读从 ~10 条指令 + 分支降为 `eco_idx_ptr[src_rank*cap+src_slot]`。
  5. count 行各模式分支 `current_expert >= eco_num_fixed` → `const_expr(eco_phase==1)`；L2 prefetch 目标改 canon 行（原对副本槽预取自身空行=死预取），远端 canon 跳过。
  6. fallback 非流水线环同步改造：`eco_idx_ptr` 每槽统一基址（固定槽=自身行），quota_start 改走 `expert_starts` 缓存（原直读 signals）。
- **预期**：消掉均匀税大部（~3-4 µs pull 窗）+ 副本 per-token 税部分（~1-2 µs）；est 口径收益按重叠折算 ~1-2 µs。
- **待验证**：EP64 容器回收、B300 .114 ssh 不通——AST 解析通过、逐行审查一致，但 `range_constexpr` 内套动态 `while` 的 DSL 展开、peer 自映射 VA 路径尚未编译/上机。恢复后流程：远端编译 4 变体 → idx17 打包部署 → chk + sf 计时 + trace 对照 pull 窗是否回 ~28-30。
- **EP8 本地验证**（8×L20D 单机 NV18，`benchmarks/benchmark_gc_moe_eco.py`，T=128，SPLIT_COUNTS=0，calls=8 iters=5 inner=20）：
  - 新代码 eco416 **check max|diff|=0.0**，176.71 vs 基线 176.41（Δ+0.3 噪声内）；ora416 176.21 vs 176.41。
  - `range_constexpr` 内套动态 `while`、peer 自映射 `eco_idx_ptr`（本地 canon 路径在 EP8 已被真实命中——每 rank 4 副本里部分 canon 在本 rank）、`eco_canons` 打包/shuffle、phase-1 `advance_expert` 用装载后 quota 重定位——全部通过编译且逐位正确。
  - EP8 上 eco416≈ora416（机制税本来就≈0），所以本地只能做正确性+非回归验证；pull 窗收益要回 EP64 量。
  - **EP8 验证矩阵全部通过（2026-11-12，conda310 + DSL 4.8.1 + deep_gemm tileapi，均 calls≥2 check-steps=首call eager 比对）**：

| 配置 | env | check | 说明 |
|---|---|---|---|
| pipeline 默认 | — | eco416/ora416 0.0 | 相位劈分+eco_idx_ptr+eco_canons 全链路逐位正确；eco416 176.71 vs 基线 176.41（噪声内，EP8 机制税≈0 无可收） |
| fallback | GC_MOE_PULL_PIPE=0 | eco416/eco448 0.0 | 串行环同样改造：per-slot eco_idx_ptr（固定=自身行/副本=canon 行）、expert_starts 取 quota_start |
| recv0 | GC_MOE_ECO_RECV_COUNTS=0 | 0.0 | phase-1 const `_eco_table_count` 分支正确 |
| recv2 | GC_MOE_ECO_RECV_COUNTS=2 | 0.0 | phase-1 const volatile count 行分支正确 |
| recv4 | GC_MOE_ECO_RECV_COUNTS=4 | —（按设计输出错）| 编译+跑通无挂死（const-count 探针分支） |
| spec | GC_MOE_ECO_SPEC=1 | —（按设计输出错）| 编译+跑通（eco_spec 路径含 slot_total/stored_rank_count 编辑区） |
| trace | GC_MOE_ECO_TRACE=1 | 0.0 | 相位边界处 trace 门控（token_idx>=eco_fixed 残留全部为 trace-only）编译+正确 |
| eco448 | — | eco448/ora448 0.0 | 8 副本/rank 的 phase-1 重负载正确；174.99 vs ora448 173.93 |

  - EP8 结论：**正确性全绿 + 无回归**。EP8 上 eco≈ora（8 源 NVLink 全互联、副本少，canonical 读全是近端），pull 窗收益量级只能回 EP64 验证（64 源、远端 canon 占比高、副本 ~300 tok/rank）。
  - 待集群恢复后：部署同 md5 包 → `--check --check-steps 40` → eco512sf 计时对照（预期 pull 窗 33.8→~28-30 µs，est ~1-2 µs）→ trace 对照。

### 8.27 集中式副本 placement（少专家×多拷贝）离线验证 + relief-pair 上限（2026-11-12）

- **动机**（用户提出）：单副本 token 占比最大化 + 极少副本专家 + 高冗余度 → 同 block 峰值下降低副本 tax。
- **生成器** `~/work/mpo/ep64/gen_concentrated.py`（EP64）/`gen_conc32.py`（EP32）：top-K 热专家 × ⌈mass/(0.45·lvl)⌉ 拷贝（互异 rank），单副本 LPT 切分，`permute_local` 打包。
- **EP64 回放**（200 call，eco512sf 对照）：c512k16/k32（16/32 副本专家）tok imb 1.117-1.123、**blk peak 11.07-11.15（不劣于 sf 11.16）、time imb 1.097-1.101 ≈ sf 1.096**；rep_t 份额 17-27% vs sf ~39%。
- **EP32 回放**（200 call）：集中式**严格优于**现有 metas——c448k16 time peak 7473 vs eco448 7538；tok imb 1.13 vs 1.16；blk peak 20.0 持平。候选：**eco448c16 / eco512c16 / eco512c32**。
- **meta 格式硬约束（新发现）**：`EcoLayout` relief pairs ≤ 4×P（P512→2048, P576→2304）。高浓度（K 少×拷贝多）使 pair 数 ~Σ copies² 超限：**P576c16 需 2604、c576c32 需 2480 均 FAIL**。推论：集中式可用的拷贝上限 ≈ √(4P/K)，EP64 上 c576c16（16×13=2704）同样不可用——高冗余方向必须搭配更多副本专家（摊薄 copies²）而非更少。
- **产物**：EP32 metas+phy2log 已生成 `~/work/mpo/data/ep32/eco{448,512}c{16,24,32,48}_metas{,_phy2log}.npy`；EP64 phy2log `eco{512,576}c{16,32,48}_metas_phy2log.npy`（576 系仅 k32/k48 可能过 meta）。
- **待集群验证**：EP32 8×4 L20B（ep32_queue.py 就绪）→ chk → 计时 → trace。

### 8.28 eco2 重排放置 + HydroPlanner（用户迭代版，blk peak 贴住理论地板）

- **placement `~/work/mpo/ep64/eco2_place.py`**（每层独立，`stats_hist.npy` 剖面）：
  1. 副本数按需比例：`c_e = ceil(p95块需求_e/τ)`，τ 二分使总数 ≈ P——热专家自然深副本、冷专家单副本（K≈40-150 由数据决定）。
  2. 固定先行：全部单副本专家 profile-LPT（逐步剖面装入"放后全局峰最小"的 rank，预留均匀副本槽）→ 各 rank 基底均衡、同相爆发错开、每 rank 保底单副本任务。
  3. 副本落位（用户方案，替代 anticorr）：**min 期望负载 rank × 随机待放专家**，charge `nb_mean_e/c_e` 分数块，约束同专家不重复 rank + 有空槽。种子固定 rng(0)。
  - 注：anticorr（⟨nb_e², load⟩ 打分）在 r64 上曾优于标量贪心 ~0.5-0.9，但用户方案实测**更简单且更好**（r64 bm192 −0.06~-0.10，r32 全打平）；纯随机在 deg≥12 时会撞重复 rank。
- **在线 planner `~/work/mpo/ep64/hydro.py` HydroPlanner**（stateless，逐 call）：单副本钉死；多副本按需求降序 `ceil(n_e/bm)` 整块水填到当前最少块副本 rank → relocate ≤8 遍 → 残余峰多源 BFS 增广（≤64 链、深度≤8，等价 maxflow 精确最优）。~4-5k 标量操作/call。
- **实测 blk peak**（48000 calls，meas_calls 流；括号内为 anticorr 旧值）：

| | bm96 | bm128 | bm192 |
|---|---|---|---|
| EP64 P512 | 12.00 | 10.00 | **8.03**（8.13） |
| EP64 P576 | 11.99 | 10.00 | **8.02**（8.08） |
| EP32 P512 | — | 14.25 | **13.02**（13.02） |
| EP32 P576 | — | 14.25 | **13.02**（13.02） |
| EP32 bm64 | — | — | P512/576 19.30 |

- **差距分解**（EP32 bm192）：LB_vol 12.61 → 真地板 13.02（整块税+逐 call 波动，布局无关）→ 实际 13.02 = 地板；EP64 bm192 同形（8.0x vs 地板 ~8.0）。P416/448 的 ~0.5 残差为资格约束+预算稀薄，用户决定不优化（P512/576 够用）。
- **产物**：`~/work/mpo/data/real1500_vllm_dsv41/r{32,64}/eco2_place{P}_bm{bm}.npy` 全部用新方案重生成。
- **待办**：HydroPlanner 的 CUDA plan kernel（`gc_moe/eco/eco_unit_kernel.cu` 的 `UNIT_HYDRO` 分支）bit-exact 调试中——counts 交换/元数据/publish 复用原管线，plan 阶段换成水填+relocate+BFS。

### 8.29 HydroPlanner CUDA kernel：全量 bit-exact + 延迟优化至 ~36k（2026-11-14）

- **正确性**：`hydro_role_test.py`（`ARCH=sm_103a GC_MOE_ECO_HYDRO=1 BM=192 HREF=8 HAUG=64`）全 8 层 × 6 call = 48 plans **0 quota/start mismatch**，blk peak cuda==py。
- **延迟轨迹**（单 SM plan CTA，milestone 周期计数）：592k → 400k（BFS u32 key+payload 解码，杀 u64 CAS spin）→ 254k（BFS 散射全 16 warp 并行）→ 39.9k（BFS 重构：散射扫全部单元+`bedge` u32 key 自携 payload、apply 只动 hplaced/upos/useq ~8 ops）→ 33.8k（`__reduce_min_sync` 归约）→ 31.8k（epilog/compaction 并行化）→ **28.1k（phase-2 消除：串行循环胜者 lane 直接写 `hplaced`，winfo/edst 快照机制全删）**。
- **当前构成**（48-call mean，cyc）：plan 36.1 = fixed+mord 0.6 + pre 3.2 + serial 14.7 + epi 0.8 + reloc 2.9 + build 1.5 + bfs 12.2（bfs 因层而异 2.4-12k，repair 量驱动）+ publish ~4。
  - L3 专项：plan 28.1，bfs 4.6；greedy 串行 ~12.1k（~380/专家：L gather ~75 + CREDUX ~150 + emit/epilog ~80）。
- **关键结论**：
  - 本卡 `CREDUX.MIN`（`__reduce_min_sync`）显著快于多步 butterfly/shfl 树/smem atomicMin（均实测更差）；部分掩码 `(1<<kk)-1` 再省 ~1.2k。
  - `clane` 预展平（`ckey` 区 1024 ints）+ `epk`/`clane` 一拍预取是实测有效的唯一预取方案；spec L 预取/lreg 驻留/lane0 纯串行/闭式枚举/邻接合并发牌全部实测更差或不可行（稠密冲突图：32/32 专家两两有共享 rank）。
  - **勿用 host-pinned `stats` 做热路径计数**——一条 atomicAdd 每边 ~µs 级 stall 曾毒化全部测量；smem 累加器+末尾一次写才对。
  - serial 窗口缺口根因 = 按专家串行的 phase-2 随 nu(38-111) 线性放大 → 改为串行循环内联 emit 后窗口 = greedy + ~1.3k。
- **剩余可压**：bfs init/level 固定开销 ~3k、pre 的 5 个 plan_sync ~1.5k、publish ~3.7k；greedy ~12k 已近串行语义地板（进一步需破坏 mord 顺序或接受近似）。
- **对拍工具**：`hydro_role_test.py <meta> ep64 <layers逗号> <calls>`；`PERCALL=1` 打每-call serial/nm/nu/greedy；dbg 字段 stats[4030..4043]（hsc[14-19] 现为 A1/barrier 探针，仅 stats!=nullptr 时生效）。

### 8.30 EP32 hydro 零开销冲刺（2026-11-15/16，默认数据 seg1024_disj）

- **bench 数据 r32d**（`~/work/mpo/data/r32d`，远端 `ep64s/data/r32d`）：从测量窗口均匀抽 30 步 × 40 层 = 1200 call（`r32d_prep.py`）；routing_off / routing_eplb{448,512} / eco{P}{c16,c32,eco2}_metas；`ora*` = hydro 参考 plan 写成 plain routing（零开销 ECO 基准，`r32d_ora_var.py <dir> <P><pl> <tag> '<kwargs>' <bm>`，kwargs 含 order/split 时用 `hydro.HydroPlanner.plan`）。
- **block M**：gc-moe 启发式对 P448–576 选 bm64，但实测（p50 µs）EPLB448：bm96 100.5 < bm128 102.5–102.9 < bm64 107.6–108.6；oracle c32：bm96 98.3–98.6，bm128 101.8–102.2，bm64 101.9–102.9。**公平对比点是 bm96**（ECO 用 `GC_MOE_ECO_BLOCK_M=96`，plain 用 `--block-m 96`）。
- **token 均衡对 oracle 无影响**（bm64：hydro 原样 102.4 / 均分 102.5–102.9 / 精修 8–16 步 102.4–102.6）→ 不在 kernel 里做 token 精修；block 峰值才是主因。
- **plan kernel 优化**（全部与 `hydro.py` 逐位一致，开关见 kernel 顶部宏 / env `GC_MOE_ECO_UNIT_*`）：
  - `HSPLIT=1` 均分单元 token（块计划不变，hydro 发布改用 `q[]`）；`HORD=1` 副本少的专家先 greedy（greedy 直接达 LB 50%→76%）；
  - `HCF=3/HCFLV=3` 闭式 deal：nb≥3 的专家用 1 次 CREDUX + 3 次 ballot 一次算出全部 pick（与逐单元 argmin 顺序完全相同）。**关键：ballot 必须放在 warp 一致分支里无条件执行，store 用"选择地址 + 私有 sink"写法**——条件 ballot / 条件 store 会被编译成逐个 BSSY/BRA.DIV，反而慢 2 倍；
  - relocate / BFS 用每专家副本 rank 位掩码（`erm`，在 `ogids`）代替逐副本循环；publish 去掉 early-exit 让 load 流水。
  - c32@512 bm64：published 50.3k → 38.7k cycles；bm96：34.5k；c16 bm96：25.5k。剩余大头是 greedy 每专家固定开销（约 450–575 cycles/专家）。
  - 失败尝试：独立 ballot 循环版闭式（smem 存 mask）更慢；无 relocate/BFS 只省约 2k。
- **集群结果**（p50 µs，bm96，节点含 a0014）：EPLB448 100.5；oracle c32 98.3–98.4，hydro c32（h5）106.3（+7.9）；oracle c16 100.1–100.4，hydro c16 102.7–102.8（+2.5）。trace（bm64 h4）：plan 已基本不阻塞 role1（stall 2.5 µs/wait），**主要开销在 role1 phase-1（副本槽）拉取：约 30% 的 token 花了约 24 µs，单 token 成本约为 phase-0 的 3 倍**；正在用 `GC_MOE_ECO_IDX_PROBE=1`（副本槽不读 canonical index，输出错，仅计时）定位。
- **基础设施**：`push32.sh <dest> <files>`（3 路并行上传到 node0 再经 node0 http:25290 内网扇出；单流约 0.4 MB/s，切勿 8 路同节点）；`runq32.sh` 每行 `tag|env|args|pkg`；包 gc-moe-h2…h5/h5p。a0013 被外部 49 GB 作业占用（52–80% util），2026-11-16 换成 a0014（121 GB 孤儿显存、0% util，环境从 node0 同步）。
- **c32 开销分解（bm96，trace/探针，2026-11-16）**：
  - `IDX_PROBE`（副本槽不读远端 canonical index）：role1_end −3.2 µs，但端到端只 −0.4 µs → 不是主因。
  - `SPEC=1`（不等 plan，用上一 bank 的旧 quota，仅计时）：p50 106.5 → 104.2，plan 等待约占 2.3 µs；其余约 5.7 µs 是副本阶段结构性开销。
  - ring 字段 15（副本块 L1 等 token 的最长时间）只有 0.6 µs：GEMM 消费者没有在等副本 token。端到端约等于"最后一块 token 到齐（≈role1_end）+ 约 35 µs 的 GEMM 尾巴"，在 oracle 和 ECO 里都成立；所以关键是 role1_end（ECO 49.7 vs oracle 39.0）。
  - role1 phase-1：每个 warp 只有约 1–2 个副本 token，但单 token 串行链很长（plan 载入 → count 行 → peel 从深位置走起 → 远端 index → payload → drain 时远端 SF/weight → S2G/publish），约 10–16 µs；新加的 bench 字段 `rep_entry_sum` / `rep_tok_sum`（ring 4 / 12，所有 warp 的累加）分别约 2290 / 813 µs 每 call。
  - **交错拉取（`GC_MOE_ECO_INTERLEAVE=1`，plan 一可见就先拉本 warp 的副本 token 再回头拉固定 token）是负收益**：c16 102.8 → 110.9，c32 106.5 → 107.4。原因：GEMM 按 pool 块顺序消费，打乱固定块的拉取顺序会拖慢整条 GEMM 流水。默认关闭（check 能过）。
- **结论（截至 h6）**：bm96 下 hydro c16 约 102.8，EPLB448 约 100.6–102.5（噪声较大），oracle c16 约 99.8；hydro c32 约 106.5，oracle c32 约 98.4。下一步方向：缩短副本 token 串行链（count 行随 plan 预取、peel 起点预计算、SF/weight 与 payload 同时发出）以及 placement 扫描（c20/c24）。
- **突破（2026-11-16，q_r32d13）：dispatch warp 数是 c32 开销主因**。`GC_MOE_ROUTING_WARPS=16`（默认 bm≥96 时 12）让每 rank 的 16 SM × 16 = 256 个拉取 warp 覆盖约 230 个副本 token（每 warp ≤1 个），副本阶段串行链不再叠两次。p50 µs（bm96，单次）：

| DISPATCH_SMS / ROUTING_WARPS | EPLB448 | oracle c32 | hydro c32 | hydro c16 |
|---|---|---|---|---|
| 16 / 12（默认） | 100.85 | 98.03 | 106.61 | 102.88 |
| **16 / 16** | 99.23 | 98.11 | **99.46** | 99.69 |
| 24 / 12 | 100.93 | 99.76 | 110.66 | 103.11 |
| 20 / 16 | 100.03 | 98.33 | 99.93 | 99.63 |

  hydro c32 只比 oracle 多 1.35 µs，与同设置 EPLB448 持平。加 dispatch SM 反而变差（compute SM 变少）。其余负收益/无收益实验：`GC_MOE_PEEL_BS=1`（二分 peel，无变化）、c20/c24（约 104，不如 c16）。h7 包默认路径有 DSL 编译错误（变量名冲突），已在 h8 修复。
- **w16 确认（q_r32d14/15/16，p50 µs，bm96，包 gc-moe-h8）**：EPLB448 98.97–99.28；oracle c32 97.91–98.31，hydro c32 99.46–99.80（**比 oracle 多约 1.5，比 EPLB 慢约 0.6**）；oracle c16 98.55–98.8，hydro c16 99.61–100.31（多约 1.3）。ROUTING_WARPS 20/24 下 hydro 持平（约 99.5–99.9），EPLB w24 变差（100.5）→ 用 16。`GC_MOE_ECO_HELPER=1`（plan 结束后回收它的 SM）更差（100.3–100.5）。trace（c32 w16）：role3_end 87.1 vs oracle 86.5；role1 仍平均等 plan 3.4 µs/wait（plan 在 25.6 µs 发布，r1 phase-0 在约 22–27 µs 结束）→ 剩余开销主要是 plan 延迟和 plan 占用的 2 个 compute SM。
- 失败：在 plan kernel 里把 epk/clane 两遍合成一遍并用 atomicOr 建 rank 掩码，逐位一致但 greedy 变慢（34.5k → 36.4k，疑似代码布局影响），已回退。
- **当前推荐运行配置**：`GC_MOE_ECO_HYDRO=1 GC_MOE_ECO_UNIT_HSPLIT=1 GC_MOE_ECO_UNIT_HORD=1 GC_MOE_ECO_BLOCK_M=96 GC_MOE_ROUTING_WARPS=16 GC_MOE_ECO_SPLIT_COUNTS=0`，包 gc-moe-h8，c32@P512 metas（`r32d/eco512c32_metas.npy`）。
- **收尾（q_r32d17/18，w16，p50 µs，各 3 次）**：oracle c32 97.97–98.27；oracle c32 + `GC_MOE_DBG_COMPUTE_MINUS=2`（plain 模式也空出 2 个 compute SM，即与 ECO 同等资源）98.14–98.6；hydro c32 99.55–99.72；oracle c16（-2 SM）98.51–98.95，hydro c16 99.78–100.05；EPLB448 99.06–99.21。**2 SM 预留只值约 0.2 µs；hydro 相对同资源 oracle 还剩约 1.2–1.3 µs**（c32 主要是 role1 平均等 plan 3.4 µs/wait；c16 plan 不阻塞，剩余在副本拉取机制）。
- `GC_MOE_ECO_NO_RESERVE=1`（compute 占满整个分区，与 plan CTA 重叠的那个 cluster 等 plan 退出后才启动）：eager check 能过，但 CUDA graph 计时 run 在 NCCL 初始化后挂死 → 不可用（开关保留，默认关）。
- 剩余可做：c32 plan 再快约 3 µs 才能消掉 role1 等待（greedy 每专家固定开销是瓶颈，已试的闭式/融合/寄存器负载都到顶）；副本拉取单 token 串行链（count 行预取、SF/weight 与 payload 同时发出）。

### 8.31 EP32 bm192 零开销目标（2026-11-16 起，用户新目标）
- 要求：bm192；保持 hydro block 峰值；hydro（c32/c16/eco2，P448/512/576 均可）实际时间≈同一 plan 的 oracle（离线 plan 写成 plain routing）；baseline = EPLB448 / EPLB512。
- **bm192 的钉法**：`GC_MOE_ECO_BLOCK_M` 只接受 ≤128，bm192 用 bench 的 `--block-m 192`（ECO 和 plain 两种模式都钉住，plan kernel 的 UNIT_BM 也随之是 192）。
- 数据（r32d，本地与远端 `ep64s/data/r32d`）：`ora{512c32,512c16,448c16,448c32}_b192f.npy`（hydro 参考，split=even、order=few、bm192）；eco2 bm192 放置：`eco{448,512,576}e2b_metas{,_phy2log}.npy` + `ora{P}e2b_b192.npy`（tail/hot 参考）。
- 本地 plan 周期（bm192，`hydro_role_test.py`，HSPLIT=1 HORD=1）：c32@512 published 40.3k（serial 19.4k、bfs 6.0k）；c16@512 31.6k；c16@448 31.7k；全部逐位一致，blk peak 与参考相同。
- 队列：`q_b192a.txt`（EPLB448/512、oracle 与 hydro 的 c32@512 / c16@512 / c16@448，w16，2 次 + c32 trace）。
- **bm192 首轮结果（q_b192a/b，w16，p50 µs，各 2 次）**：EPLB448 102.2–102.5，EPLB512 105.3–105.7；oracle c32@512（f）103.8–104.0，hydro c32 106.9–107.4（+3.2）；oracle c16@512 105.7–105.8，hydro 109.0–109.2（+3.3）；oracle c16@448 105.6–105.7，hydro 108.5–108.7（+2.9）。
- **bm192 下 token 均衡有用**（与 bm64 不同）：保持 block 峰值的 token 精修 oracle（`_b192t`：split=even + tokref 32 含交换，tok imb 1.35 → 1.14）c32 103.4/102.9，只挪不换（`_b192tm`，1.20）103.1/103.4；c16@512 104.7/105.0；c16@448 104.4/104.7。trace：oracle c32(f) 与 EPLB448 的 role1_end（37.7）、role3_start（79.9 vs 79.6）几乎相同，差在 combine（role3 12.2 vs 10.7 µs，按 token 量计）。block 峰值少（13.06 vs 14.37）并没有让 GEMM 早结束（GEMM 结束≈最后 token 到齐 + 尾巴）。
- 因此 bm192 有两件事：(1) hydro 相对 oracle 的约 3 µs 开销（c32 trace：plan 26.3 µs 发布，role1 平均等 3.5 µs）；(2) plan 质量：kernel 里要做保持 block 峰值的 token 精修，oracle 才能追上 / 超过 EPLB448。
- **bm192 c16 trace（q_b192c）**：plan 21.4 µs 发布，role1 只等 1.6 µs/wait，但 r1_plan_seen 32.9（phase-0 结束晚），role1_end 41.9 vs oracle 37.9（+4.0），role3_end 95.8 vs 93.6。即 c16 的开销是"副本 token 在 phase 0 全部结束后才拉"。`SPEC=1` 在 bm192 下不能当探针（旧 quota 让 c16 反而变慢到 111）。BFS 迭代上限（HAUG 4/8）对周期无影响。
- **split pull（包 gc-moe-h10，`GC_MOE_ECO_SPLIT_PULL=<W_r>` + `GC_MOE_ROUTING_WARPS=16+W_r`）**：每 SM 后 W_r 个 dispatch warp 只拉副本 token（plan 一到就开始，与固定 token 并行），前 16 个 warp 只拉固定 token，pool 顺序不变。正在测（q_b192d：W_r=8，c16/c32 check + 计时 + trace）。
- **split pull 结果（q_b192d，bm192，p50 µs）**：c16 + `ROUTING_WARPS=24 SPLIT_PULL=8`：107.31 / 107.23（w16 无 split：108.92 / 108.80；oracle c16 105.73 / 105.91）→ 开销 +3.3 → **+1.4**；c32：105.98 / 106.21（无 split 约 107.1；oracle c32 约 103.9）→ +2.2。两者 check 均 OK。trace（c16）：role1_end 38.9 vs oracle 37.9，role3_end 95.1 vs 93.6。
- **split pull 扫描（q_b192e，bm192，p50 µs）**：`ROUTING_WARPS=24 SPLIT_PULL=8` 仍最好：c16 107.13，c32 105.99；W 20/4：c16 108.2，c32 106.1；W 28/12 与 32/16 启动失败（每 SM 896/1024 线程超限，不可用）。同轮 oracle c16 105.59、c32 104.09，EPLB448（w16）102.7。**当前开销：c16 +1.5，c32 +1.9**。
- 下一步候选：(1) c16 的 plan 在 21 µs 发布，副本 warp 约 9 µs 拉完，早于固定 warp → plan 有约 5 µs 余量，可以在 kernel 里加保持 block 峰值的 token 精修（bm192 oracle 能快 0.6–1.0 µs）而不增加开销；(2) c32 继续压 plan 延迟；(3) 用 trace 定位 c16 剩余的 role1_end +1.0 / role3_end +1.5。
- **split pull 分配（q_b192f，RW=24，p50 µs）**：SPLIT_PULL 8/10/12：c16 107.11–107.93 / 106.95（另一次 110.28 离群）/ 107.37–107.59；c32 105.93–106.31 / 105.92–105.97 / 105.98–106.02；oracle c16 105.98 / 106.04。分配比例基本无影响 → 保持 24/8。
- **截至此时的 bm192 状态（多次合并）**：oracle c16 约 105.8，hydro c16（split pull）约 107.2（**+1.35**）；oracle c32 约 104.1，hydro c32 约 106.0（**+1.9**）；EPLB448 约 102.6；EPLB512 约 105.5。推荐 bm192 配置：`GC_MOE_ECO_HYDRO=1 GC_MOE_ECO_UNIT_HSPLIT=1 GC_MOE_ECO_UNIT_HORD=1 GC_MOE_ROUTING_WARPS=24 GC_MOE_ECO_SPLIT_PULL=8 GC_MOE_ECO_SPLIT_COUNTS=0`，bench `--block-m 192`，包 gc-moe-h10。
- 剩余思路：副本 warp 在等 plan 的约 7 µs 里预载副本槽的 count 行 / canonical 指针以缩短 plan 后的串行链；kernel 内保持 block 峰值的 token 精修（c16 plan 有余量，bm192 oracle 可快 0.6–1.0 µs）；c32 plan 再提速。
- **其他放置 + split pull（q_b192g，bm192，p50 µs）**：c16@448 hydro 107.1 vs oracle 105.5–105.6（+1.55）；**eco2（bm192 放置）oracle 与 EPLB448 持平**：P448 102.9–103.1、P512 102.9、P576 102.6，但 hydro 开销大：P448 107.3–107.4（+4.4），P512 116.5–116.6（+13.7），P576 131.8（+29，可动专家多、plan 慢）。eco2_512 check OK。→ eco2_448 是最有希望的组合（oracle≈EPLB448），要把它的开销压下去。
- **eco2 放置的逐位一致问题（2026-11-16 发现）**：bm192 eco2 放置里有少量专家两份副本落在同一 rank（eco448e2b / 512e2b / 576e2b 全部层合计 5 / 19 / 23 例；c16/c32 为 0）。GPU meta 把同 rank 的第二份标为 passive（永远不给 quota），而 `hydro.py` 参考照样把它当副本用 → `hydro_role_test` 在 eco2 上 quota_mismatch 2/16（P448）、4/16（P512）；eco2 的 oracle（`ora{P}e2b_b192`）也因此和 GPU plan 不完全相同。修法：放置后处理去掉同 rank 重复（把重复副本的槽与另一 rank 上单副本专家的槽对换），重建 metas 和 oracle；或让参考忽略 passive 副本。eco2 本地 plan 周期（bm192）：P448 published 43.9k（nm 27–37，serial 20.5k，bfs 8.6k）；P512 69.1k（nm 65–99，serial 43.2k）。
- **eco2 去重放置**：`~/work/mpo/ep64/dedup_place.py <in phy2log> <out> <hist_stats>`（同 rank 的多余副本与"该专家不在的 rank 上最轻的单副本专家"对换，再 permute_local）。生成 `r32d/eco{448,512,576}e2d_metas{,_phy2log}.npy`（移动 5 / 25 / 28 个槽）+ `ora{P}e2d_b192f.npy`（split=even，order=few）。eco2@448 去重后 hydro_role_test 0/32 mismatch，blk peak 13.08 不变。队列 q_b192h：e2d448 check + hydro/oracle/EPLB448 各 2 次 + trace。
- **eco2@448 去重结果（q_b192h，bm192，p50 µs）**：check OK；oracle 102.57 / 102.60 ≈ EPLB448 102.2 / 102.2；hydro 107.34 / 107.27（**+4.7**）。trace：plan 30.4 µs 才发布（本地 43.9k cycles：serial 20.5k，bfs 8.6k），副本 warp 平均等 plan 9.1 µs，role1_end 47.6 vs oracle 36.4。→ eco2 的瓶颈完全是 plan 延迟；要让它可用，plan 需要从输入后约 22 µs 压到约 13 µs（约 26k cycles）。c16/c32 的 oracle（105.8 / 104.1）比 eco2 差约 2–3 µs，但开销小（+1.35 / +1.9）。
- **eco2 限副本专家数（`ECO2_KMAX=<K>` 加到 `eco2_place.py`：只保留 nb_p95 最大的 K 个专家做副本，腾出的副本按单副本增益补给它们）**：bm192 放置去重后生成 `r32d/eco{448,512}k{16,24}_metas{,_phy2log}.npy` + `ora{P}k{K}_b192f.npy`。blk peak 13.14（448k16）/ 13.22（512k16）/ 13.15（448k24）/ 13.16（512k24），比不限 K 的 eco2（13.07–13.08）略高；512k16 plan published 28.6k、逐位一致。待上机：hydro vs oracle（split pull）。
- **eco2 限 K 结果（q_b192i，bm192，p50 µs）**：448k24 hydro 108.9–109.0 vs oracle 105.6–106.0；448k16 hydro 108.0 vs oracle 106.6–106.8；EPLB448 102.2–102.4。**限 K 让 oracle 变差约 3–4 µs**（eco2 优势来自大量冷专家各 2 份副本）→ 放弃限 K。结论：bm192 下 oracle 最好的是完整 eco2（≈EPLB448），但它的 plan 慢（+4.7）；开销最小的是 c16/c32 + split pull（+1.35 / +1.9），但其 oracle 比 EPLB448 慢 2–3.5 µs。下一步主攻：完整 eco2@448 的 plan 延迟（serial 20.5k + bfs 8.6k）。
- eco2@448（去重）bm192 本地 plan：HORD=0 published 48.4k（serial 21.8k，reloc 4.4k，bfs 8.1k），HORD=1 49.2k（serial 21.7k，bfs 10.3k）——greedy 顺序对 eco2 无帮助；blk peak 13.136（10 层 × 6 call）。要让 eco2 开销接近 0，plan 需约减半：候选是 BFS（8–10k，eco2 下 greedy+relocate 常达不到下界）和每专家固定开销。
- eco2@448（去重）bm192 离线（1200 call，split=even、order=few）：greedy 峰值高于最终 60.8%，relocate 后仍高于最终 33.7%，BFS 改变 plan 34.2% → BFS 对 eco2 是常态必需（保持 block 峰值就省不掉），plan 延迟只能靠缩短每专家固定开销 / BFS 每轮固定开销，或换"副本少但 oracle 好"的放置。
- **c32@P448 + split pull（q_b192j，bm192，p50 µs）**：check OK；hydro 106.04 / 106.09 vs oracle 103.73 / 103.96（+2.2）；blk peak 13.18。与 c32@512（106.0 vs 104.1）相当，没有更好。
- eco2@448（去重）bm192 本地（层 1,9,17,25,33 × 6 call）：published 58.5k，其中 **bfs 平均 18.9k**、serial 20.9k、reloc 4.0k → BFS 是 eco2 plan 的最大可压项（每轮 BFS = 多层，每层 bedge 清零 + plan_sync + 全 16 warp 散射 + plan_sync + warp0 归约/入队 + plan_sync，再 apply + plan_sync）。下一步：把每层的 3 次全块同步合并 / 让单 warp 在边少时独自完成散射，并保持逐位一致。
- 下一步具体动作（待做）：`eco_unit_kernel.cu` BFS 段（约 1123–1285 行，`for (int it = 0; it < kHydAug; ++ it)` 到 `milestone(30)`）每轮 5 次 plan_sync（建源 1、每层 3、apply 1）。改法：每层把 bedge 清零并进上一层的 warp0 段（省一次同步）；边数少时由 warp0 单独散射（免全块同步）；改完用 `hydro_role_test.py`（eco448e2d、BM=192、HSPLIT=1、HORD=1）验 0 mismatch，再上集群测 eco2@448（目标 hydro≈oracle 102.6≈EPLB448）。
- BFS 每层少一次同步（下一层的 bedge 清零 + 未访问掩码挪到 warp0 上一段末尾做）：逐位一致（eco448e2d、c16 均 0 mismatch），但 eco2@448 bfs 只从 18.9k 降到 18.2k（published 58.5k → 58.6k，无实质变化）→ BFS 开销主要不在同步次数，需要看每轮里 warp0 的串行归约/入队与 apply。改动保留（源码备份 /tmp/eco_unit_kernel.cu.pre_bfsync）。
- **BFS 分解（eco2@448 去重，bm192，层 1,9,17,25,33 × 6 call，每 call 平均）**：iters 1.48，levels 3.28；scatter 段（含散射后同步）5.1k，red+enq 计数（从散射开始到 warp0 解码/入队后同步，含 scatter）12.4k → warp0 解码 + 入队约 7.3k；apply 2.3k。下一步优化 warp0 入队：现在对每个发现的 rank 逐个 shfl + 4 次 ballot 求排位（`for jm = act`），可改成一次排序网络 / 前缀排名。`hydro_role_test.py` 现在会打印 "bfs per call (mean)"（stats[4030:4036] 跨 call 平均）。
- **BFS 入队并行排名（包 gc-moe-h11，kernel md5 56b9795047）**：warp0 入队时每个 lane 用 32 步 shfl 扫过全部 64 个候选键数比自己小的个数（键唯一，~0u 为空），代替"每个发现的 rank 一次 shfl + 4 次 ballot"的依赖链。逐位一致（eco448e2d、c16 均 0 mismatch）。eco2@448 bm192：red+enq 12.4k → 9.1k，published 58.6k → **54.1k**；c16@512：35.0k → 32.9k。上机队列 q_b192k（e2d448 check + hydro e2d448 / c16 + oracle e2d448 + EPLB448，各 2 次）。
- **h11 上机（q_b192k，bm192，p50 µs）**：check OK；hydro e2d448 107.06 / 107.38（h10 107.3，无变化）；hydro c16@512 107.02 / 107.05（h10 约 107.2）；oracle e2d448 102.45 / 102.73；EPLB448 102.11 / 102.47。plan 省约 4.5k cycles（约 2 µs）对端到端没有可见收益 → eco2 的 plan 仍在约 28 µs 后才发布，要靠量级上的改变（plan 再砍一半，或换放置）。
- **为什么 eco2 的 oracle 更好（离线统计，bm192，每 3 个 call 取 1）**：blk 峰值 / 均值、active 峰值几乎相同（e2d448 13.07/12.67/13.05，c16@512 13.07/12.67/13.01，c32@512 13.06/12.67/13.06）；e2d448 的 token 不均衡反而最差（1.55，峰值 rank 1191 token；c32 1.35 / 1036；EPLB448 1.13 / 868）。oracle 实测排序 e2d（102.6）< c32（约 103.9，P448/512 相同）< c16（约 105.6，P448/512 相同）≈ 与副本专家数正相关、与 P 无关 → 不是 token、block、active 或 slot 数，疑似与"每个副本专家的副本数"有关（c16 每专家约 5 份，c32 约 3 份，eco2 多为 2–3 份）：副本越多，每份副本的 token 来自越多的 source → pull / combine 的 (slot, source) 对数越多。待验证：统计每 rank 的 (slot, source) 非零对数峰值。
- 假设被否定：每 rank 非零 (slot, source) 对数峰值 / 均值：EPLB448 274.5 / 252.3，e2d448 259.9 / 194.2，c32@448 229.0 / 194.4，c32@512 232.0 / 194.6，c16@448 229.6 / 194.2，c16@512 228.3 / 194.3 —— e2d 的扇入峰值反而最高，却最快 → oracle 差异不来自 pull/combine 扇入。仍未解释的差异：e2d 102.6 < c32 103.9 < c16 105.6（多次复测一致）。
- **同放置 A/B（q_b192l，bm192，p50 µs）**：EPLB448 放置（`place448.npy`，42 专家 ×2 + 11 专家 ×3；去重后 `r32d/eco448ep_metas*`）上跑 hydro plan 的 oracle（`ora448ep_b192f`，blk 13.22，tok imb 1.40）= 103.11 / 103.36，**比同放置下 EPLB 自己的轮询切分（102.33 / 102.36，blk 14.37，tok imb 1.13）还慢约 0.9**；同轮 e2d448 oracle 102.68 / 102.31，c32@448 oracle 103.55 / 103.42。→ bm192 下"压 block 峰值"本身不带来端到端收益，oracle 上限大致封顶在 EPLB448 附近；hydro 的目标只能是"开销≈0 且不输 EPLB 太多"。eco448ep 本地 plan published 57.8k（与 eco2 同级，太慢）。
- **c20/c24/c32@512 + split pull（q_b192m，h11，bm192，p50 µs）**：c20 hydro 106.24 / 106.58 vs oracle 105.17 / 105.08（**+1.3**）；c24 105.87 / 105.97 vs 104.77 / 104.64（**+1.2**）；c32 105.74 / 105.62 vs 103.95 / 103.84（+1.8）。开销约 1.2–1.8 µs 与副本专家数相关不大；oracle 随 K 增大而变好。
- **c24 trace（q_b192n，bm192）**：hydro（split pull）与 oracle 的单 call 时间线已基本重合：role3_start 81.5 vs 81.1，role3_end 93.4 vs 94.9 / 93.3（oracle 空出 2 SM 时）；plan_exit 24.8 µs。**剩余约 1.2 µs 出现在 call 周期上**（period 120.0 vs 118.9 / 118.4），即 call 之间。猜测：graph 里多一个 plan 节点、且 plan 最先发射，dispatch / compute 晚一个发射。试验开关 `GC_MOE_ECO_PLAN_LAST=1`（runtime 里发射顺序改为 1,2,4,3，包 gc-moe-h12），队列 q_b192o（check + 3 次交错 A/B + oracle）。
- `GC_MOE_ECO_PLAN_LAST=1`（q_b192o，c24@512，p50 µs，3 次）：106.92 / 106.34 / 106.61，默认顺序 106.28 / 106.12 / 106.16，oracle 104.75 / 104.98 / 104.70 → plan 后发射略差（+0.4），**发射顺序不是 period 多出的原因**；保持默认（plan 最先）。c24 当前开销 +1.4。
- **ECO 固定机制开销（q_b192q，bm192）**：ECO 不接受无副本（P=384 identity 报 "multiples of 8, with replicas"），改用近 identity 的 `eco416cold`（每 rank 多 1 槽放最冷专家的副本，plan 几乎无事可做；`r32d/eco416cold_metas*` + `ora416cold_b192f`，blk 18.2）。hydro（split pull）129.45 / 129.54 vs oracle（w24）128.23 / 128.29（w16 131.4，更差）→ **与 plan 算法无关的固定开销约 1.2 µs**，与 c24 的剩余开销（+1.2–1.4）一致。trace：ECO 的 rank barrier 24.7 vs plain 15.4 µs（这个偏斜放置下 dispatch 前段明显变长），period 152.2 vs 146.2。→ 剩余开销在 count 交换 / dispatch 前段 / plan 节点等固定机制，不在 hydro plan 本身。
- **split-count 两 CTA plan 在 EP32 bm192 有效（q_b192r，c24@512 + split pull，p50 µs）**：`GC_MOE_ECO_SPLIT_COUNTS=1` 105.34 / 105.17 vs 默认 0 的 106.08 / 106.07，oracle 104.71 / 104.80 → **开销 +1.3 → +0.5**；check OK，2 次计时无崩溃（EP64 上曾必崩，EP32 需 ≥3 次零失败才算稳定）。`GC_MOE_ECO_RECV_COUNTS=4`（不读 count 行，探针）106.57 / 106.28，无帮助。
- **bm192 目标达成（q_b192r + q_b192s，包 gc-moe-h12，p50 µs）**：hydro c24@P512 + split pull + split-count：105.34 / 105.17 / 105.37 / 105.21 / 105.30（5 次零失败）；同一 plan 的 oracle（无 plan kernel）104.71 / 104.80 / 105.06 / 104.74 / 105.02 → **开销约 +0.4 µs（0.4%，在噪声量级内）**；blk 峰值 13.07（= 下界，与 oracle 相同）。对比 EPLB512 105.67（另轮 105.3–105.7）持平，EPLB448 102.2–102.5 快约 3 µs（bm192 下压 block 峰值不带来端到端收益，见上方同放置 A/B）。c32 同配置 105.25–105.31（另一次 109.65 离群）vs oracle 103.86–103.96，+1.4。
- **bm192 推荐配置**：`GC_MOE_ECO_HYDRO=1 GC_MOE_ECO_UNIT_HSPLIT=1 GC_MOE_ECO_UNIT_HORD=1 GC_MOE_ECO_SPLIT_COUNTS=1 GC_MOE_ROUTING_WARPS=24 GC_MOE_ECO_SPLIT_PULL=8`，bench `--block-m 192`，metas `r32d/eco512c24_metas.npy`，包 gc-moe-h12（kernel 含闭式 deal、掩码 relocate/BFS、BFS 并行入队排名）。

### 8.32 bm96 + split pull + split-count（2026-11-17 起）
- 动机：bm96 下 oracle c32（约 98.1）比 EPLB448（约 99.1）快，但当时 hydro 开销约 1.5 µs 吃掉了优势；把 bm192 上奏效的 split pull + split-count 搬到 bm96，看 hydro 能否真正赢 EPLB。
- 队列 q_b96a：c32 check；hydro c32/c24/c16（`GC_MOE_ECO_SPLIT_COUNTS=1 ROUTING_WARPS=24 SPLIT_PULL=8 GC_MOE_ECO_BLOCK_M=96`）vs oracle c32/c24（`ora512c{32,24}_b96f`，w16，`--block-m 96`）vs EPLB448/512（w16，bm96），3 次交错。
- **q_b96a 结果（包 gc-moe-h12，p50 µs，3 次）**：hydro c32 98.78 / 98.91 / 99.12（中位 98.91），oracle c32 98.55 / 98.17 / 97.96（中位 98.17）→ 开销 +0.7；hydro c24 99.11 / 99.24 / 99.40 vs oracle 98.49 / 98.43 / 98.57 → +0.8；hydro c16 99.99 / 99.19 / 99.08；EPLB448 99.57 / 99.16 / 99.17（中位 99.17），EPLB512 101.44。→ **bm96 下 hydro c32 比 EPLB448 快约 0.3 µs（98.91 vs 99.17，在噪声量级），比 EPLB512 快约 2.5 µs**；开销比旧配置（w16、无 split，约 +1.5）降到约 +0.7。check OK，3 次零失败。
- **c32@bm96 trace（q_b96b）**：plan_input 7.0，plan 23.0 µs 发布（输入后约 16 µs），副本 warp 平均等 plan 2.5 µs/wait，role1_end 40.7 vs oracle 37.3（+3.4），role3_start 76.9 vs 75.1，role3_end 87.4 vs 86.3（+1.1），period 111.1 vs 110.9（相同）。→ bm96 下剩余开销就是 **plan 延迟**（副本拉取晚结束 → GEMM 尾巴晚），不是固定机制；要再快需把 c32 的 plan 从输入后 16 µs 压到约 12 µs（greedy 每专家固定开销是大头，约 18k cycles）。
- **bm 扫描（q_bmsweep，同一配置对比 bm64/128/192）**：plain（native384、EPLB448、EPLB512）一律 `ROUTING_WARPS=16 --block-m <bm>`；hydro c32/c24@512 用推荐配置（SPLIT_COUNTS=1、RW=24、SPLIT_PULL=8；bm≤128 用 `GC_MOE_ECO_BLOCK_M`，bm192 用 `--block-m 192`）；各 2 次交错；hydro bm64/128 先 check。新 oracle `ora512c32_b128f`。
- **bm 扫描结果（q_bmsweep，包 gc-moe-h12，p50 µs，2 次；bm96 参照取 q_b96a 同配置）**：

| 算法 | bm64 | bm96（参照） | bm128 | bm192 |
|---|---|---|---|---|
| native384 | 142.3 / 143.4 | — | **131.1 / 131.2** | 132.2 / 132.1 |
| EPLB448 | 103.95 / 103.9 | **99.2–99.6** | 101.0 / 101.0 | 102.2 / 102.1 |
| EPLB512 | 106.2 / 106.1 | 101.4 | 102.2 / 102.5 | 105.9 / 105.4 |
| hydro c32@512 | 102.2 / 101.7 | **98.8–99.1** | 102.0 / 102.1 | 105.3 / 105.3 |
| hydro c24@512 | 102.4 / 102.6 | 99.1–99.4 | 101.7 / 101.4 | 105.6 / 105.4 |

  三档里 bm128 对所有算法都最快或并列最快（hydro c32 的 bm64 与 bm128 持平），bm192 对 EPLB / hydro 最慢；但 bm96 比三档都快约 2 µs（EPLB448、hydro）。各 bm 下 hydro vs EPLB448：bm64 hydro 快 2.0，bm96 hydro 快约 0.3，bm128 EPLB 快 0.6（c24）/ 1.0（c32），bm192 EPLB 快 3.2。hydro bm64 / bm128 check OK。
- **bm 扫描配套的 block 峰值 / token 不均衡（r32d 1200 call 全量，每 call 最忙 rank 的 block 数取平均；token 不均衡 = 每 rank token 的 max/mean；hydro 用与 kernel 逐位一致的 oracle `ora512c{32,24}_b{bm}f`）**：下界 bm64 19.67 / bm96 16.07 / bm128 14.57 / bm192 13.07。native384 blk 37.12 / 27.58 / 22.70 / 18.43，tok 2.527；EPLB448 22.53 / 18.57 / 16.76 / 14.37，tok 1.130；EPLB512 23.39 / 19.46 / 16.66 / 16.00，tok 1.120；hydro c32 与 c24 各 bm 均等于下界，tok c32 1.158 / 1.211 / 1.256 / 1.352，c24 1.163 / 1.221 / 1.274 / 1.351。
- **oracle 补表（q_orafill，plain w16，p50 µs，2 次）**：c32 bm64 100.89 / 100.69，bm128 100.32 / 100.33；c24 bm64 101.06 / 100.94，bm128 100.81 / 101.06。连同已有（c32 bm96 97.96–98.55、bm192 103.84–104.24；c24 bm96 98.43–98.57、bm192 104.64–105.06）→ hydro 相对 oracle 的开销：c32 bm64 +1.1、bm96 +0.7、bm128 +1.7、bm192 +1.3；c24 bm64 +1.5、bm96 +0.8、bm128 +0.6、bm192 +0.4。注意 oracle 把副本间 token 按 source 主序切（kernel 是按 source 轮流的 peel 序），每份副本的来源更集中，oracle 可能略偏乐观。
- **GEMM 占比与 block 数（q_gemm，bm96，plain w16，p50 µs，2 次）**：`GC_MOE_DBG_COMPUTE_MINUS` 拿掉 compute SM：136 SM → oracle c32 98.49 / 98.11，EPLB448 99.24 / 99.66；102 SM → 104.11 / 104.01，107.28 / 107.07；68 SM → 123.11 / 123.23，128.52 / 127.90。trace（oracle c32）：136 SM barrier 14.6 / role1_end 37.3 / GEMM 结束（role3_start）75.1 / role3_end 86.3；68 SM 只有 GEMM 结束推迟到 100.8（+25.7），dispatch 与 combine 不变。→ GEMM 跨度约 16→75 µs（约占一个 call 的 55–60%），其中随 SM 吞吐伸缩的部分在 136 SM 下约 25 µs；**68 SM 时 EPLB / oracle 的增量比 28.8 / 24.9 = 1.157 ≈ block 峰值比 18.6 / 16.1 = 1.155**（活跃专家 14.0 vs 15.2、权重字节 263 vs 286 MB 方向相反）→ 吞吐受限部分与 block 数成正比；但满 SM 时 GEMM 只部分在关键路径上（受最后 token 到齐 + L1→L2 链限制），13.5% 的 block 优势只换来约 1.1 µs。
- **GEMM 逐 call 时间线（包 gc-moe-h13，只在 plain + `GC_MOE_ECO_TRACE=1` 时生效的新 trace 点）**：ring 字段 13（反转，最早）/ 3（最晚）= 第一 / 最后一个 L1 任务开始（输入到齐后），14（反转）/ 7 = 第一 / 最后一个 L2 任务开始；字段 1 = GEMM 全部完成（role3 开始）。bench `--trace-dump @D/r32d/<tag>` 存 `<tag>_ring.npy`（[32 rank, 40 call, 16 字段]，µs，相对本 call role1 开始）。本地副本 `~/work/mpo/ep64/results/dumps/tr45_{ora_c32,eplb448}_ring.npy`（bm96，w16）。典型 call（oracle c32，各 rank 中位数平均）：barrier 14.4，首个 L1 21.2，首个 L2 32.9，拉取完 36.6，最后 L1 57.0，最后 L2 65.3，GEMM 完 76.9，combine 完 88.1；EPLB448：最后 L1 60.5，最后 L2 69.0，GEMM 完 78.9。每次 graph 回放的 call 0 是边界（period 约 456 µs），不计。
- **torch profile（Perfetto 用）**：bench 已支持 env `GC_TP=<远端绝对路径前缀>`（+ `GC_TP_NRANK=1` 只导出 rank 0），在计时前对一次 graph 回放（40 call）做 `torch.profiler` 并导出 `<前缀>_rank<r>.json.gz`。oracle c32@512 bm96 w16 的 rank 0 文件：`~/work/mpo/ep64/results/dumps/tp_ora_c32_b96_rank0.json{,.gz}`。每 call 3 个 kernel：dispatch（grid 16 × 512 线程）约 33–38 µs，compute（grid 136 × 384）约 68–75 µs，finalize / combine（grid 64 × 384）约 10 µs，另有 4 个约 1.5 µs 的 `memcpy32_post`（bench 在 graph 里准备 routing 的拷贝）。profile 里的 call 0 被跨 rank 同步拖到约 58 ms，不计。kernel 内部阶段 torch profiler 看不到，另用 ring trace 生成 `phases_ora_c32_b96.json`（rank 0 与最慢 rank 2，call 1–7：pre-barrier / pull / L1、L2 任务开始窗口 / GEMM 尾巴 / combine，来自 tr45_ora_c32 那次 trace run）。
- **L1/L2 窗口 vs block 数（逐 rank × 逐 call 回归，tr45 ring dump + 同 call routing 1160–1199，bm96 w16，1248 样本）**：首个 L1 / 首个 L2 开始与 block 数无关（R² 0.00）。L1 窗口：oracle c32 每块 +4.6 µs（R² 0.72，block 14–17），EPLB448 每块 +2.1 µs（R² 0.61，13–19）；L2 窗口：+2.7（R² 0.37）/ +2.4（R² 0.73）；GEMM 完成：+3.7（0.36）/ +2.2（0.47）。token 数解释力更弱（L1 最后开始 R² 0.19 / 0.11），加入 token 不改善 R²；pull 结束主要随 token（R² 0.40 / 0.28）。→ 窗口 = 与 block 无关的数据受限段 + 随 block 数增长的算力受限段，不是严格正比。

### 8.33 上游同步（2026-11-16，分支 eco_ep64_up：e61e933 合并 + b4a950b 修复，包 gc-moe-u5）
- fork/main 与本地 main 已快进到上游 7902831。ECO 分支合并策略：plain 用上游无 barrier 协议；ECO 同协议但保留 dispatch rank barrier（恢复 nvlink_barrier、nvl_barrier_* 信号字段、3 个 PTX 封装），ECO finalize 用 plan quota 自算 pool blocks，ECO combine 等全部 peer。合并后修复：role1 动态 smem 给 ECO canonical 表留位（否则每次非法访存）、eco_fixed_slots 编译参数补齐 tensor_map_18..21、DSL 变量名冲突。
- 本地 EP4（GPU 2-5，`run_local4.sh`，数据 `~/work/mpo/data/r4`）：plain 与 h12 逐位一致；ECO（auction）各配置 --check 全 0。hydro 在 R=4 编译不过（scratch 静态断言，设计规模 R≥32）。
- **EP32（q_u5，p50 µs，2 次）**：plain dump（EPLB448 bm96，20 call × 32 rank）u5 与 h12 逐位一致；hydro check：c32 bm96、c24 bm192 均 OK。
  - bm96：EPLB448 w16 h12 99.14 / 99.17 → u5 98.59（另一次 102.79 离群）；**EPLB448 u5 默认 warp（上游新默认 24）97.54 / 97.67**；oracle c32 w16 h12 98.25 / 98.31 → u5 98.21 / 97.91；**hydro c32 h12 98.99 / 99.26 → u5 97.52 / 97.71**。
  - bm192：EPLB448 h12 102.30 / 102.36 → u5 102.11 / 101.93；**hydro c24 h12 105.37 / 105.40 → u5 107.54 / 107.66（回退 +2.2）**；oracle c24 u5 104.95 / 104.67 → u5 下 hydro c24 开销 +2.8（h11/h12 时 +0.4）。待查。
- **新基线（q_u5base，gc-moe-u5，bm96，plain 用上游默认 24 warp，p50 µs，3 次）**：EPLB448 97.93 / 97.76 / 97.80（中位 97.80）；EPLB512 100.00 / 100.45 / 100.01；oracle c32 96.27 / 95.86 / 95.93（95.93）；oracle c24 96.55 / 96.37 / 96.51；hydro c32（w24 sp8 sc1）97.64 / 97.70 / 97.76（97.70）；hydro c24 97.38 / 97.67 / 97.79（97.67）；EPLB448 全默认（启发式 bm64）106.08 / 106.13 / 106.24。→ oracle 随 24 warp 变快约 2 µs（98.1 → 95.9），比 EPLB448 快 1.9；hydro c32 与 EPLB448 持平（−0.1），相对 oracle 开销回到 c32 +1.8 / c24 +1.2。以后 EP32 对比以此为基线。
- **GEMM 是否打满 HBM（2026-11-16）**：L20A：152 SM，L2 129 MiB，HBM 位宽 7936 bit × 3996 MHz → 理论 7.93 TB/s，实测 2 GiB device copy 6.80 TB/s。GPM 计数器（`nvidia-smi dmon --gpm-metrics`）在容器里全是 "-"，ncu 会串行化并发 role kernel（跨 rank 等待会死锁）→ 只能按数据量 / 时间窗推算。bench 新增 `GC_SPIN=<n>`（计时前对同一份输入连续回放 n 次 graph，供外部采样）。推算（oracle c32 bm96，最忙 rank）：每专家权重 18.8 MB（FP4 + scale），活跃槽 15.2 → 约 286 MB/call，加激活约 30 MB；GEMM 跨度约 56 µs（首个 L1 21.2 → 完成 76.9）→ 平均约 5.6 TB/s，约为实测可达带宽的 80%+；算术强度约 2×96 FLOP / 0.53 B ≈ 360 FLOP/B，属访存受限区。
- **"最忙 rank" 按实测定义（q_tr62，u5 新基线，bm96，trace + ring dump，calls 1160–1199 中的 39 个）**：每 call 取 GEMM 最后完成（ring 字段 1）的 rank。EPLB448：block 峰值 18.49，最忙 rank 的 block 17.82、活跃槽 13.62、token 806，最忙 rank 恰为 block 峰值 rank 的比例 56%，GEMM 完成 max / median 83.1 / 76.6；EPLB512：19.49 / 19.05 / 15.74 / 806 / 64% / 85.4 / 77.7；oracle c32：16.08 / 16.08 / 14.46 / 812 / 100% / 80.5 / 75.0；oracle c24：16.08 / 16.08 / 14.15 / 826 / 100% / 81.4 / 75.2；hydro c32：16.08 / 16.03 / 14.56 / 808 / 95% / 79.9 / 74.7；hydro c24：16.08 / 16.03 / 14.31 / 820 / 95% / 79.1 / 74.1。hydro / oracle 多数 rank 都在 16 块（并列峰值），所以"最忙 = 峰值 rank"几乎恒成立；EPLB 的 block 峰值会高估最慢 rank 的 block 数约 0.4–0.7。
- **成本模型验证：GEMM 窗口 ≈ a + W1·n_e + W2·(n_b − n_e)**（tr62 四个 plain 配置，逐 rank × 逐 call 4992 样本，y = 首个 L1 开始 → GEMM 完成，均值 55.8 µs）：R² 只用 n_b 0.670 / 只用 n_e 0.276 / W1·n_e + W2·extra 0.688 / 再加 token 0.720。合并拟合 W1 = 2.62 ± 0.03 µs，W2 = 2.16 ± 0.03 µs，W2/W1 = 0.82（分配置 0.84–0.91），截距约 14 µs；token 每 100 个约 +1 µs。→ 同专家的额外 block 只便宜约 15–18%（L2 复用收益有限），成本 ≈ 2.2·n_b + 0.46·n_e。局限：观测数据，n_e 与 extra 负相关（−0.48）、范围窄（n_e 11–16，extra 0–6），oracle 配置内斜率被放大（约 4.4）；要坐实需构造 routing 独立改变 n_e 与 n_b 的受控实验。
- **受控实验（进行中，q_ctl）**：`~/work/mpo/ep64/gen_ctl.py <dir>` 在各节点本地生成 `ep64s/data/ctl/ctl_ne{n}_nb{b}.npy`（[400, 32, 128, 6]，P512，每目的 rank 768 token，所有 rank 同一模式，同专家的 block 在 pool 中连续）。模式 (n_e, n_b)：(2,8) (4,8) (8,8) (4,11) (8,15) (12,12) (12,19) (16,16) (16,23)。plain、bm96、上游默认 24 warp、gc-moe-u5，各 2 次（`--chunks 10 --passes 3`）。用 p50 拟合 a + W1·n_e + W2·(n_b − n_e)。
- **受控实验结果（q_ctl，p50 µs，2 次均值）**：(2,8) 79.34，(4,8) 79.78，(8,8) 80.28，(4,11) 82.49，(8,15) 88.45，(12,12) 86.70，(12,19) 97.18，(16,16) 96.19，(16,23) 106.96。拟合 a + W1·n_e + W2·(n_b − n_e)：全部 9 点 W1 1.86、W2 1.43、W2/W1 0.77、R² 0.983；负载区（n_b ≥ 11）W1 2.08、W2 1.52、W2/W1 0.73、R² 0.988。固定 n_e 加 7 块：n_e 12 +1.50 µs/块，n_e 16 +1.54，n_e 8 +1.17；n_b = 8 时 n_e 2 → 8 只 +0.9 µs（轻载时 GEMM 藏在 dispatch 后面）。(16,16) 96.2 vs (12,19) 97.2：少 4 个专家多 3 块反而略慢。→ 模型形式成立；同专家额外 block 便宜约 25%（L2 复用），负载区成本 ≈ 1.5·n_b + 0.55·n_e µs。
- **L20A 实测算力**：torch._scaled_mm FP8 dense 8192³ = 2.93 PFLOPS（152 SM），折合 136 compute SM 约 2.6 PF；HBM 实测 6.80 TB/s。每 bm96 block 6.79 GFLOP（满 96 行）→ 纯算力约 2.6 µs/块；每新专家 18.8 MB → 纯 HBM 约 2.76 µs。受控实验边际成本 W2 ≈ 1.5、W1 ≈ 2.1 µs 都低于这两个下限 → 负载区 GEMM 仍有部分与 dispatch 重叠 / 额外 block 行数少（1 token 的 block 可能按 valid_m 少算），尚未打满 HBM 或张量核。
- **受控实验 2（进行中，q_ctl2）**：n_e ∈ {12, 14, 16} × n_b ∈ {14..22}（768 token 约束下可行的 11 组：(12,14/16/18)、(14,14/16/18/20)、(16,16/18/20/22)），额外 block 由 97 token 的专家产生（一满一 1 token）。2 次计时 + 1 次 trace（ring dump，用 GEMM 窗口算带宽/算力占用）。
- **q_ctl2 计时（p50 µs，前 2 次，部分）**：(12,14) 88.86/88.88；(12,16) 93.36/93.37；(12,18) 95.87/95.65；(14,14) 90.22/89.85；(14,16) 94.65/94.30；(14,18) 99.79/97.61；(14,20) 101.16/101.17；(16,16) 95.75/95.49；(16,18) 97.77/97.84；(16,20) 102.87；(16,22) 105.56。trace（tc71_*，ring dump 在各节点 `ep64s/data/ctl/tc71_<p>_ring.npy`）跑完后用来算每 rank 的 GEMM 窗口与带宽 / 算力占用。
- **受控实验 2：完整 GEMM 窗口（tc71 trace，首个 L1 开始 → GEMM 完成，32 rank × 39 call 中位数）**：(12,14) 49.6，(12,16) 54.8，(12,18) 57.0，(14,14) 50.0，(14,16) 55.1，(14,18) 58.9，(14,20) 63.7，(16,16) 55.2，(16,18) 59.5，(16,20) 64.2，(16,22) 68.3 µs；首个 L1 约 19 µs、拉取结束约 30 µs（约 11 µs 与 dispatch 重叠）。拟合 window = 14.96 + 2.53·n_e + 2.15·(n_b − n_e)，W2/W1 0.85，R² 0.992，最大残差 1.2 µs。窗口内平均 HBM（每专家权重计一次）4.0–5.5 TB/s = 实测上限的 58–80%；按满 96 行算的算力 1.9–2.2 PF。额外 block 只有 1 个 token 却仍要 2.15 µs → 成本是每 block 固定的（每个 M-block 重新流一遍专家权重 + 任务 / 流水开销），不是 FLOP，也不是 HBM 总量；HBM 未打满。要让 HBM 成为瓶颈：同专家多个 M-block 在一个任务里共享权重 tile（权重读一次、对多块做 MMA）。
- **L2 复用的直接原因（2026-11-16）**：上游 `_routed_weight_linger`：均匀估计每专家 token ≥ 2·block_m 才用 EVICT_NORMAL，否则 routed 权重 TMA 加载带 **EVICT_FIRST**。EP32 T=128：128·6/16 = 48 < 192 → EVICT_FIRST → 同专家第二个 block 大概率 L2 未命中、重读 HBM（解释 W2/W1 ≈ 0.85）。新增实验开关 `GC_MOE_ROUTED_LINGER=1/0`（强制 EVICT_NORMAL / EVICT_FIRST），下一步用受控模式对比 W2。
- **`GC_MOE_ROUTED_LINGER` 对比（q_linger，trace 运行，p50 µs）**：强制 EVICT_NORMAL（=1）全面更慢：(12,14) 97.92 vs 默认 EVICT_FIRST 92.98；(12,18) 104.88 vs 100.62；(16,16) 106.27 vs 99.51；(16,22) 119.16 vs 111.58（+5–8 µs）。→ 让权重留在 L2 反而挤占 L2（激活 / 中间结果 / 其他专家）；EVICT_FIRST 是对的，单靠缓存提示拿不到同专家 block 间的权重复用。真正的复用要改任务结构（同专家多 M-block 共享一次权重 tile 读取）。ring dump：`ep64s/data/ctl/tl72_*_ring.npy`。
- **按块选择驻留提示（`GC_MOE_ROUTED_LINGER=2`：满 M-block 用 EVICT_NORMAL，专家的不满末块用 EVICT_FIRST；包 gc-moe-u7，q_l73，p50 µs，各 1 次）**：(12,18) 98.92 vs 默认 95.40；(16,22) 108.72 vs 105.80；(16,16) 95.97 vs 95.69；(14,20) 104.47 vs 101.24；EPLB448 101.43 vs 97.86 → 全部变慢 0.3–3.6 µs。只要有权重以 EVICT_NORMAL 留在 L2 就会挤占其他数据，缓存提示路线基本走不通；同专家权重复用需改任务结构（方案 1）。
- **精确驻留提示（`GC_MOE_ROUTED_LINGER=3`：scheduler 在任务字 2（m_block，compute 侧不读）的 bit 31 标记"该专家后面还有 M-block"，只有这时权重用 EVICT_NORMAL；包 gc-moe-u8；本地 EP4 plain 输出逐位一致；q_l74，p50 µs，各 1 次）**：(12,18) 99.05 vs 默认 95.49；(16,22) 108.88 vs 105.50；(16,16) 95.77 vs 95.63（无多块专家，等同默认）；(14,20) 104.59 vs 100.81；EPLB448 101.08 vs 98.01 → 与近似版（=2）几乎相同，仍慢 3–4 µs。结论：误判不是原因；只要多块专家的权重以 EVICT_NORMAL 留在 L2 就变慢，缓存提示路线放弃（开关保留，默认不变）。
- **耗时函数重新拟合（真实路由 tr62 四配置 4992 样本 + 受控 tc71 十一模式 13728 样本，bm96，逐 rank GEMM 窗口）**：合并拟合 window = 15.2 + 2.53·n_e + 2.12·(n_b − n_e)，R² 0.909，rmse 1.58 µs；加 token：8.5 + 2.52·n_e + 2.11·extra + 0.89·tok/100，R² 0.913。即 ≈ 2.12·n_b + 0.41·n_e（+ 0.9 µs / 100 token）。满块与不满块成本相同（真实路由 2.16 vs 2.11）。真实路由单独：R² 0.69–0.72，rmse 1.7–1.8；同一 call 内（固定效应）W1 2.88、W2 2.19、Wt 0.84。**对最慢 rank 的预测弱**：模型挑中实际最慢 rank 的比例 EPLB448 28% / EPLB512 36% / oracle c32 10% / c24 13%（call 内 rank 相关 0.62–0.76）；模型最大值比实际最慢 rank 的 GEMM 完成时刻低 1.7–4.4 µs（32 个 rank 取最大的噪声项，约 2σ）。hydro 压平 block 后最慢 rank 主要由模型外因素决定。样本内评估、trace 运行、39 个 call。

### 8.34 固定 benchmark：`benchmarks/benchmark_gc_moe_eco.py`（2026-11-16 起，提交 8a645c7）
- 计时方式同上游 `benchmark_gc_moe_se.py`：每个抽样 call 单独录 1-call CUDA graph，每窗口前 barrier + 冲刷 L2 + sleep，窗口内回放 `--inner-iters` 次取平均，`--iters` 个窗口取每 rank 中位数，报最慢 rank；`--calls` 个均匀抽样 call 再取 mean / median。
- 共享专家：默认 DSV4.1-Flash 预设（1 个，权重 K 粒度 32 → 权重 recipe (1,32,32)、激活 recipe (1,1,32)），按 test_gc_moe_se.py 构造；`--num-shared-experts 0` 只测 routed。`--block-m` 钉死所有模式的 bm（共享输入 SF 布局随之）。模式：`NAME:plain:384`、`NAME:eplb:P:routing_eplbP.npy`、`NAME:ora:P:oraFILE`（配 `--phy2log @D/r32d/eco{P}c32_metas_phy2log.npy`）、`NAME:eco:P:metas.npy`；`--check` 比较 ora / eco 与 plain 输出。
- EP32 用法：包 `gc-moe-u9`（含 gc_moe/ + benchmarks/），`SCRIPT=../gc-moe-u9/benchmarks/benchmark_gc_moe_eco.py ./ep32e_launch.sh 8 <tag> ...`，SG_ENV 的 PYTHONPATH 只放 gcenv + dg341（不能再放别的 gc_moe，脚本断言 gc_moe 来自自己的仓库）。没有 RESULT 行，结果看 node0 日志的 `==` 行或 `--output-json`。无 DeepGEMM 的主机自动用 gcbench/dg_quant.py（已补 per_block_cast_to_fp8、2-D SF 布局）。
- 注意：同一个 call 连续回放 inner_iters 次，权重在回放之间会部分留在 L2（每窗口只冲刷一次），比真实逐层 decode 偏乐观；需要时加 `--cold-l2`（每次回放前冲刷，并扣除冲刷本身的时间）。
- 冒烟（se1，EP32，bm96，shared 1，hydro 推荐 env，4 call × 10 窗口 × 20 回放）：plain384 99.32，eplb448 88.29，ora512c32 87.33，eco512c32 92.72 µs（mean）；check ora / eco 与 plain max|diff| 0。本地 EP4（auction）check 也为 0。注：本地 EP4 那次运行时 GPU 2-5 其实被别的任务占着（87–98% util），只当功能验证，计时作废。
- **用户要求（2026-11-16）：以后每次计时都加 `--cold-l2`**（每次回放前冲刷 L2，并扣除冲刷时间）。
- **se2（新 benchmark + `--cold-l2`，EP32 bm96，shared 1，calls 0/400/799/1199，10 窗口 × 20 回放，gc-moe-u9）**：mean µs plain384 106.67 / eplb448 94.50 / ora512c32 92.24 / hydro eco512c32 99.52（check 0）。实测最慢 rank（每 call 取 per-rank 中位数最大的 rank）：block 13.50 / 15.25 / 15.50 / 15.50，活跃专家 11.50 / 13.25 / 14.00 / 13.75，token 596 / 740 / 812 / 778；block 峰值 21.25 / 18.00 / 15.75 / 15.75，活跃峰值 12.00 / 14.00 / 14.75 / 14.75；block 不均衡度（max/mean）1.399 / 1.120 / 1.037 / 1.037，token 不均衡度 1.764 / 1.118 / 1.208 / 1.208；最慢 rank 恰是 block 峰值 rank 的比例 0% / 0% / 75% / 75%。只有 4 个 call，统计量噪声大。hydro 的负载按逐位一致的 oracle plan 统计。JSON：results/se2.json。
- **耗时函数 vs se2 实测**：每个 rank 的端到端中位数在同一 call 内只差 2–3.5 µs（std 0.5–0.9），与该 rank 负载无关（模型逐 rank 差 3–36 µs，相关 −0.36…+0.49）：combine 要等所有 rank，端到端时间被全局同步拉平，实测"最慢 rank"多为 rank 0 / 8（位置效应），**不能用端到端逐 rank 时间验证逐 rank 成本函数**。可验证的是 call 级：measured slowest = 35.6 + 1.01 × max_r f_r（f = 8.46 + 2.52·n_e + 2.11·(n_b − n_e) + 0.89·tok/100），plain/eplb/ora 12 个点相关 0.97，斜率 1.01；EPLB448 预测 95.72 vs 实测 94.50，oracle 91.50 vs 92.24；EPLB448 − oracle 预测 +4.19、实测 +2.26（模型高估 block 平衡的收益，与 8.33 一致）。hydro 实测比同 plan 预测多约 8 µs = 在线 plan 开销。逐 rank 验证需要 trace 里的逐 rank GEMM 完成时刻。
- **新 benchmark 的 trace（提交于 2026-11-16）**：`--trace-calls N` + env `GC_MOE_ECO_TRACE=1`：每个抽样 call 录一个"冲刷 L2 + 清零 trace ring + call"的 graph，每次连续回放 3 次（避免各 rank 启动错开），读最后一次的 ring 条目，按 rank 汇集到 JSON 的 `per_call[i].trace_us`（barrier / pull_end / l1_first / l1_last / l2_last / gemm_done / role2_exit / role3_done，µs，自该 rank role1 起点）。ECO 模式下 l1_last / l2_last / role2_exit 字段被别的事件复用，只看 barrier / pull_end / gemm_done / role3_done。eager 方式不可用（各 rank 启动相差约 500 µs）。驱动脚本 `~/work/mpo/ep64/se_run.sh <tag> "<env>" <args>`（结果 JSON 拷到 results/se/），40-call 队列 `se_q40.sh`。
- **se40（新 benchmark，cold-l2，shared 1，bm96，40 call；计时 se40_t_c32/c24，trace se40_tr_c32/c24（5 次 × 3 连续回放）；gc-moe-u10）**：e2e mean µs plain384 124.20 / eplb448 94.62 / ora512c32 92.99 / eco512c32 97.29 / ora512c24 93.67 / eco512c24 97.90，check 全 0。
  - 逐 rank GEMM 完成时刻（trace）与模型同 call 内相关 0.98 / 0.89 / 0.81 / 0.79 / 0.81 / 0.77；模型挑中 GEMM 最慢 rank：90% / 57% / 22% / 18% / 15% / 25%（oracle/hydro 多 rank 并列 16 块，最慢者 n_e、token 略高：14.35 / 853 vs 峰值 rank 14.07 / 811）。
  - 重新拟合（eplb + ora，cold L2 + shared）：gd = 23.95 + 3.08·n_e + 2.51·(n_b − n_e) + 0.91·tok/100，R² 0.763，rmse 1.52；含 plain384 R² 0.938；加 l1_first（首块数据到达）R² 0.844。max_r 预测：EPLB448 误差 +0.01，oracle −1.6 / −2.2，hydro −2.3（平坦负载下 32 rank 取最大放大残差）。
  - call 级：e2e = 16.71 + 0.92 × max GEMM done，相关 0.99。GEMM 之后的尾巴（e2e − max GD）：plain 7.68 / eplb 8.57 / ora 10.68–10.86 / hydro 14.27–15.01。尾巴随"接近最大值才完成的 rank 数"增大（相关 +0.54：1.1 / 3.5 / 5.5 / 6.7 个）→ GEMM 完成越平，combine 越难与 GEMM 重叠。EPLB − oracle：max GD 差 3.74，尾巴差 −2.11，e2e 差 1.63。hydro 比 oracle 尾巴多约 3.6–4.2 µs，嫌疑：合并后 ECO combine 改为等所有 peer（`_role3_combine` 的 ECO 分支）。
- **hydro 比同 plan oracle 慢的拆解（se40 trace，c32）**：barrier 11.0 → 14.1，拉取结束 36.3 → 48.4，最晚 GEMM 82.3 → 83.0，最后 GEMM → combine 结束 7.77 → 9.02，每 rank combine 段中位 10.24 → 11.14。注意：之前的"尾巴 = e2e − max GEMM done"混用了计时 run 与 trace run，且含 role1 起点之前的时间，不可靠，作废。
- **ECO combine 等全部 peer 的代价（A/B，gc-moe-u11，探针 `GC_MOE_ECO_COMBINE_PROBE=1` = 改用 plain 的按 token 等待，输出可能不对，只计时；cold-l2，40 call，2 轮交错）**：e2e eco512c32 98.34 / 97.30 → 96.14 / 96.25（−1.6），eco512c24 97.16 / 97.81 → 96.12 / 96.36（−1.25），对照 oracle 92.59 / 93.47 vs 93.10 / 93.35（不变）。trace：combine 段中位 11.47 → 10.07（c32）、11.20 → 9.80（c24），与 oracle 10.11–10.26 持平；最后 GEMM → combine 结束 9.19 → 7.77。→ 确认约 1.2–1.6 µs 来自等全部 peer；剩余约 3 µs 差距在 combine 之外（barrier、拉取、role1 起点前）。
- **token 不均衡不是 hydro vs oracle 差距的原因**：两者是同一个 plan，每 rank 的 token 数、block 数、活跃专家数完全相同；差别只在在线机制（plan 等待、副本拉取顺序、combine 等待方式）。
- 正确版"只等对应 rank"：若按 token 各专家所有副本所在 rank 的并集等待，平均仍要等 17.2（c32）/ 17.8（c24）个 rank（共 32），收益不确定；精确做法需要源 rank 知道副本 token 实际由哪个 rank 计算（plan quota + 规范序位置决定）。
- **精确 combine 等待（提交 022a20c，包 gc-moe-u12，默认开；`GC_MOE_ECO_COMBINE_EXACT=0` 回退）**：role1 在路由循环里本地记录每个路由的 canonical 槽 j，拉取结束后按 p = Σ_r min(c_r, j) + #{r < self: c_r > j} 和各副本 [quota_start, quota_start + quota) 算出计算 rank，combine 只等它。验证：`GC_MOE_ECO_ROUTE_VERIFY=1`（拉取方远端记录实际拉取者，bench --check 比对）c32 / c24 各 24576 / 24576 路由一致，输出逐位一致；ECO_MAX_ROUTES = 1920 × 8（capacity 对齐到 1920）。
  - 计时（cold-l2，40 call，2 轮交错，p50 mean µs）：eco512c32 exact 97.22 / 97.90 vs 全等 97.84 / 98.12（−0.4）；eco512c24 97.48 / 98.54 vs 98.41 / 97.63（0）；oracle 对照 92.18–93.25。→ 端到端几乎无收益。
  - trace：每 rank combine 段中位 11.50 → 10.47（c32）、11.01 → 10.17（c24），已降到 oracle 水平（10.0–10.1）；最后 GEMM → combine 结束 8.78 → 8.16 / 8.90 → 7.92；但最晚 GEMM 完成 83.4 → 84.3 / 82.8 → 83.8（+1.0，oracle 对照不变 82.4–82.5）抵消了收益。嫌疑：role1 末尾的计算 pass 与仍在拉副本的 warp 抢同 SM 的访存，推迟副本拉取。之前探针的 −1.2–1.6 µs 偏乐观（它按 routed_id // 16 等错 rank，可能等得更早）。下一步候选：把该计算移到 plan CTA（发布 quota 后，等 eco_pub_issued 再算，不占 dispatch SM）。
- **模式 2（plan CTA 计算 peer，提交 2b6afa4，gc-moe-u13）初版**：verify c32 / c24 全部 24576 路由一致、输出逐位一致；但计时反而更慢（c32 99.08 / 99.33 vs 模式 0 97.63 / 98.47；模式 1 97.83 / 98.32），trace 最晚"GEMM 完成"85.7 vs 82.7。原因：trace 的 gemm_done = role3 起点，而 role3 在 graph 里依赖 plan kernel 与 role1 完成；初版每 warp 串行处理一个路由（每路由多次依赖的 L2 往返），16 个 warp × 48 路由把 plan kernel 拖到约 85 µs，挡在关键路径上（模式 1 的 +1 µs 同理：延长了 role1 结束）。修正：每线程一个路由、各源计数独立加载（gc-moe-u14，队列 se_qexact3.sh）。
- **模式 2 修正版（每线程一个路由，提交见 git log，gc-moe-u14，现为默认）**：verify c32 / c24 24576 / 24576 一致，输出逐位一致。计时（cold-l2，40 call，2 轮交错，mean µs）：eco512c32 96.20 / 97.04 vs 模式 0 97.81 / 98.01（−1.3）；eco512c24 96.44 / 96.58 vs 98.38 / 98.05（−1.7）；oracle 对照 92.89–93.27。trace：combine 段中位 11.16 → 10.01（c32）、10.94 → 9.81（c24），已低于 oracle（10.3）；最晚 GEMM 完成 83.5 vs 82.7（+0.8，残余）；barrier 14.8 vs 13.7。hydro − oracle 从约 4.8 µs 缩到约 3.4 µs（c32），剩余主要在 barrier / 副本拉取（pull_end 49 vs 36）。

### 8.35 在线规划目标的迭代（goal：耗时函数要能预测最慢 rank）
- 数据：se40_tr_c32/c24 + sx3_tr_m2/m0 的逐 rank GEMM 完成时刻（16640 个 rank×call，`/tmp/feat.py` 生成，特征从同一 call 的 routing 算；hydro 用逐位一致的 oracle plan）。
- 同 call 内（去掉 call 均值）单特征相关：n_b 0.79、tok 0.54、tok_remote 0.54、n_e 0.49、pull_end 0.48、maxsrc 0.35、barrier 0.21；maxblk / pad / full ≈ 0。拟合 gd ≈ 0.54·n_e + 2.52·n_b + 0.007·tok（R² 0.678）；加 barrier / l1_first / pull_end 到 0.746（这些不是 planner 可控量）。
- 逐 rank 位置偏差存在（如 rank 14/15/20/21 偏早约 1 µs，12/13/16/17 偏晚 0.7–1），同一时段的 run 之间相关 0.5–0.8，跨时段 0–0.5；学成逐 rank 偏置只在同一时段有小幅帮助（挑中最慢 rank 10–30% → 18–40%），跨时段无效 → 不作为规划目标。
- **关键：残差 σ ≈ 1.35 µs 近似独立噪声。** hydro / oracle 有约 14 个 rank 落在最高值 1σ 内（EPLB448 只有 3 个），max 由噪声决定。按 E[max_r(f_r + ε_r)]（ε ~ N(0, σ²)）预测 call 级最慢 rank 的 GEMM 完成：系统误差 −1.61 → −0.01 µs，rmse 2.09 → 1.19（相关 0.72 → 0.74）。→ 规划目标应改为"期望最大值"（平滑 max），而不是确定性 max；在 block 已达下界时，可调的只剩并列 rank 的 n_e 与 token。
- **跨 plan 验证（q se_qplans，gc-moe-u14，cold-l2，40 call，2 轮；与 EPLB448 同 run 配对的差值 µs）**：eplb512 +1.53 / +1.08；oracle c16 −2.72 / −2.57，c20 −2.88 / −2.01，c24 −2.05 / −3.01，c32 −3.63 / −2.66。模型 E[max]×0.92 预测：eplb512 +2.3，c16/c20/c24/c32 −4.2…−4.4（四者几乎相同，实测也在噪声内无法区分）→ 排序正确，幅度约为实测的 1.6 倍（实测 ≈ 0.6 × E[max] 差）。
- **规划余量**：hydro plan 的 E[max] 比"同总量完全拉平"高 1.4–1.5（模型 µs）；EPLB 高 3.8。局部搜索（只在同专家副本之间挪 token，保持 block 峰值 16.02，最小化 LSE(f/σ)）把 c32/c24 的 E[max] 降 0.92 / 0.94（预计端到端约 −0.55 µs）。实测验证：`/tmp/ls_write.py` 写出 40 个抽样 call 的原 plan（`ora512{c}_b96f_s40.npy`）与搜索后 plan（`_ls40.npy`），队列 se_qls.sh（同 run 配对、顺序交替、3 轮）。
- **E[max] 目标的实测验证（se_qls，oracle plain 模式，40 个抽样 call，同 run 配对、顺序交替、3 轮，µs）**：c32 原 93.06 / 93.08 / 92.59 → 局部搜索 91.70 / 92.20 / 91.90（均值 −0.98）；c24 93.37 / 92.86 / 93.26 → 91.50 / 92.30 / 92.01（−1.23）。模型预测约 −0.55（按 0.6 换算）→ 方向与量级都对，实测更大。block 峰值不变（16.02）。注意：推送多个文件时 push32.sh 可能漏节点（4/6/7），推送后逐节点核对。
- **局部搜索接受的移动**（c32，20 call）：672 次只挪约 17 个 token（不改 block，平衡 token）；约 300 次把整个副本（约 80 token，1 块）从高代价 rank 挪到低代价 rank（源 −1 块、目标 +1 块，block 峰值不变）。即 hydro 只按 block 平衡、并列时按槽号选副本，没有平衡 n_e 和 token。
- **可在线实现的变体**（`~/work/mpo/ep64/hydro_f.py`：HydroPlanner 贪心选副本改用代价 f = 2.521·blocks + 0.541·active + 0.007·tok，block 修复不变）：`tie`（block 优先、f 断并列）E[max] −0.29；`w`（选使 f 最小的副本）−0.46 / −0.39；`w` + 最多 8 轮"从最高代价 rank 挪一个单元到同专家低代价副本"−0.62 / −0.60（局部搜索 −0.92 / −0.94）；block 峰值都保持 16.02。oracle 文件 `ora512{c}_b96f_{w40,wr40}.npy`，实测队列 se_qvar.sh。注意：并行 ssh 写同一节点的多个文件会冲突，改为逐个顺序写并核对 md5。
- **se_qvar 实测（oracle，40 call，2 轮顺序相反，取两轮均值，µs）**：c32 原 92.52 / w 92.05（−0.47）/ wr 92.63（+0.11）/ ls 91.92（−0.60）；c24 原 93.28 / w 92.62（−0.66）/ wr 92.93（−0.35）/ ls 92.25（−1.03）。w 四次全为负（−0.26…−0.83）；wr 不稳定（c32 r1 均值 92.93 而中位 91.27，有离群）；同一 run 内后跑的模式似乎略快，需交替顺序。结论：E[max] 目标对变体的方向判断正确，"按 f 选副本"在 oracle 上约 −0.5 µs。
- **在线实现：hydro 成本后处理 `GC_MOE_ECO_UNIT_HPOST=K`（默认 0）**：plan kernel 发布前最多 K 轮：在代价 f = 2521·blocks + 541·active + 7·tokens 最高的 rank（并列取最小号），把一个 even 单元或整个副本挪到同专家另一副本，选使 max(f_src, f_dst) 降最多者（并列：源槽、数量、目标槽最小）。Python 参考 `~/work/mpo/ep64/hydro.py` 的 `HydroPlanner.plan(..., post=K)`（`_post`），`hydro_role_test.py` 读同一 env。本地单卡对拍（`/tmp/r32d_counts6.npy`，eco512c32，6 层 × 6 call）：K = 4 / 8 均 0 mismatch，block 峰值 16.00 不变，36 个 plan 中 33 个被改动。代价：published 33.8k → 41.0k（K = 4）/ 47.2k（K = 8）cycles，每轮约 1.6k（warp 归约 + 增量 f 也未降，瓶颈是每轮的串行依赖），即 plan 发布晚约 3.5 / 6.5 µs。集群 A/B：队列 se_qhpost.sh（包 gc-moe-u15）。
- **HPOST 在线 A/B（se_qhpost，gc-moe-u15，cold-l2，40 call，2 轮，µs）**：hydro c32 K0 96.81 / 97.07，K4 96.71 / 97.30，K8 97.48 / 97.13；c24 K0 96.47 / 96.66，K4 96.95 / 97.00，K8 96.44 / 96.72 → 无收益：plan 晚发布 3.5–6.5 µs 推迟副本拉取，抵消了 oracle 上约 0.5 µs 的计划质量收益。需要不增加 plan 延迟的实现（把代价放进 hydro 贪心本身）。check 输出逐位一致。
- **se_qtie（oracle，40 call，2 轮顺序相反，µs）**：c32 原 92.50 / 93.06，tie 93.02 / 93.04，w 91.68 / 92.75；c24 原 93.43 / 92.53，tie 92.79 / 92.42，w 92.39 / 93.42。与 se_qvar 合计：w 相对原 plan 约 −0.45（c32 −0.47 / −0.57，c24 −0.66 / −0.08），tie 约 −0.06（不显著）。单次测量噪声约 ±0.5 µs，0.5 µs 以下的差异需要 ≥ 4 轮才能判断。
- **8.35 小结**：
  1. 逐 rank 成本 f = 0.54·n_e + 2.52·n_b + 0.007·tok（同 call 内 R² 0.68，残差 σ ≈ 1.35 µs 近独立）；最慢 rank 的 GEMM 完成按 E[max_r(f_r + ε)] 预测，call 级无偏（误差 −0.01，rmse 1.19）。确定性 max 低估 1.6 µs。
  2. 跨 plan：实测端到端差 ≈ 0.6 × E[max] 差，排序正确；同 block 峰值的 c16/c20/c24/c32 模型与实测都无法区分。
  3. 规划余量：hydro 比"同总量完全拉平"高约 1.4 模型 µs；离线局部搜索（只挪副本间 token）实测 −1.0 / −1.2 µs，"按 f 选副本"（w）约 −0.45 µs，均不增加 block 峰值。
  4. 在线：后处理（HPOST）每轮约 1.6k cycles，plan 晚发布 3.5–6.5 µs，端到端无收益；w 需要把加权代价放进 hydro 的闭式分配（按 block 层级成批发放的结构会失效，可能同样增加延迟），未实现。
- **重大发现（2026-11-17）：hydro 推荐配置下"分离拉取"从未生效。** `GC_MOE_ROUTING_WARPS=24` 时双缓冲拉取区 24 × 2 × 5120 B 超过 SMEM 上限 → `pull_stages` = 1 → role1 走串行拉取路径，而 `GC_MOE_ECO_SPLIT_PULL` 只在 `_pull_pipeline` 里实现。合并前 h12 同样如此（pull_stages 1）。之前归功于分离拉取的收益其实来自 24 个 warp 本身。trace（sbr_tr，加了 ring 字段 12 = 本 rank 拿到全部计数）：hydro 与 oracle 拿到计数的时刻相同（max 15.8 vs 16.4），即"barrier 晚 3.7 µs"是比较了不同事件的假象（ECO 的字段 0 在 nvlink barrier 之后、plain 的在 acquire 之前）；真正的差距是副本拉取：plan 发布 24.1（max 27.5），而副本拉取首次发出 29.9（max 34.1）、拉取结束 48.5 vs oracle 37.5——串行路径里 warp 要先拉完单副本 token 才去等 plan。字段 11 已被"call 周期"占用，不能复用。
- **让分离拉取真正生效的 A/B（se_qsp，gc-moe-u16，cold-l2，40 call，2 轮，µs；ROUTING_WARPS 同时作用于 oracle）**：w24（串行路径，现推荐）hydro c32 96.51 / 96.70，c24 96.36 / 96.25，oracle 92.98 / 93.27；w16 + split4（流水线，分离拉取生效）hydro 110.29 / 110.50，105.41 / 104.87，oracle 95.45 / 95.19；w20 + split4 110.93 / 105.63；w20 + split6 99.60 / 99.38。check 0。→ 分离拉取真正生效后大幅变慢（每 SM 只有 4–6 个 warp 拉副本，副本 token 被串行化）；w24 串行路径仍最好。副本拉取晚的原因待细化：串行路径里副本 token 约 30 µs 才开始（warp 先拉单副本 token，plan 24–27 µs 才发布），之后约 18 µs 才拉完，单 token 延迟远高于 plain，需要逐 token 打点（候选：canonical 行的远端 index 读、副本槽的 peel 计算、计数表等待）。

### 8.36 串行拉取修复（上游默认：16 dispatch SM × 24 warp，role1 单级拉取）
- 确认上游默认：`GC_MOE_DISPATCH_SMS` 默认 `8 if num_ranks <= 8 else 16`（EP32 = 16 SM），`routing_warps = 24`（本地专家 ≤ 64 且 bm ≥ 96）。上游本来就是单级拉取；双缓冲 `_pull_pipeline` 和分离拉取都是我们加的，24 warp 时放不下 → 现在设了 `GC_MOE_ECO_SPLIT_PULL` 但流水线不可用会直接报错（不再静默忽略）。
- **逐 token 阶段打点**（ECO trace 槽 44–50 副本 token / 52–58 单副本 token，benchmark JSON `pull_phase_ns`；sph_tr，gc-moe-u17，hydro c32 / c24，20 call）：每 token µs（副本 vs 单副本）：定位（peel）0.99 / 0.78，读 index 1.53 / 0.33（副本去 canonical 所在 rank 远端读），等 ring 空位 0.02，发起激活拉取 0.54 / 0.18，读 scale + 权重 2.45 / 2.97，收尾（等激活到齐 + S2G 写回 + 等写完 + 释放）2.76 / 2.01；合计 8.3 / 6.3 µs。每 rank 每 call 约 350 个副本 token + 417 个单副本 token，384 个 warp 每个约 2 个 token 串行处理 → 每 warp 约 15 µs。plan 发布 23.9，首个副本发出 29.8，拉取结束 max 50.3。
- 结论：瓶颈是单级路径的逐 token 串行链（无跨 token 重叠），副本 token 额外多一次远端 index 读（+1.2 µs）。修复方向（不增加 SMEM）：(1) 收尾延后——S2G 发出后不立即等写完，先做下一个 token 的定位 / 读 index / 读 scale 权重，在下一次往暂存区发 G2S 前再等读完成（wait_group.read），写完成后再释放 l1_full；(2) 副本 token 提前预取下一个 token 的远端 index。(1) 对 plain（EPLB / oracle）同样有效。
- **收尾延后 `GC_MOE_PULL_DEFER=1`（已实现，未测）**：单级路径里 S2G 写回发出后不再原地 wait_group(0) + 释放 l1_full，而是记为待处理；在下一次需要暂存区之前（等 ring 空位 / 发 G2S 之前）、等 plan 之前、以及循环结束时才等写完并释放 → 与下一个 token 的定位 + 读 index（约 1.1–2.5 µs）重叠。部署包 gc-moe-u18 时集群 SSO 失效（push 报 "direct terminal material was not found"），队列 se_qdefer.sh（check + DEFER 0/1 交错 A/B：eplb448、oracle c32、hydro c32/c24，上游默认 24 warp）已停，待重新登录后重推 u18 再跑。
- **2.5 µs 的测量（绝对时间戳：trace 槽 59 = plan kernel 开始，60 = 本 call 最后一个 role3 warp，61 = 上个 call 的 60；benchmark JSON `abs_us`；sab_tr，gc-moe-u19，20 call 中位数，trace 模式，含 graph 内 L2 冲刷）**：从上个 call 的 role3 结束算起，role1 开始 oracle 81.15 / hydro 82.43（+1.28），hydro 的 plan kernel 在 82.18 开始（只比 role1 早 0.25）；call 内（role1 开始 → role3 结束）oracle 90.62 / hydro 91.13（+0.5）。→ hydro 多出的时间主要是 graph 里 plan 节点排在 role1 前面发射，role1 晚启动约 1.3 µs。新增 `GC_MOE_ECO_LAUNCH_ORDER=<数字>`（如 142 = dispatch、plan、compute，finalize 恒在最后），A/B 队列 se_qorder.sh（412 / 142 / 124，包 gc-moe-u20）。
- **发射顺序 A/B（se_qorder，gc-moe-u20，cold-l2，40 call，取中位数；第 1 轮有离群干扰，均值失真，只看中位数）**：hydro c32 412（plan 先，默认）96.95 / 95.82，142 98.15 / 97.94，124 98.85 / 97.84；c24 98.10 / 95.79，96.61 / 96.16，96.97 / 96.68；oracle 对照 92.94–94.49。→ plan 先发射仍最好；让 role1 先发射反而慢 1–2 µs（plan 晚 → 副本拉取更晚）。即 plan 节点带来的约 1.3 µs 发射延迟不能靠调顺序消除，要去掉只能把 plan 并进已有的 kernel（例如作为 compute kernel 里的一个额外 cluster），少一个 graph 节点。
- **副本拉取表 `GC_MOE_ECO_REP_TABLE=1`（已实现，待测，默认关）**：plan CTA 发布 quota 后，等 role1 过 rank barrier（role1 新写 `eco_idx_ready`），为本 rank 每个副本拉取位置（role1 token 游标 − 单副本 token 数，副本槽升序、各取 [start, start + quota) 规范位置）用二分求轮次 j 与源 rank，再远端读 canonical 行，写本地表 `eco_rep_table`（src_rank << 16 | token_topk），完成后发 `eco_rep_ready`。role1 串行路径处理副本 token 时等 `eco_rep_ready` 后查表，跳过 peel 定位与远端 index 读。只改 ECO 分支，plain（EPLB / oracle）代码不变。
- **副本拉取表第 1 版实测（se_qrep，gc-moe-u21）**：verify c32/c24 24576/24576 一致、输出逐位一致；但开表后 hydro c32 113.5 / 112.9、c24 108.2 / 106.6（关表 96.6 / 96.5）。trace：查表后每副本 token 定位 0.57、读 index 0.05 µs（原 0.99 / 1.53），但首个副本发出 50.7（原 29.8）：建表按副本槽逐个串行（每槽一轮 32 次计数加载 + 二分 + 远端读，约 3 µs × 槽数），表约 50 µs 才就绪。已改为对全部副本拉取位置一次性并行（gc-moe-u22 待测）。
- **2026-11-17 暂停并回滚无效改动**：回滚 `GC_MOE_ECO_REP_TABLE`（副本拉取表：v1 慢约 17 µs，v2 仍慢 1–4 µs）、`GC_MOE_PULL_DEFER`（未测，影响 plain 路径）、`GC_MOE_ECO_UNIT_HPOST`（在线无收益）、`GC_MOE_ROUTED_LINGER`（全部变慢；含 scheduler 任务字 bit 31 标记）、`GC_MOE_ECO_LAUNCH_ORDER`（plan 先发射仍最好）。保留：精确 combine 等待（模式 2，−1.3 / −1.7 µs）、各类 trace 打点（ring 字段 12、绝对时间戳槽 59–61、串行拉取逐 token 阶段槽 44–58）、分离拉取不可用时报错、benchmark（共享专家、cold-l2、--trace-calls）。回滚后只做了本地 pyflakes + plan kernel 编译检查，集群上尚未重跑 check / 计时。

### 8.37 在线规划耗时函数重拟合（EP32，新 benchmark 40-call trace，16640 个 rank×call，同 call 内去均值）
- 每 rank GEMM 完成时刻 ≈ 2.52(±0.03)·blocks + 0.54(±0.02)·active + 0.0066(±0.0002)·tokens µs，R² 0.678，残差 σ 1.35 µs；换成 auction 的 token 单位：**wb ≈ 384，wa ≈ 82，σ ≈ 205**（只用 plain 模式拟合：wb 320，wa 72，σ 167）。旧 EP64 auction 表（bm128/192）是 wb 293 / wa 52。
- 最慢 rank 用 E[max_r(f_r + ε)]（ε ~ N(0, σ²)）预测，call 级无偏；确定性 max 低估约 1.6 µs。
- 修改计划见对话（2026-11-17）：目标换成 f + 平滑 max；先离线（1200 call）比较 hydro / hydro_f(w) / auction(新权重) 的 E[max] 与 plan 延迟，再 oracle 实测（≥ 4 轮交替），最后在线实现且 plan 发布不晚于现在（约 24 µs）。
- **离线对比（1200 call，`/tmp/offcmp.py`，新系数 wb 2.521 / wa 0.541 / wt 0.0066 µs，σ 1.35）**：E[max] / 确定性 max / block 峰值——c32：hydro 56.06 / 54.36 / 16.07，hydro-w 55.67（−0.40）/ 53.85 / 16.07，auction(wb 384, wa 82) 55.84（−0.22）/ 54.67 / 16.61；c24：hydro 56.00 / 54.34 / 16.07，hydro-w 55.58（−0.41）/ 53.81 / 16.07，auction 55.82（−0.18）/ 54.67 / 16.62。→ hydro-w 最好（block 峰值不变）；auction 换新权重后 block 峰值升 0.55，E[max] 只降约 0.2。下一步：hydro-w 的 oracle 已在 40 call 上实测约 −0.45 µs（se_qvar / se_qtie），按计划需 ≥ 4 轮确认后再考虑在线实现（零额外延迟地并进 hydro 闭式贪心）。

### 8.38 shared expert 时间线（plain trace 新增 ring 字段：10 首个 shared L1 任务开始（倒置）、2 最后一个 shared L1 开始、5 最后一个 shared L2 开始；gc-moe-u23，sse_tr1/0，EP32 bm96，20 call 中位数，µs 自 role1 开始）
- 带 1 个 shared expert：shared L1 第一个任务 2.0 开始、最后一个 3.0 开始，shared L2 最后一个 9.5 开始（约 12 µs 前做完）；routed L1 首个任务 21.5–21.9；拿到计数 15.4–16.2；最晚 GEMM 完成 eplb448 87.2 / oracle c32 83.7。不带 shared：routed L1 首个 20.9–22.1，最晚 GEMM 86.8 / 84.5 → shared expert 完全藏在 dispatch 里，不推迟 routed GEMM。
- 计时（cold-l2，40 call）：oracle c32 shared1 92.53 vs shared0 92.74（无差）；eplb448 shared0 95.56，shared1 那次被干扰（mean 247 / median 338），需重测。
- 含义：shared 算完后到 routed L1 开工之间 GEMM SM 仍空闲约 9 µs（约 12 → 21 µs），等 dispatch（计数 ~16 µs、首个 routed block 凑齐 ~21 µs）。trace 模式下 benchmark 打印的计时（~1300 µs）不可用，只看 trace 时间线。

### 8.39 标准测试约定（用户一句 prompt 触发）
- **代码**：仓库 `/data0/home/nanjun.cp/ep64_exp/gc-moe` 分支 `eco_ep64_up`；改动后 `push32.sh gc-moe-u<NN>` 推 8 节点（md5 逐节点校验），`PKG=gc-moe-u<NN> ./se_run.sh <tag> "<env>" <args>` 跑 EP32（8 节点 × 4 卡），结果 `~/work/mpo/ep64/results/se/<tag>.json`。
- **benchmark**：`benchmarks/benchmark_gc_moe_eco.py`。固定设定：`--off @D/r32d/routing_off.npy --block-m 96 --cold-l2`；shared expert 默认 1（gran_k 32），`--num-shared-experts 0` 为对照。
- **标准 mode 集**（同一次 run 内 call 交错）：`plain384:plain:384`、`eplb448:eplb:448:$DD/routing_eplb448.npy`、`ora512cXX:ora:512:$DD/ora512cXX_b96f.npy`（+`--phy2log $DD/eco{P}cXX_metas_phy2log.npy`）、`eco512cXX:eco:512:$DD/eco512cXX_metas.npy`（hydro 在线），XX ∈ {c32, c24}；`DD=@D/r32d`。
- **hydro env**：`GC_MOE_ECO_SPLIT_COUNTS=1 GC_MOE_ECO_HYDRO=1 GC_MOE_ECO_UNIT_HSPLIT=1 GC_MOE_ECO_UNIT_HORD=1`（其余走代码默认：ROUTING_WARPS=24、COMBINE_EXACT=2、串行拉取）。
- **触发短语**：
  - "跑标准 check"：`--calls 4 --check`（新代码先过），verify 加 `GC_MOE_ECO_ROUTE_VERIFY=1`。
  - "跑标准 A/B"：先 check，再 `--calls 40 --warmup 3 --iters 10 --inner-iters 20`，全 mode 集，≥2 轮（<1 µs 的差异 ≥4 轮交错）。
  - "跑标准 trace"：`GC_MOE_ECO_TRACE=1 --calls 20 --warmup 1 --iters 2 --inner-iters 2 --trace-calls 3`（只作归因，不报性能）。
  - "改完跑标准验证" = push 新包 + check + 标准 A/B。
- 噪声 ±0.5 µs，报中位数；trace 模式下打印的计时不可用。参考点（shared1/bm96/cold-l2）：eplb448 ~94.6、ora512c32 ~92.5–93.0、hydro c32 ~95.8–96.5 µs。

### 8.40 "2 块 unit + 再平衡" 规划：离线实现 + L20A 32 卡 A/B（2026-11-17）
- **离线参考实现**（`/tmp/r32d_ora_2b.py`，hydro-w plan → 合并副本（每槽最多 2 块）→ token 再平衡）烘焙成 oracle routing `r32d/ora512c32_b96f_2b.npy`：blk peak 16.07 不变，active mean 13.77→12.75，最慢 rank active 14.47→14.34，det max 53.85→**52.92**，E[max] 55.67→**55.13**（σ=1.35，2000 次噪声）。正确性：4.2M 路由 0 不匹配，每专家 token 总量守恒。
- **L20A 32 卡 A/B**（16 台 L20A 中挑 8 台空闲：.72/.86/.87/.88/.92/.76/.96/.84；容器内 `/opt/conda310` torch 2.13+cu130、`gcenv/site` cutlass、`ep64s/dg341`、包 `ep64s/pkg/gc-moe-u23`；NCCL 必须 `NCCL_SOCKET_FAMILY=AF_INET NCCL_SOCKET_IFNAME=eth0`，否则被容器环境强制 IPv6/bond0 卡住）。4 轮交替模式顺序，每轮 40 call、cold-l2、bm96、shared1：
  | 轮 | ora0（hydro plan） | ora2b（2 块+再平衡） | 差 |
  |---|---|---|---|
  | r1 | 93.55 | 90.70 | −2.85 |
  | r2 | 92.79 | 91.81 | −0.98 |
  | r3 | 93.82 | 91.04 | −2.78 |
  | r4 | 92.07 | 91.59 | −0.48 |
  | 均值 | **93.06** | **91.29** | **−1.77** |
- 4/4 轮同向；模型预测 det max −0.93、e2e 约 −0.5，L20A 上实测 −1.77（L20A 只有 148 SM、更算力受限，block/active 优势转化更充分）。L20A 绝对耗时约 92 µs，与 B300 EP32 的 94–99 不同硬件不可直接比。
- 工具：`l20a_launch.sh`（8 容器 torchrun）、`l20a_run.sh <tag> <args>`、`l20a_q2b.sh`（4 轮 A/B）；数据经容器内网 http（.86:25298）分发，500 MB/s。

### 8.41 在线再平衡 pass（`GC_MOE_ECO_UNIT_HREBAL=K`，默认 0）
- **kernel 实现**（`eco_unit_kernel.cu`，BFS 之后、publish 之前，warp0）：K 轮，每轮取 f = kWB·blocks + kWA·active + tokens 最大的 rank（并列取最小下标），把该 rank 某槽的 t 个 token 挪到同一专家的另一个副本（别的 rank），t ∈ [q%bm 或 bm, 16, 48]；按 key = 1000·max(f) + sum(f) 选最优（并列取最小候选下标）。候选裁剪：每槽最多看 8 个副本、只考虑活跃副本（离线代价 +0.07 det max），pair = slot<<3|copy 移位解码。
- **正确性**：`hydro_role_test.py`（参考 = HydroPlanner + hydro_rebal.py，同整数权重）40 call × 1 层 **0 quota / 0 start mismatch**，块峰值一致；调试确认 round0 选择与参考逐位相同（a=1, key 8492968, s=27, b=75, t=78）。
- **开销**（plan kernel，warm，kcycles）：基线 plan 24.2 / publish 32.5；K=8 → bfs +4.0~5.5k（约 2–2.7 µs），K=16 → +7k（约 3.5 µs）。默认关，A/B 后再定。
- **离线收益**：hydro-w 53.85 → K8 53.39 / K16 53.26（det max），块峰值 16.07 不变；E[max] 55.67 → 55.47 / 55.39。
- 同时把 `ECO_TIME_WEIGHTS[96]` 从 (335, 38) 更新为重拟合值 **(384, 82)**。
- **待办**：集群 A/B（L20A 或 EP32）确认 +2–2.7 µs 的 publish 延迟不会吃掉约 0.5 µs 的收益；若超预算，降低 K 或把 top-3/sumf 改成增量维护。

### 8.42 再平衡 pass 的寄存器化重写（开销从 ~5.5k 降到 ~1.5k cycles）
- **重写要点**：每个 rank 的 blocks/active/tokens 放进寄存器（lane = rank），每轮只用 4 个 warp CREDUX（max、次大、第三大、sum）+ 每候选 2 个 shfl 读目标 rank 的状态；不再用 shared memory 数组、atomic 初始化、5 步 shuffle 归约。候选循环用**均匀 trip count**（`kPairIt = ceil(kS*8/32)`）保证 shfl 在 warp 内收敛。
- **修掉一个真 bug**：key 里的 sum 项用了本 lane 的 f，应为源 rank（r1）的 f（`sumf - v1 - f(rb) + fa + fb`）。
- **正确性**：`hydro_role_test.py` 40 call × 1 层 **0 quota / 0 start mismatch**（K=8、K=16 均通过），参考 `hydro_rebal.py`（cap=8 按副本下标、active-only，与 kernel 语义一致）。
- **开销**（warm，kcycles，40 call）：基线 plan 24.2 / publish 32.5；K=8 与 K=16 均为 plan 25.7 / publish 33.7（**+1.5k / +1.2k cycles ≈ 0.6–0.75 µs**）——多数 call 在 4–6 轮后无改进即退出，K 的上限很少跑满。
- **离线收益**（cap8+active）：hydro-w 53.85 → K8 53.44 / K16 53.35（det max），块峰值 16.07 不变。
- **已验证无效/不划算的简化**：整槽移动（t=q[s]）无收益；去掉 sum 项损失 0.08；去掉第三大（others 用全局 max）损失 0.18；只试最轻副本损失 0.43。

### 8.43 再平衡的端到端 A/B（L20A 32 卡，烘焙 plan，4 轮交错）
- 把 kernel 等价 plan（hydro-w + `hydro_rebal(rounds=8, cap=8, active_only=True)`）烘焙成 `r32d/ora512c32_b96f_k8.npy`（det max 53.44、块峰值 16.07、active 均值 13.65）。
- 4 轮交错（每轮 40 call，cold-l2，bm96，shared1），每轮取 40 call 最慢 rank 中位数（µs）：
  | 轮 | ora0（hydro） | orak8（再平衡） | ora2b（merge+LS） |
  |---|---|---|---|
  | r1 | 93.61 | 91.90 | 90.82 |
  | r2 | 92.26 | 92.38 | 91.71 |
  | r3 | 92.83 | 92.78 | 91.72 |
  | r4 | 93.13 | 92.19 | 91.89 |
  | 均值 | **92.96** | **92.31（−0.65）** | **91.54（−1.42）** |
- 结论：K8 再平衡的**规划质量收益约 −0.65 µs**（4 轮中 3 轮为正，单轮噪声约 ±0.5）；merge+LS 的更强 plan 约 −1.42。
- **在线 eco 路径在 L20A 容器上挂住**（NCCL "Connected all rings" 之后无进展，>14 min），未完成在线 A/B；烘焙 plan 的对比隔离了规划质量，但"publish 晚 1.2k cycles（约 0.6 µs）是否被 plan 等待的 slack 吸收"仍需在线验证。容器内可能需要 nvcc / JIT 缓存前置，待查。

### 8.44 在线 eco A/B（L20A）：publish 延迟不被吸收，再平衡在线为负收益
- **修复的两个基础设施问题**：① 8 台容器中 .84 缺 `eco512c32_metas.npy`（分发只带了 routing 文件），一个 rank 抛 FileNotFoundError，其余 rank 卡在 barrier（表现为"挂住"）；② 早期清理只 `pkill -f torch.distributed`（torchrun 父进程），bench 子进程残留并占住 26311/25311 端口与显存，导致后续 run `EADDRINUSE`。现在 runner 同时 `pkill -9 -f benchmark_gc_moe_eco`。
- **在线 A/B**（8 容器 × 4 卡，40 call，cold-l2，bm96，shared1，同 run 内三模式交错）：
  | 轮 | K | ora0 | ora2b | eco（在线 hydro） | eco−ora0 |
  |---|---|---|---|---|---|
  | r1 | 0 | 93.05 | 92.09 | 96.22 | +3.17 |
  | r2 | 8 | 93.09 | 91.62 | 97.34 | +4.25 |
  | r4 | 0 | 93.00 | 91.60 | 96.03 | +3.02 |
  | r3/r5/r6 | — | 失败（JSON 空 / 引号导致 $W 未展开），作废 |
- **结论**：
  1. 在线 hydro 比烘焙的同一 plan（ora0）慢约 **3.0–3.2 µs**（ECO 机制开销：plan 等待 + 副本拉取顺序），与历史一致；
  2. **K=8 的再平衡在线更慢约 1.1 µs**（97.34 vs 96.2）——publish 晚 1.2k cycles（约 0.6 µs）**没有被 plan 等待的 slack 吸收**，反而放大了；
  3. 与烘焙 plan 的 −0.65 µs 质量收益相抵后，在线版本为负。与 HPOST 的教训一致：任何 publish 延迟都会吃掉规划质量收益。
- **可行方向**：把再平衡并进现有 relocate/BFS（不新增 pass）、或把 K 降到 2–4 并验证 publish 不变；否则只能作为离线/oracle plan 使用。

### 8.45 Push dispatch 设计定稿（deepEP-LL 思路改造，2026-11-17 深夜）
- **动机**：单 call 50 次回放 trace（`results/dumps/single_call_eager/`）显示 `l1_first` 中位 ~22 µs，而 shared expert 在 ~10 µs 内结束 → routed GEMM 有 ~10 µs 气泡；各 rank 开工偏差中位 ~3.8 µs / 最大 ~7.5 µs，与 block 优化收益（~4.2 µs）同量级，必须把首 block 数据到达提前到 shared 结束之前。
- **现状机制**（`gc_moe/impls/sm100_fp8_fp4_gc_moe.py`）：pull 模式——源端 routing 循环写 4B source-index（`(dst_local*num_ranks+src)*capacity+pos` 处写 `token_topk`），目的端 dispatch warps 按 slot 序逐 token resolve（count 行+index 行 acquire）→ TMA `bulk_g2s` 拉 activation → drain 写 `l1_acts[pool%ring]` + `l1_acts_sf` + `l1_topk_weights` + `token_src_metadata` + `red_release l1_full_count[ring_b]`。首 block 下界 = max_src(routing_end) + pull 流时间。
- **实测**（本机 8×L20D，NV18 全互联）：remote `st.global` 写带宽 **~706 GB/s**（≈本地 HBM）；remote `atom.add` 串行延迟 **~1.8 µs** 但完全流水化（2560 并行 ≈ 0.01 µs/op）。每 rank 入向数据 ~4MB → 纯传输 ~6 µs；当前 pull 的 15–25 µs 窗口主要是远程读往返 + index/resolve 串行等待。
- **设计（v1，非 ECO/EPLB 先行）**：
  1. **src 侧 assignment**：routing 循环改为只算不写远端 index——每 CTA 把 (dst_rank,dst_local,pos,token_topk) 记进 smem 发表（pos = 现有 `atom_shared_add_u32` 游标，与 pull 的 canonical src-major 序一致）。
  2. **count 矩阵广播**：src 的 (src→slot) 计数行不再只发给 slot owner，而广播到**所有 rank**（复用 `source_idx` 区域做 overlay：32×512×u64=64KB ≪ 8MB 原索引区）→ 每个 rank 本地有完整矩阵，~2–3 µs 可得。
  3. **src 计算 pool 位置**（全本地）：`pool_pos = pool_base(s) + prefix_{j<src}(s) + pos`，`pool_base(s)` = dst r 上前序 slot 的 padded block 前缀、`prefix` = src-major 前缀——与 pull 的 peel/canonical 序**逐位一致**（输出 bit-exact）。
  4. **X 序发射**：X = 对所有 dst 轮转、每个 dst 按其 slot 的 pool 序（role2 消费序）；每个 rank 用同一矩阵算同一 X；src 把其 (dst,slot) 分片按 X 位置排，逐片：远程写 act(5120B)+sf(160B)+w(4B)+meta(12B) 到确切 pool 地址，片尾 `red.release` 加 `l1_full_count[ring_b]`（**协议与 pull drain 完全相同 → role2/role3 零改动**）。
  5. **armed 门**：src 的第一个远端写之前，需确认 dst 已清完本 gen 的 `l1_full_count`——dst 在 call 开始置 `push_armed_count`（remote add），src 本地 poll ==num_ranks 后发第一个数据写（约 +1–2 µs，可和 assignment 重叠）。
  6. **dst 侧**：dispatch warps 跳过整条 pull pipeline（无 index、无 resolve、无 remote read）；l1_full_count 仍由 src 的 release-add 驱动；GEMM 不变。
- **为什么不需要原子声明/3D 340MB staging**：pos 本来就是源端本地游标；跨 src 的 slot 内偏移靠"计数矩阵广播 + 本地前缀"解决，代价仅 ~4KB×32 的计数广播和 ~2–3 µs 等待，换来零额外数据 buffer。
- **已知边界**：① gen>0（pool>ring 1920 tokens）时 src 需远端 poll `l1_empty_count` 才能复用 ring——EP32/P512 pool~17 block < ring 20 不触发，留兼容逻辑；② ECO 副本 token 的 copy 选择要等 `plan_published`（~15 µs）——v1 仅做非 ECO；ECO 扩展：单副本走 push（t≈0），副本 token 或在 plan 后用 quota 表本地算 copy、或保留 index+pull 混合路径；③ 多节点（MNNVL）release 需 `red.release.sys`（u32 版要补 PTX），单机 EP8 原型用 gpu scope。
- **备选方案（已比较放弃）**：per-src FIFO staging + dst compact（+130MB buffer、+2–3 µs 本地搬运）；atomic-claim + 2D 定容 slot 区（+87–174MB、省 count 等待但角色大改）；deepEP-LL 三维 (slot,src,pos) 340MB。
- **预测**：routing/assignment ~2–3 µs + 首批 slice 发射 ≈ 首 block ~5–8 µs（vs 现在 ~22 µs）；shared 结束（~10 µs）后 routed GEMM 无缝接上。
- **实施顺序**：① 本设计落地 AGENTS ✓ → ② layout 加 push 信号/matrix overlay → ③ kernel 非 ECO 路径：routing 记表 + 广播 + X 序发射（GC_MOE_PUSH_DISPATCH 门控）→ ④ dst 跳 pull + armed 门 → ⑤ 本机 EP8 bit-exact + 时序 → ⑥ 在线 EP32。
- **Phase A 实现落地（2026-11-17，EP8 本机，与上条同文件）**：`GC_MOE_PUSH_DISPATCH=1` + 非 ECO + `_push_fits`（matrix overlay ≤ source_idx 区、hidden%64==0、capacity*topk≤16384、seq 数组 ≤ pull smem 区）门控。
  - **协议**：pass-1 各 CTA smem 计数 → `atom_add_u64` 发布到 `expert_send_count`（同时把 CTA 基址写回 `expert_token_count`，ord_g = 基址+局部序 = rank 级全局 ord）；pass-2 不碰远端，只做 `ord_g`/`ord_l`（CTA 内 seq_cnt 原子）双声明；每 CTA 把 384 计数行广播写进所有 dst 的 matrix bank（source_idx 区 overlay，按 gen 双 bank，tag=parity+1）→ 各 CTA 等全部 8 个 tag 后本地建 emit_base/prefix/total 三张表（src-major 前缀 + 块对齐 pool 前缀，**与 pull 的 canonical 序逐位一致**）。
  - **X 序发射**：CTA 内 counting sort（seq_cnt→chunk sums→X-permuted pfx→scatter 到 seq_list），X 位置 = `(e%L)*R + e//L`（相邻实例打向不同 dst，每个 dst 的首个 slot 都在前 R 个 X 位置）；warp 按 `pos ≡ warp mod 16` 步进 seq_list，全 lane 协作拷贝 act(10×16B batched-load→store)+sf+weight+meta，`fence_acq_rel_sys`+`sync_warp`+lane0 `red_release_sys_add_u32` 到 dst `l1_full_count[ring_b]`（inc：末 token 补 `block_m - pos%block_m`，与 pull drain 同协议）；gen>0 远端 poll `l1_empty_count`。
  - **教训/坑**：`copy_global_16`（ld+st 融合 asm 带 ~memory clobber）**逐拷贝吃满 load 延迟、完全不流水**——必须拆成"整组 `ld_global_u32x4` 进寄存器 → 整组 `st_global_u32x4`"（v1 慢 ~71µs 的主因）；DSL `cutlass.range` 循环变量实为 Int32（混用 Uint32 边界触发 TYPE_UNSTABLE_JOIN）；smem 里跨 warp 的 packed 表不需要 valid 位——重读 `routes` 判有效性即可。
  - **trace 打点**（`GC_MOE_ECO_TRACE=1`，ring fields）：f2=矩阵广播完，f5=矩阵全就绪，f12=counts 齐，f4/f6=发射完，f11/f15=routing 完（max 聚合，取的是最慢 call，注意解读）。**v1 基线（token-major、串行拷贝）**：bit-exact 对拍 8 rank × 2 call 全 0，但 kernel_us_est 256 vs pull 190——emission 段独占 ~71µs（f5 20.5 → f4 91.6），修复方向即本条两优化。
  - **v2 实测（2026-11-18，EP8 T=128 plain routing_off，8×B300）**：bit-exact 8 rank × 2 call 全 0 ✓；kernel_us_est **206.1 vs pull 210.6**；trace ring（µs）：role1_end **53.9**（v1 91.8，pull 55.4）、f3 GEMM start 143.7（pull 144.3）、role3_start 163.6（pull 164.1）、role3_end **187.2**（pull 188.3）、period **208.4**（pull 218.5）。→ **emission 瓶颈完全消除，push ≈ pull 平手或略优**；批量拷贝（2 波 ×5 vec，20 reg）+ X 序 counting sort 生效。
  - **结论与解读**：① f2/f5/f4 探针在 v2 trace 中失真（f2/f5=0、f4≈10.7）是 persistent kernel 的**跨 call 归属噪声**——block0 先于慢 CTA 更新 slot24/33+清 entry，迟到 CTA 的早期探针落进下一 call 的 entry（elapsed≈0），不是协议错误（bit-exact 为证）；要精确打点需 per-CTA call 序号。② push 未赢的真正原因：**首块提前到达没转化成 GEMM 提前**——f3/f1 与 pull 逐位一致，role2/role3 启动被 shared expert 尾巴（~163µs）门控而非首块到达；此规模（T=128 EP8）下矩阵 wait ~15-20µs 也不在关键路径上。③ push 的理论收益（消掉 pull drain 串行窗口）在本工作负载被 shared-tail 主导的结构里兑现不了，**更大 T 或 shared 更短的配置才可能显出差距**。
  - **v3 结构优化（2026-11-18，已实测）**：**push 的 pass-2 整段消除**——pass-1 的 `atom_shared_add_u32` 返回值本来就是 CTA-local ord（存 `seq_ord[tt]`），publish 循环顺手把 `seq_cnt[e]` 填成 pre-swap 计数，`ord_g = etc[e](基址) + ord_l` 在发射时现推（`emit_prefix[e]` 构建时折入 `etc[e]`，发射循环零改动）；pass-2 路由遍历对 push 变空转（`pass2_end = route_base`）。token 在 (slot,src) 段内排布变为 pass-1 声明序（与 pull 不同的置换），meta 自带 src_token/topk_slot → per-token 输出不受影响。**实测（干净窗口，kernel_us_est µs）**：s2(imb47) 185.3 vs pull 186.3；s0(imb19) **186.7 vs 194.8**；T256 **205.2 vs 208.6**（trace 下 role1_end 96.9，2× 数据线性扩展）；T128 off 181.7 vs 183.2——**全部干净样本 push ≥ pull，倾斜/大 T 优势更大**（早前 T256 235、s0 +4.5、T256t-pull 276 均为外部作业污染样本，标注 procs_now>0 即弃）。对拍：off/s0/s2/T256 全部 max|diff|=0。另加 `_push_fits` 不满足时的显式 stderr 警告（防 EP32 smem 收紧下静默回退 pull）。
  - **待补**：ECO 扩展（副本 slot 需 plan 后 copy 选择）；gen>0 ring 复用路径未实测（EP8 pool_blocks<ring_blocks 不触发）；更大 T / EP32 的 A/B（push 收益可能随 dispatch 占比上升显现）；EP32 下 `_push_fits` 的 smem 约束（pull_stages=1 时 seq 表 ~131KB 可能超 pull 区）。

## push dispatch 进度（2026-11-18，deepep-dispatch 分支，未提交）
- 现状：**pair 布局 + deepEP 新协议**（pass-1 1µs / publish 本地原子 / 计数排序 1.5µs / 发射无等待 / 每 (expert,src) 一次 count 推送）。EP8 中位 ~1150µs，EP4 实测 1254µs，bit-exact 已验证（早期 push114 对照 0.0）。
- 已验证的替代：**稠密直写**（publish 改远端原子 → 跨 rank 序数 → 写 `local*capacity+ord`）EP4 实测 **425µs**（3×），但 bit-exact 未通过（地址映射：发射端窗口化 vs 消费端紧凑，已加"发布窗口化块号"修复待验证）。
- 已排除：TMA 远端 bulk store（~100µs/op）、每 token 远端原子（768/rank 太贵 +70-90µs）、专职 count warp（32 lane 并发 release.sys 互相拖慢）、gather 多段 TMA（swizzle 需 8 行对齐，EP32 每源仅 2 行）。
- 下一步（P2-P4）：**pair 写入 + dst 侧窗口内 repack**（role2 warp3@CTA≠0 空闲 warp 做拷贝，shared-expert GEMM 期间并行），目标 l1_first 10-16µs + 总 ~425µs。
- 关键实测：远端原子 ~1.8-2.5µs；fabric 每 rank 写带宽 ~27GB/s；GEMM 每块固定开销 ~3µs；块数 EP4 370→96（稠密）、EP8 274→48。

### push-repack 方案（pair 写 + dst 侧 repack → dense 消费）进行中（2026-11-19）
- 设计：emission 不变（pair 窗 + pair count）；role2 非 leader CTA（cta_rank!=0）warp3 做 repack——
  claim_task(dep_blk_cnt 当计数器) 领 expert → 等 R 个 pair count → 顺序 cumsum 得 dense offset →
  拷贝 acts(hidden B) + weight(4B) + meta(12B) + sf(块内 pos*4 布局 160B) → st_release_u32 发布
  dep_expert_base[e]=total+1。scheduler 按 dense 解码（bound=local_experts*min(eblocks,R*pair_blocks)），
  publish pool_block=dense_block+e*eblocks+m；消费端/epilogue 零改动。
- 已修 bug：footer 清扫范围 pool_blocks=total_pair_blocks→pair_rows//block_m（dense 块 1536-4415
  的生命周期计数器原来永不清，call≥2 挂死）。
- 环境：env_b300.sh + GC_MOE_PUSH_DISPATCH=1 + torch.distributed.run nproc=8 + routing_off.npy bm32。
  注意：主机 GPU 与其他用户作业共享（lingqingyan symmetric_ae_scan），性能数需等空卡。
- 之前失败教训：TYPE_UNSTABLE_JOIN 用 while/cutlass.range 规避；make_ptr 要 Uint8 重铸或裸 Uint64 地址；
  卡死定位法 cuda-gdb -p <worker> -batch -ex "info cuda warps"（attach 到 worker 不是 torchrun 父进程）。

## pull dispatch 优化：融合单遍 + count 发布并行实验（2026-11-19，deepep-dispatch 分支）
- **背景**：EP8 plain pull trace（bm32，T=128，routing_off）：barrier ~10µs、count_ready ~13-20µs（spread ~0.4µs）、l1_first ~19.7µs、call ~194µs。count_ready−barrier ≈ 6µs = block0 单 CTA 发布（384 expert × st+atom）+ NVLink 传播 + dst 轮询。
- **融合单遍 dispatch（已实现，bit-exact 验证，env `GC_MOE_FUSED_DISPATCH` 默认 1，=0 回退）**：pass1 直方图 + merge + pass2 索引重扫 → 单循环 `slot=atom_add_u64(send_count[e],1)`（gmem 原子，返回值即 slot、终值即 count）+ 远端 `dst_index` 写；发布时把 `(dispatch_sms<<32)|count` 补标。send_count 复用 role3 footer 每 call 清零区域，零新增状态，dst/scheduler 协议零改动。
  - **必须 gating 的点**：pass2 循环靠 `pass2_end = route_base if (push_dispatch or fused_dispatch)` 置空跳过——漏掉会让旧 pass2 用未初始化 smem 计数二次写 index，输出错（max|diff|~8.6，dump-y A/B 抓到）。
  - **性能结论（EP8 A/B，4 calls）**：barrier/count_ready/l1_first/pull_end 全部持平（±0.5µs 噪声内），总时延 fused 200.4 vs 非 fused 207.9（单次，噪声 ±5µs）。**dispatch 段瓶颈是 768 次/ rank 远端 index store 本身，融合只省直方图+重扫 ~1-2µs，被 gmem 原子 slot 依赖吃掉大半**。保留：结构更简、EP32/64 routes 更多时收益可能更大；默认开。
- **count 发布并行化（已实现→实测回退）**：全部 dispatch CTA 按 expert 连续分块发布（CTA b → experts [b*48,(b+1)*48)，恰好=dst b 的全部 local experts）。结果：barrier −0.7µs、count_ready −0.8µs **改善**，但 l1_first **+2µs 恶化**、总时延 **+8~15µs 恶化**（两次复测一致，非噪声）。推断：8 SM 同时向各 dst 打 48 个远端 atom 的瞬时突发在 dst L2 形成 atom-RMW 积压，挤占随后 pull 读的 fabric/L2 带宽；block0 串行发布的 ~3µs  trickle 反而匹配 dst 排空速率。**已回退为 block0 发布**。教训：远端原子突发的时序模式比总数更敏感，发布并行化此路不通。
- **剩余 count_ready→l1_first 间隙（~5µs）构成**：发布 issue ~2-3µs（block0 两 loop 各 3 轮远端 op）+ 传播 ~1-2µs + dst acquire/轮询 ~1µs。可挖空间 ~1-2µs（如 block0 单 loop 融合 st+fence+atom 省一轮），收益有限。
- **bit-exact 方法**：`--check --dump-y DIR`（dump-y 必须配 --check 才生效，在首个采样 call 存 `<mode>.rank<R>.npy`）；跨 env A/B 用 `GC_MOE_FUSED_DISPATCH=0/1` 各跑一次再 np.abs diff——fused 修复后 8 rank 全 exact。
- **环境备注**：fused=0 的 dump run 出现过一次 7.7ms 异常慢（疑共享主机/首启），y 输出仍有效；EP32/EP64 集群仍被 sat_app 角色卡住，EP8 本地是唯一验证环境。

### pull dispatch 深挖：publish 才是真瓶颈（2026-11-19 补充）
- **分段探针**（`_eco_trace_rel` slots 37-40 + bench `probe_ns`，GC_MOE_ECO_TRACE=1 plain 也写）：
  37=route-loop 起点 ~1.0µs | 38=pass 末 ~2.0µs（**融合 pass 仅 ~1.0-1.2µs**）| 39=grid_sync 后 ~3-4µs |
  40=publish+sync 后 ~9-10µs（**block0 发布 ~5.5-6.5µs = dispatch 元数据的 60%+**）| barrier f0 ~10µs。
- **publish 成本结构**：block0 单 SM 发 384 matrix st + 384 `atom.relaxed.sys`（每线程 3 轮串行远端 op）——远端 op 注入速率/延迟 bound，非带宽。
- **优化尝试**：① count 发布 8-CTA 分块（pubpar）→ count_ready -0.8µs 但 atom 突发在 dst L2 背压，总时延 +8~15µs **回退**；② `atom_sys_add_u64`→`red_release_sys_add_u64`（fire-and-forget release 归约，同时替代 fence_acq_rel_sys）+ **pubpar 复用**（red 无 RMW 回读，突发对 dst 更温和）→ probe 显示 publish ~6.5→~3µs，**bit-exact 8 rank 全过**（y_cur vs y_off exact）。干净计时待 GPU 空闲（13:00 起 rtp_llm 8 卡作业 100% 占用）。
- **认知修正**：index 与 count 在 fused 单遍里本就是同一原子操作的两面（slot 分配+计数累计），不存在"先 index 后 count"的串行；count 无法更早的根因是 count[e,src] 需全部 routes 扫完。真正的串行段是 publish 的 768 次远端 op。
- **剩余间隙**：count_ready(~16)−barrier(~10) ≈ 6µs = dst 自身 dispatch 未完成不能 poll + NVLink 传播 + acquire——dst 侧 poll 前移（与 publish 重叠）是下一个候选但需防死锁（wait/publish 循环依赖）。

### x-order 首块优先 dispatch（2026-11-19 设计定稿；阶段 A 已实现待验证）

完整设计文档：**`XORDER_DISPATCH.md`**（仓库根目录）。要点：
- 核心思想：发送序 ≡ 池序 ≡ 消费序，统一 x-order（每 dst 内按历史热度降序、全局按 dst 轮转交错）；src 按 x-order 发 count、dst pool_block 前缀按 x-order 算 → 早到 count 直接对应早 pool_block。
- x-order 在**专家权重重排时生成一次、复用至下次重排**（零在线成本，隐式商定）；bench 先从 routing_off.npy 聚合 top-K/dst 注入。
- 已证伪：per-src 独立首块选择（8 src 交集≈0）、在线商定（成本否决）、静态 top-1（命中率~5%）；静态 top-8/dst best-of 期望 ~85 tokens/call（oracle 125）。
- dst 侧约束：count 等待 per-lane 独立但 warp wave 末尾收敛 → 主线是 **x-order 驱动的渐进调度**（count 到一个排一个，逻辑最简；两阶段只是其特例）。
- x-order 选集：**greedy max-coverage**（直接优化 E[本call集合内最热expert tokens]，K=8/dst → ~90 tokens、P≥32 78%；K=12 → ~106/90%）。热度跨段稳定 ~90%，单表即可。热度必要性：GEMM 块按 bm 补齐，半空块占 TC 时间等同满块——热度序让早窗口填满块（~2-4× 早期吞吐）。
- **ECO 约束**：早发布集只能选 fixed/单副本 slots（replica 的 count 走 eco_quota ~15-35µs，dispatch count 无人读）；表生成按 EcoMeta num_fixed/phy2log 过滤候选。
- 预期：l1_first ~18-20→~10-14µs（物理下限 ~8-11µs 的 ~85-90%）；call -3~4%。pull 吞吐不变（带宽 bound）。
- 实施分 A（表+src 按序发布）/B（dst 渐进调度）/C（可选残余）三阶段，env `GC_MOE_XORDER` 回退。
- **阶段 A 实现（2026-11-21，未提交）**：`GC_MOE_XORDER=1` + `GC_MOE_XORDER_ROT=0/1`（默认 1）。
  表经 role1 空闲的 `tensor_maps` 槽位传入（non-eco 专用；eco/push 不兼容，prepare 侧校验+静默关闭）。
  发布循环把 canonical expert-id 序改为 per-src 相位旋转序：位置 p→轮 j=p/R、偏移 k=p%R→
  dst=(rank+1+k)%R、local=table[dst*L+j]；两循环（matrix 行+sum atom）同序。CTA 仍按位置分块。
  链路：`fp8_fp4_gc_moe(xorder_table=t)` → `launch_gc_moe` → `prepare_role_binaries`（校验
  int32 CUDA [R*L]、cache key 加 flag、compile_args pos5=xo_view、role1_args 7 元组）；
  `_binary_key` 含 kernel vars 自动区分 cubin。bench `--xorder K`（plain 模式）从 off 聚合
  greedy 表（`build_xorder_table`，已在 ep8 routing_off.npy 验证：置换正确、早集/晚集热度比 ~1.4×）。
  阶段 A 无时延收益（dst 仍等全量 count），价值=正确性验证+阶段 B 地基+ROT 开关可单独 A/B
  （理论上消同地址 atomic 串行或反而加速 publish）。待 GPU 空档：--check bit-exact + trace。

### x-order 阶段 B 结果 + native 渐进 pull（2026-11-21/22，deepep-dispatch 分支，未提交）

**EP32 三方 trace 结论（p384:plain:384, bm96, T=128, routing_off.npy 真实路由, 8 calls × 32 rank）**：

| µs（slowest-rank 中位） | native | xorder+lazy off | xorder+lazy on |
|---|---|---|---|
| barrier（route+pub 发完） | 10.0 | 12.1 | 12.1 |
| count 全到（field12） | 17.3 | 22.4 | ~21（lazy 跳过轮询） |
| pull_end | 36.3 | 41.9 | 40.8 |
| l1_first | **24.9** | 30.3 | 29.2 |
| gemm_done | 80.9 | ~88 | ~87 |

**x-order 在 EP32/bm96 净负 ~4.3µs l1_first**：前端 +2µs（smem staging+自行发态+旋转发布）、
首块=最热 expert 满块 96 token（vs native local0 ~64 token 半块），"热 count 早到"不成立
（每 dst_local count 仍需 32 src release-atomic 齐到，~15-20µs）。XORDER_DISPATCH.md 的
~10-14µs 目标是 EP8/bm32 外推，无效。`GC_MOE_XORDER` 保持默认关。
B1（x-order pool 布局）+ B2a（scheduler 渐进发布）+ B2b（lazy pull `GC_MOE_XO_LAZY`）
全部 bit-exact（32 dumps MATCH）；期间修了 peek-claim 竞态 + off-by-one 两个死锁 bug。
lazy pull 净效应 ~-0.7µs（count 簇到，skew 仅 ~1.2µs < 轮询开销）。

**native 渐进 pull（prog_native，用户选定"基于 native 微调"）——代码已完成，未验证**：
- 机制：复用 advance_expert_xo/xo_cover/xo_wait_position 全套，加 `native_order` constexpr
  （poll_expert=pos 恒等映射，无 xo 表无发布行）。role1 pull 游标按 local expert 序
  0..L-1 到哪个等哪个 count；role2 scheduler 同序渐进摄入，routed_m_blocks 渐长。
- impls：`self.prog_native`（env `GC_MOE_PROG_PULL` 默认 '1'；要求 not xorder/eco/push、
  pull_pipe+chunks==1+stages==2）。bulk count 等待跳过（`not (xo_lazy_pull or prog_native)`×2）、
  两处 advance_expert_xo 调用点 + post-pull own_pool_blocks 门控放宽、
  `tensor_maps if tensor_maps is not None else expert_counts` 兜底 xo 参数、
  schedule_tasks 调用加 `self.prog_native`。
- scheduler：`prog_native` param；count-init `elif xorder_pool or prog_native`（flag 等待
  仅 xorder_pool 内）；两处 xo_cover 调用 `xorder_pool or prog_native` + 传 prog_native 作
  native_order + `xo_ptr if xo_ptr is not None else recv_ptr`；l1_warmup seed 同放宽。
  locate 后 `local_expert=xo_ptr[..]` remap 保持 xorder_pool-only（native 恒等自动跳过）。
- **待办**：推送 `gc_moe/scheduler/gc_moe.py` + `gc_moe/impls/sm100_fp8_fp4_gc_moe.py` 到
  8 节点 **`$W/ep64s/pkg/gc-moe-u30/gc_moe/`**（l20a_launch.sh 的 PYTHONPATH/bench 均指向
  ep64s/pkg/gc-moe-u30，非 sgpkg/gc-moe！）；先 `--check --dump-y` bit-exact（32 dumps），
  再 trace A/B（`GC_MOE_PROG_PULL=1` vs `=0`，同 tr_native 参数无 --xorder），
  对比 l1_first/pull_end/total vs native 基线（24.9/36.3/152.8 trace, ~136 非 trace）。
- **集群变更（2026-11-20）**：job 重启为 session S915073，wt sat_app 授权失效；
  新入口 `ssh hole-<IP>`（用户提供）。新集群 = task lk_L20A_35554b93.inference 的
  parts {0,1,10,11,12,13,14,15}：node-rank→IP = 0:.79(part0,master) 1:.72(part1)
  2:.87(part10) 3:.96(part11) 4:.92(part12) 5:.76(part13) 6:.90(part14) 7:.88(part15)。
  包树 = `$W/sgpkg/gc-moe`（gcenv/env.sh 的 PYTHONPATH），bench =
  `sgpkg/gc-moe/benchmarks/benchmark_gc_moe_eco.py`。启动脚本 `l20a_hole_launch.sh`/
  `l20a_hole_run.sh`（~/work/mpo/ep64/）。无 env 的 part{3,4,5,7,8} 不可用（.84 另有外部 GPU 负载）。
- **prog_native 实测（2026-11-20，新集群 EP32 p384:plain:384 bm96）**：
  bit-exact 通过（32 rank dump ON/OFF 全 MATCH）；非 trace 计时 ON 114.36 vs OFF 113.59µs
  （**+0.77µs 净负**）；trace（GC_MOE_ECO_TRACE=1, 8 calls）l1_first 26.63 vs 27.27（-0.64µs
  机制生效）但下游 +1~1.5µs → 总 142.00 vs 141.18。**结论同 xo_lazy_pull：count 在 EP32
  成簇到达，渐进等待的逐位置 poll+sync 开销 > skew 节省，净负**。
  `GC_MOE_PROG_PULL` 已改默认关（=1 才启用），代码保留作对照。
  `GC_MOE_PROG_PULL` 已改默认关（=1 才启用），代码保留作对照。

- **sharded count sums（GC_MOE_SHARD_COUNTS，用户选定）——已实现+bit-exact+实测：中性，默认关**：
  设计：src 把 tagged count（(dispatch_sms<<32)|count，release-atomic）写进
  `recv_count[src*L+dst_local]` 自有槽位（替代原 loop1 plain store + loop2 count_sum
  原子两条远端 op → 只剩 1 条/位置）；block0 的 dispatch warps（warp=expert，lane=src）
  轮询 32 个槽、warp_redux 求和、`red_release_sys_add_u64` 写回 `count_sum[local]`
  （格式 (num_ranks*dispatch_sms)<<32|total 与原子累加结果完全一致 → 全部消费端零改动：
  role1 bulk wait、role2 schedule_tasks、role3 stats、prog_native/xo 路径都不用动）。
  recv_count 明细读取方全部 `Uint32()` 截断取 low32=count，tag 高位无影响。
  - **坑1（已修）**：gather 轮询内的 `read_clock64()` 超时 + `device_trap()` 触发
    ptxas/cutlass 误编译 → IMA/unspecified launch failure。去掉超时（纯自旋）即正常——
    发布侧保证全 (dst,local) 槽必到，死等只在真 bug 时挂。若要恢复超时需换写法再验证。
  - **坑2（已修）**：`cutlass.range` 的 Uint32 实参触发 TYPE_UNSTABLE_JOIN；改 while 循环
    + 显式 Uint32 变量规避。
  - 实现位置：impls `self.shard_counts`（env `GC_MOE_SHARD_COUNTS` 默认 '0'；要求
    not xorder/eco/push_dispatch）；发布 loop1 的 st 改 `if not shard_counts`，
    loop2 的 remote_sum 在 shard 下指向 recv_count 槽；gather 块在发布 sync_threads 后、
    `_eco_trace_rel(40)` 前（block0 only）。
  - **实测（EP32 p384:plain:384 bm96，同集群）**：bit-exact 32 rank dump 全 MATCH；
    非 trace 16-call mean 113.36 vs native 113.59（-0.2µs 噪声级）；trace（同 tr_native
    8 calls）：barrier 字段（=gather 完成≈count-ready）16.83 vs 原生 field0 8.9
    （原生 count-ready 在 field12≈17.3）→ **count-ready 两种协议都是 ~17µs，原子排干
    不是瓶颈**；l1_first 24.91 vs 24.96、pull_end 34.04 vs 33.89、gemm_done 76.45 vs
    76.52、role3 131.8 vs 132.4 → **全持平**。
  - **结论**：~17µs count-ready 由 src 侧 route-pass 完成 skew + NVLink 飞行时间主导，
    dst 侧原子串行排干可忽略；且 l1_first 本来就被首块 pull 投递（~17µs）卡住，
    count 提前也传导不到 l1_first。**这是第 4 个被证伪的 count 侧方案
    （x-order / xo_lazy / prog_native / shard sums）——count-ready 与首块投递都是
    fabric-bound，dst 侧无解。下一个真杠杆只剩 pull 投递路径本身（Stage C）。**

## native pull 细粒度 profiling（2026-11-20，deepep-dispatch 分支）

基础设施（已部署 L20A hole- 集群 8 节点，md5: impls a4ac753f, bench c63f8644, layout 16b64dbb）：
- `ECO_TRACE_RING_FIELDS` 16→32；role1 新增 14 个相位戳（ring fields 16-29，ns since role1 start，
  u32；min 字段走 inverted-max：13,14,16,20-25；max：17-19,26-29 中 26/27/28，单写者 store：18/19/29）。
  串行 pull 路径与 _pull_pipeline 各有一套同字段戳（本 config bm96 走串行 while 循环）。
- bench `--profile-calls N --profile-dir DIR`：计时后把 call() 抓成 CUDA graph，torch.profiler 下
  replay N 次（rank 间保持启动同步，避免 eager ~ms 错开吹大跨 rank 等待），导出 kineto rank<R>.json
  + ring 行 profile_ring_ns [rank][call][32]。
- 本地合成：`~/work/mpo/ep64/merge_perfetto.py <bench.json> <rankdir> <out.json> p384` →
  Perfetto/Chrome JSON（ui.perfetto.dev Import JSON）。role 按 grid 区分（16=role1 dispatch/768thr，
  136=role2 compute，64=role3 finalize+bf16），call 按 kineto correlation id 聚类（每 replay 一组），
  ring 行锚定 role1 kernel ts（kineto µs = ring ns/1000，period 校验 188.4=188.4 ✓）。

run6（prof_native6）32-rank 稳态相位（med over 32r×8calls，集群退化期 ~+25%）：
route 0.4→17.7 | grid_sync+join →22.1 | pub1 22.8 | pub2(count_sum原子issue) 26.0 | f0 26.7 |
counts_ready 54.3（等原子排干 ~27.6µs）| pull_entry 54.5 | first_issue 55.8 | sf 56.4 |
x_arrive 56.7 | s2g 57.0 | pub_min 59.9 | l1_first 62.4 | pub_max 74.2 | pull_exit 74.7 |
role1_exit 75.3 | role2 GEMM ~110.7 | role3 ~68.9 | period 188.3。
kernel dur: role1 77.8 / role2 110.7 / role3 68.9µs。首块窗口 counts_ready→l1_first ≈ 8µs。
注意：x_arrive 仅 ~0.3µs 是 L2 热数据假象（replay 复用同一 x）；冷启动会更长。
集群状态：11:11 后 fabric 相位整体 ~2×（tr_native 17.4µs counts vs 54µs），timed median
130→166µs；GPU 空闲、温度正常，疑似租户 NVLink 争抢，非插桩开销（guarded 单写者戳）。

## native pull 时间线标准测试流程（nat_tr1 方法，2026-11-20 确立，以后就这么测）

要"某模式的 32-rank 相位时间线"一律走这条：benchmark 直跑 + ring 打点 + trace-calls，
**不要再用 prof_wrap.py / --profile-calls / eager profiling 那套**（那些是 kernel 边界
视角；相位分解只认 ring）。

### 跑法（L20A hole- 集群 EP32，8 节点 × 4 卡 = 32 rank）

- 代码：本仓库 `deepep-dispatch` 分支；改动后把 `gc_moe/`、`benchmarks/` 推到 8 节点
  `$W/sgpkg/gc-moe/`（`ssh hole-<IP>` 逐文件 `cat | ssh` 推送 + md5 校验；包树是
  sgpkg/gc-moe 不是 ep64s/pkg）。
- 命令模板（`~/work/mpo/ep64/`）：

  ```bash
  ./l20a_hole_run.sh <tag> "GC_MOE_ECO_TRACE=1" \
    --off sgpkg/data/routing_off.npy --calls 20 --block-m 96 --cold-l2 \
    --mode plain384:plain:384 \
    --warmup 1 --iters 2 --inner-iters 2 --trace-calls 3 \
    --output-json runs/<tag>.json
  ```

  - env 必须作为第 2 个参数传（放前面只影响本地 wrapper，远端 torchrun 收不到）。
  - `--off` 用节点上的 `sgpkg/data/routing_off.npy`（已是 EP32 版 [1200,32,128,6]）；
    `--output-json runs/<tag>.json` 由 l20a_hole_run.sh 从节点0抓回 `results/se/<tag>.json`。
  - mode 按需换（native pull = `plain384:plain:384`；其它见 §8.39 mode 集）；
    shared expert 默认 1（gran_k 32），`--num-shared-experts 0` 为对照。
- 产出：`results/se/<tag>.json` → `results.<mode>.per_call`（20 个均匀采样 call_id）×
  `trace_us`（8 个命名里程碑）+ `trace_raw_us`（32 rank × 32 ring 字段，µs 自本 rank
  role1 起点）。trace-calls 每 call 建"flush+清 ring+call"graph 回放 3 次取稳态、
  3 个 run 取中位（§8.39 已述）。

### ring 字段表（ECO_TRACE_RING_FIELDS=32，µs since role1 start）

聚合语义：inverted={10(仅 plain),13,14,16,20,21,22,23,24,25} 存 `0xffffffff-elapsed`
取**最早**；`is_max=True` 取**最晚**；`is_max=False` 仅 block0/tid0 单写。
bench 已自动反演 {13,14,16,20-25}；**字段 10 需手动 `4294967.295 - v` 反演**。

| f | 含义 | f | 含义 |
|---|---|---|---|
| 0 | post-publish 点（pub2 后） | 16 | route pass start（最早） |
| 1 | role3 start = gemm_done | 17 | route pass done（最晚 warp） |
| 2 | last shared L1 start* | 18 | grid_sync done（b0t0） |
| 3 | last routed L1 start | 19 | recv_count publish issued（b0t0） |
| 4 | route 尾（pass 间） | 20 | pull loop entry（最早） |
| 5 | last shared L2 start* | 21 | first bulk_g2s issue（最早） |
| 6 | pull_end | 22 | SF+weight done（最早） |
| 7 | last routed L2 start | 23 | first x arrive/mbar done（最早） |
| 8 | last role2 CTA exit | 24 | s2g done（最早） |
| 9 | role3 done | 25 | first l1_full publish（最早） |
| 10 | first shared L1 start*（plain 倒置） | 26 | 同 25（串行路径同点戳） |
| 11 | routing/period（多写者，慎用） | 27 | pull loop exit（最晚） |
| 12 | counts held（最晚 warp） | 28 | role1_exit |
| 13 | first routed L1 inputs ready | 29 | count_sum publish issued（b0t0） |
| 14 | first routed L2 start | 30/31 | 未用（plain） |
| 15 | eco 专用（role2 input wait ns） | | |

*shared 系字段仅 `shared_experts>0` 且非 eco 时写。串行 while 路径与 `_pull_pipeline`
同字段两套戳；bm96/T128 走串行路径（run4 实测 fields 20-27 只在串行戳有效）。

### 可视化

```bash
python3 ~/work/mpo/ep64/nat_tr1_chrome.py results/se/<tag>.json <mode> results/<tag>_chrome.json
```

→ Chrome/Perfetto JSON：pid=call_id(20) × tid=rank×{dispatch,compute}(64) × ~15 相位带
（dispatch：prologue/route/grid join/publish/wait counts/pull 五段/tail；
compute：shared L1+L2/wait routed/L1 GEMM/L2 GEMM/combine）。
打开 https://ui.perfetto.dev → Import JSON。参考产出：`~/work/mpo/ep64/results/nat_tr1_chrome.json`
（nat_tr1 run：plain384 bm96 cold-l2 shared1，10900 events）。

### 注意

- trace 模式下 JSON 打印的 mean/median **不可用**（trace run 污染计时），只看时间线；
  要性能数字另跑不带 TRACE env 的计时（§8.39 标准 A/B）。
- `--cold-l2` 每 replay 前刷 L2：绝对值偏大，相位结构有效。
- shared expert 相位结尾 = `f5 + 2.5µs`（shared L2 GEMM 时长估计，sse_tr1 验证
  64 个 rank-mode 对误差 0；换 bm/配置需重估）。
- nat_tr1 call631 中位数参考（集群健康期）：route_done 11.6 / counts held 42.3 /
  pull_exit 62.9 / l1_first 51.3 / gemm_done 103.9 / role3_done 167.9µs；
  dispatch 全程 ~58µs vs compute ~168µs，compute 是关键路径。

### nat_eplb 变体（2026-11-20，同流程换 eplb448 routing）

同 nat_tr1 流程，仅 `--mode plain384:plain:384` →
`--mode eplb448:eplb:448:sgpkg/data/routing_eplb448.npy`（EP32 数据集
`~/work/mpo/data/ep32/`，md5 3147ac95…，已推 8 节点 `sgpkg/data/`；
**注意 hole 集群 `sgpkg/data/routing_off.npy` = data/ep32 版 md5 8989b4…，
不是 r32d 版**——eplb 文件必须与 off 同数据集）。产出
`~/work/mpo/ep64/results/nat_eplb_chrome.json`（10893 events）。

**⚠️ 跑前必须验卡 + 注意 fabric 噪声**：hole 集群是 lk_L20A 共享推理任务的
parts {0,1,10-15}，NVLink fabric 域横跨整个任务，**其它 part 租户的流量会
和我们跨节点 count/pull 原子争抢同一条 fabric，nvidia-smi 看不见**。实测
run-to-run 噪声 ±10µs（dispatch 相位）。规程：跑前 `nvidia-smi` 确认 4 卡
util 0%（GPU 级），但 fabric 级只能靠重复 run / 对照识别；单次 run 的
fabric 相位绝对值可能虚高 ~10µs，**A/B 结论要用干净窗口的重复 run 确认**。
（nat_eplb_v1 即被污染：counts_ready 35.9 vs 干净 26.0；nat_tr1 不敏感因
自身 routing 不均衡淹没噪声。）

干净数据（v2，已验证 8 节点 4 卡全 0% util，20-call 中位 µs med[rank min..max]）：

| 相位 | nat_tr1 plain384 v2 | nat_eplb eplb448 v2 | nat_eplb_r32d |
|---|---|---|---|
| route_done f17 | 11.3 [9–25] | 11.3 [5–15] | 11.4 [5–15] |
| wait counts f12−f29 | 15.2 [3–22] | **8.6** [3–16] | 8.2 |
| counts_ready f12 | 38.7 | **26.0** | 25.6 |
| pull_exit f27 | 59.4 | **48.6** | 48.0 |
| l1_first f13 | 47.1 | **35.2** | 35.2 |
| gemm_done f1 | 99.3 | 96.1 | 95.4 |
| role3_done f9 | 162.8 | **106.9** | 106.1 |
| shared band 宽 | 24.3 [18–34] | **17.0** [12–22] | 17.1 |

结论（修正版）：
- **eplb448 在干净数据上收益比 v1 以为的大**：counts −12.7µs、wait −6.6、
  l1_first −11.9、role3_done −56µs、band −7.3µs。
- **ep32 vs r32d eplb448 routing 性能完全等价**（v2 vs r32d 逐项相同）；
  之前"ep32 更难"是 v1 污染误导。
- **shared band ~17µs 是 hole 集群基线**（v1/v2/r32d 三次稳定 17.0-17.3），
  sse_tr1 的 9.9µs 是 e32 专用空闲机的水平——band 宽度主要由集群 fabric
  质量决定，routing 内容只影响最差 rank 尾部（ep32 曾 33，干净后 22）。
- wait-counts 窗口 eplb 仍更窄（8.6 vs 15.2）但非归零：max 由最慢 rank
  route pass 决定，eplb 把 skew 换了 rank 没消除（route_done 极差仍 ~10µs）。

## native pull 启动段 stall 定位：三轮 trace 结论（2026-11-20 晚，nat_st1_v2 / nat_st2_v1 / nat_st3_v1）

问题：role2（compute）从 kernel 启动到进任务循环有 ~13-25µs 的 stall，且尾部精确跟着 route_done 走
（f2 末次 shared L1 claim ≈ route_done ±1µs，corr 0.98）。三轮插桩 trace（EP32 L20A hole，T=128，
block_m 96, shared 1, 20 calls, GC_MOE_ECO_TRACE=1, ring 扩到 36 字段）逐条排除：

| 假设 | 戳 | 数据 | 结论 |
|---|---|---|---|
| tensormap prefetch 风暴（~2700 次 descriptor fetch） | f15 warp0 prefetch 后 | **1.4-1.7µs max，20 call 恒定** | **证伪**（异步指令不占 issue） |
| `griddepcontrol_wait` PDL 门控 | f33(barrier1 后)→f34(init 返回) | **+0.4µs** | **排除**（sm100 无 launch_dependents） |
| `recv_parity` volatile 全局读被洪水堵 | f34→f35 | **+0.5µs** | **排除** |
| barrier 后路径整体 | f33→f30 | **~1.3-1.7µs** | 全部快 |
| CTA 启动错峰 | f15（136 CTA 的 warp0） | **1.7µs 全部启动完** | **排除** |
| SM 争抢 | Green Context 分区 | compute+dispatch==total | 已排除（前作） |
| **最后到达 init barrier 的 CTA 与 route_done 的关系** | f32（每 CTA warp0，136 原子） | **corr(max, f17)=0.929**，med 11.0 / max 22.3µs | **唯一强相关量** |

机制判断：stall 全部在 `_initialize_shared_storage` 的 cluster barrier 前/中；barrier 之后的
griddep/volatile 读/指针设置合计仅 ~1.3µs。f15→f32 之间只有 ~10 条纯算术指令，却量到 10-20µs，
说明**插桩本身（每戳 2 次 volatile 读 + 1 atom_max）在洪水峰值期被 L2 拖慢**——这反过来证明：
该窗口内任何非缓存全局访问都会被 route 写洪水拖 10µs 量级。role2 启动段没有可"消除"的全局读，
**SMEM tensormap 方案无效**（prefetch 已证无罪）。

推论：shared band 拉宽的根因 = route pass 的 **24.5K 笔散乱 4B 远端 source_idx 写**造成的 L2/NoC 饱和，
不是 role2 的某条读。修复杠杆只有两个方向：(a) 降洪水强度（减少远端写事务数），(b) 时间错开。
self-route（dst 自算路由，消灭全部远端 index 写）是唯一能同时改善 band 和启动确定性的方案。

仪器注意：ring 字段 32-35 为本轮新增（layout ECO_TRACE_RING_FIELDS 32→36），32=CTA 到 barrier、
33=barrier1 放行、34=griddep 后、35=recv_parity 后；f32 必须每 CTA 一个 warp（全 warp 会造成
1632 原子/rank 串行化 ≈12µs 自扰，nat_st2_v1 即为此污染版）。
数据：results/se/nat_st1_v2.json（f15/f30/f31）、nat_st2_v1.json（污染版）、nat_st3_v1.json（干净版）。

## self-route 实现与实测（2026-11-20 夜，deepep-dispatch 分支，未提交）

结论：**self-route 已完整实现并 bit-exact 通过，e2e 中位 −13 µs（≈−8%）**，机制与三轮 trace 的
推断完全一致（role2 启动段 cluster barrier 从 ~33 µs 塌到 ~2 µs）。实现细节、踩坑、相位数据
全部写在 `SELF_ROUTE_PLAN.md`（已更新为完成态），这里只留要点：

- 开关 `GC_MOE_SELF_ROUTE=1`（默认 0）。**pull 是 role1 做的**，所以整个方案在 role1 内部：
  每个 dispatch CTA 独占整条 source（`src = block + k*dispatch_sms`，waves=ceil(R/dispatch_sms)），
  每 src 一次 `bulk_g2s_1d` 读 peer 的 `num_tokens*topk*8` 字节到空闲 pull staging 区，
  768 线程并行扫描后本地写 `source_idx` + `(src,expert)` count，sum 用 GPU-scope release atomic
  打 `(dispatch_sms*owned)<<32|subtotal` 让原到达门闭合。role1 跳过远端 index 写和 count 发布，
  保留 per-rank `grid_sync`（把本地行发布给自己 CTA 的 pull 循环）。
- 三个隐蔽坑：**capacity=1920 ≠ T=128**（第一版按 capacity 定缓冲 → 守卫静默禁用，跑出来的
  "on" 其实是旧路，务必看日志有没有 `GC_MOE_SELF_ROUTE=1 ignored`）；count 发布必须
  `physical_warp == 0`（全 warp 会 24× 打 tag → recv-count timeout trap）；cute-dsl 不允许
  dynamic for 套 dynamic if（TYPE_UNSTABLE_JOIN），要用 masked stop 把循环提出来。
- 实测（EP32 hole，T=128 bm96 cold-L2 20 calls，off/on 交替）：off 中位
  172.2/166.2/175.3/169.0/163.4/161.7/164.4（mean 167.5），on 152.9/152.0/154.1/156.3/155.2
  （mean 154.1）→ **−13 µs / −8%**，Mann-Whitney p≈0.0013，两组完全不重叠。bit-exact：
  32 rank × 655,360 bf16，max|diff|=0。
- 相位（trace 横向对比）：pull_entry 47.8→32.3、pull_exit 109.3→83.7、role2 scheduler 33.3→2.1、
  role2 barrier1 33.2→1.9、gemm_done 168.6→140.2、role3_done 173.4→145.2 µs。
  扫描本身只花 ~2.1 µs（f16 earliest 0.83→2.98）。
- 稳定性：on 6 次里 1 次崩（rank8 首发 LAUNCH_FAILED，其余 rank token-readiness timeout peer=8），
  off 7 次 0 崩；与本集群历史 flaky 故障同形态，样本不足以归因，采用前需更多重复。
- 后续价值排序：扫描分块支持大 T（T≥2048 会被守卫禁用）；稳定性采样后再谈默认打开。
  注：f16/f17 的 27 µs 差大概率是插桩自扰（384 次 atom_max 打同一地址），不是真实 CTA 错峰；
  **不要扩 ECO_TRACE_RING_FIELDS**——本轮把它 36→40 加干净戳，trace 模式连续两次挂
  （token readiness timeout / SIGTERM），退回 36 立刻恢复，要加戳就改现有字段的发射者。

## shared 耗时不均的根因排查（2026-11-21/22，EP8+EP16，deepep-dispatch）

结论先行：**shared 的跨 rank 差异与 pull / routed / barrier / 内存带宽全都无关**；它是
"同一块 shared tile 在特定 GPU 上执行时间本身会 1.6–2.4× 波动"，且只在多 rank（symm_mem 分配）
下出现、随 rank 数放大。下面是把这条路走完的完整口径与排除清单，避免以后重复踩。

### 新测量工具（都在工作树里，默认关，不改变生产路径）

- `--dump-shared-dbg DIR`（bench）：落 `shared_dbg.rank<R>.npy`(68×2, 每 cluster `~start/done`)，
  `shared_dbg2.rank<R>.npy`(68×4, 见下)，`trace_ring.rank<R>.npy`(64×36)。
  eager 路径现在也会分配 ECO 区（`args.eager_iters or args.dump_shared_dbg`），否则 trace ring /
  role1-start 槽在 buffer 之外（UB）。
- `SHARED_DBG2_OFF`（layout，byte offset，16B 对齐）= `expert_send_count` 字段尾部；
  u64 槽位（注意 `signals.iterator` 是 Uint32 指针，**+N 是 N 个 word = N/2 个 u64**）：
  `[0]` L1 总工时 ns、`[1]` L1 任务数、`[2]` L2 总工时 ns、`[3]` L2 任务数、
  `[8]` ~最早 CTA 启动（min）、`[9]` 最晚 CTA 启动（max）。工时用 `sh_t0`（每个 epilogue warp 在
  任务开始时自己读 `read_globaltimer()`）累加，role2 的**三份任务循环（warp0/warp1/warp>=4）各要一份**。
- 探针开关（全部默认 0，输出可能错，只用于归因）：
  `GC_MOE_PULL_META=1|2`（peer 的 topk-weight / +SF 整表 bulk 到本地 scratch，pull 改本地读）、
  `=3`（完全不读远端 SF/weight）、`=4`（不写 ring）、`=5`（源地址全本地化）；
  `GC_MOE_NO_PULL=1`（role1 只路由，需配 NO_ROUTED）、`GC_MOE_NO_GRIDSYNC=1`（跳过 rank barrier）、
  `GC_MOE_SCAN_LOCAL=1`（self-route 扫描读本地表）、`GC_MOE_SYNC_BACKOFF=<ns>`（grid_sync 自旋退避，
  `nanosleep_u32`）、`GC_MOE_NO_ROUTED=1`（role2 不跑 routed）。
- 运行器：`/data0/home/nanjun.cp/work/mpo/ep64/pm_run.sh <tag> <hosts-csv> <nnodes> "<env>" <args>`，
  自带空闲门禁（apps==0 且 max_mem<=200 MiB）、结果落 `results/se/<tag>.json`；
  `python3 /tmp/shsum.py`（或按需重写）打印 `eager_shared_ranks` 的 med/min/max/spread。
  EP8 = `.72`+`.96`（数据在 `sgpkg/data/routing_id8_t96.npy` / `_t192` / `routing_id{2,4}.npy`）。

### shared 的真实几何（解读一切数字的前提）

`l1_clusters = (2*intermediate/128)//2 = 18`，`l2_clusters = (hidden/128)//2 = 20`；
L1 任务 = `m_blocks*l1_clusters`（T=128,bm96 → 36），L2 任务 = `m_blocks*l2_clusters`（40）；
`publish_shared_phase` 在 `total <= compute_sms//2` 时走**静态分配**（cluster c 领 task c），否则全局认领。
**L2 任务必须等 `shared_l2_full_count == l1_clusters*shared_experts*2`**，即 L1 全部完成后才开始 →
L1/L2 之间有硬屏障，任一 L1 慢就整体后移。

### 排除清单（EP8/EP16 干净窗口，每项都跑过对照）

| 嫌疑 | 实验 | 结果 |
|---|---|---|
| pull 的 SF/weight 散读 | `PULL_META=1/2` bulk 化 | 无变化 |
| 同上 | `PULL_META=3` 完全不读 | 无变化（23.8/2.25×） |
| ring 脏写 | `PULL_META=4` | 无变化 |
| fabric 流量 | `PULL_META=5` 全本地化 | 无变化 |
| pull 整体 | `NO_PULL=1`（+NO_ROUTED） | 仍 2.41×（20.2/15.2–36.7） |
| routed GEMM | `NO_ROUTED=1` | 仍 ~2× |
| role1 barrier 自旋+驻留 | `NO_GRIDSYNC=1` | 26.0/2.61×，无改善 → 退避也无意义 |
| role1 的 16×6KB peer 读 | `SCAN_LOCAL=1` | EP8 29.1/2.79×、EP16 26.0/4.49×，无改善 |
| 时钟 / SM 数 / L2 | `clk_stats` + device props | 全部 2062/3996 MHz、152 SM、129.2 MB，逐 rank 相同 |
| HBM 带宽 | 2 GB copy + 60 s 循环 | 13.4–13.5 TB/s copy、12.5–12.9 read，p95 一致，**无外部干扰** |
| symm vs torch 本地带宽 | 128 MB sum/fill、SF 列主序散读 | 比值 0.95–1.16，无差异 |
| 共享权重物理位置 | `--se-offset-sweep 4`（0/64/128/192 MB padding） | 快 rank 不敏感（14–17）；慢 rank 同一分配 20→41 µs 漂移，**无单调趋势** |

### 正向结论（可复现）

1. **纯 shared（R=1×N 进程，plain torch 分配）**：13.6–15.4 µs，1.08–1.26×，逐卡一致。
2. **R=16 探针（pull/routed/barrier 全关）**：20.2–26.0 µs，2.4–2.9×；R=8 探针 17–29 µs，1.6–1.8×；
   R=4 探针 **.96 = 15.7/15.7/15.9/15.8（1.01×，完全均匀）**，**.72 = 14.9/20.8/17.0/15.9（1.40×）**。
   → 差异随 rank 数放大，且**按节点/GPU**区分（.72 的 g0–g2 系统性偏慢，g3 与 .96 全卡正常）。
3. 拆相（`SHARED_DBG2` 工时累加器）：慢 rank 是**每任务工时**更大，不是排队——
   rank0：L1 12.1 µs/任务、L2 6.7；rank1：L1 19.1、L2 16.2（1.6–2.4×）；CTA 启动偏斜只有 ~3 µs 且各 rank 相近。
4. 逐 cluster 的 start/dur 呈周期 4/8 结构（同一 rank 内），相位随 rank 变；慢的总是"早启动"的 L2 任务
   （与后一批 L1 重叠）→ L1/L2 硬屏障 + 重叠是放大器，不是根源。
5. 慢 rank 在同一权重分配上会在一次 run 内从 20 µs 漂到 41 µs（`se_sweep` 的 ref 列）→ **间歇性**。

### 下一步（未做完）

- 用 `GC_MOE_DBG_COMPUTE_MINUS` 或改 `compute_sms`，确认 role2 的 green-context 实际拿到多少 SM
  （136 CTA 需要 136 SM；若某些卡只给到 128 会退化成两波 → 正好是 2× 的形态）。
- 在慢卡上用 `--profile-calls` 抓一次 per-rank timeline，看 shared tile 是"等 TMA"还是"等 MMA"。
- 若确认是 green-context/CTA 放置问题：把 role2 的 CTA 数压到实际可用 SM 数以下，或固定 SM 亲和。

### 追加（同日）：几何与放大器确认

- `publish_shared_phase`：`total = m_blocks*n_clusters`；`total <= compute_sms//2` 时静态分配
  （cluster c 领 task c），否则全局认领。T=128,bm96 → L1 36 个、L2 40 个，**各一波**（68 cluster 够用）。
- L2 的门是**按 M-block**的：L1 epilogue 每个 cluster 的 2 个 CTA 各 `+1` 到
  `shared_l2_full_count + pool_block`，阈值 `l1_clusters*shared_experts*2`（=36，正好一块的量）。
  所以 shared 总时间 ≈ **max(36 个 L1 tile) + max(40 个 L2 tile)**：任一 tile 慢就整相位后移，
  慢卡上会**每个相位各乘一次**。
- `GC_MOE_DBG_COMPUTE_MINUS=2/8`（compute_sms 136→134/128）：EP8 spread 2.19×/2.06×，
  与 136（2.01×）无差别 → **不是 role2 CTA 放不下/green-context SM 不足**。
- 跨 6 次 EP8 run，**rank1（.72 GPU1）始终最慢（26.9–34.3 µs，其余 15–20）**；
  `.96` 在 R=4 下是 1.01× 完全均匀。时钟/SM/L2/带宽/分配器/权重位置全部相同 → 结论：
  **残差是机器级（特定 GPU）的执行时间异常，不是软件并发**；软件侧唯一的放大器是上面那个两相位结构。
- 实用含义：shared 的"均衡"不要指望跨 rank 一致；要压它的时间只能减少相位串行（例如把 L1/L2 融合，
  或让慢卡少分 tile），而 pull/routed 侧的优化与它无关。

### 根因（本轮收敛）：**"每相位取最慢 tile" × 两相位串行**，不是任何流量源

用 `shared_dbg2` 的逐 cluster 相位戳（e8s1，EP8 T=96 探针）算出来的决定性数据：

| rank | L1_max | L2_max | L1_min | L2_min | span |
|---|---|---|---|---|---|
| 3–7（快） | 10.7–11.2 | 10.2–10.6 | 8.2–8.8 | 7.4–7.8 | 17.5–17.9 |
| 2 | 12.1 | 13.2 | 8.6 | 7.8 | 20.3 |
| 0 | 14.6 | 17.3 | 8.8 | 7.5 | 24.3 |
| 1 | 16.2 | 22.7 | 8.8 | 8.0 | 30.8 |

- **典型 tile 在每个 rank 上都是 8–10 µs（min 完全一致）** → 工作量本身逐 rank 相同；
- 差异全部来自**相位内的慢 tile**：慢 rank 上 36 个 L1 tile 里有 17 个 >1.3×min、
  36 个 L2 tile 里有 18–26 个；快 rank 上只有 0–1 / 4–12 个；
- shared 的结构是 **L1 波（36 tile 一波）→ 每 M-block 的门（`shared_l2_full_count`）→ L2 波（40 tile）**，
  相位时长 = 该波最慢 tile → **慢 tile 的惩罚被完整暴露两次**（span ≈ L1_max + L2_max − 重叠）。

因此"根因"分两层：
1. **直接原因（可改）**：两相位串行 + 每相位取 max，使任何 tile 抖动按 ~2× 放大；
2. **抖动来源（不可控/环境）**：某些 rank/GPU 上相位内一大片 tile 变慢 1.3–2.8×，且随 run 漂移；
   已排除时钟/SM/L2/HBM 带宽（13.4 TB/s，p95 一致）/分配器/权重物理位置/共租（空闲采样 0%/0MiB）/
   green-context SM 数（`DBG_COMPUTE_MINUS=2/8` 无变化）/CTA 启动偏斜（~3 µs，各 rank 相近）。

**下一步实验（还没做）**：把 L2 门改成"等所有 L1 tile"（全相位分离），看 spread 是否下降 →
若下降，说明 L1(block1) 与 L2(block0) 的重叠是那个"一片变慢"的来源，可以按这个方向改调度；
若不下降，就只剩 tile 级的 L2/NoC 仲裁，软件侧唯一手段是减少相位串行（融合 L1/L2）或细分 tile 摊平 max。

### 根因（最终）：**K-loop 是延迟受限（35% 等 TMA 数据）+ 两相位取 max**，逐 GPU 的 TMA/L2 供给延迟差异被 1:1 放大

新增测量：在 MMA leader 的每个 K-block `mbarrier_wait(data_full)` 前后打点，累加到
`SHARED_DBG2` 的 `[20]`(总等待 ns) / `[22]`(等待次数)。EP8 T=96 探针（`.72`+`.96`，8 次 eager call）：

| rank | L1 总工时 | L2 总工时 | 等数据 | 等待占比 |
|---|---|---|---|---|
| 0–2（.72 g0–g2） | 597–612 µs | 371–399 | 354–374 | 36.4–37.0% |
| 3–7（.72 g3 + .96 全部） | 502–518 | 305–338 | 277–295 | 33.8–34.6% |

关键推论（把"带宽"彻底排除）：
- **L1 读 23.6 MB（W1），L2 只读 11.8 MB（W2），但两者耗时几乎相同**（快 rank：L1_max 10.7–11.2 vs
  L2_max 10.2–10.6）→ tile 时间**与字节数无关**，是**每个 K-block 的固定流水成本 + TMA 延迟**，
  不是 HBM 带宽（13.4 TB/s 的 1/3 都没用满，与前面所有带宽类实验的否定结果自洽）；
- 约 **35% 的时间在等 `data_full`**，说明流水深度（`stages` 已按 SMEM 上限取最大，见
  `heuristics/gc_moe.py: stages = budget_stages`）不足以完全掩盖 TMA 延迟 → tile 时间 ≈
  40 个 K-block × (每块固定成本 + 未被掩盖的 TMA 延迟)；
- 因此**逐 GPU 的 TMA/L2 供给延迟差异（1.3–2.8×）会 1:1 变成 tile 时间差异**，再被
  "L1 波 → 门 → L2 波，每波取最慢 tile"暴露两次 → rank 级 1.3–2.9× 的 spread。
- 逐 GPU 的慢是**稳定**的（`.72` g0–g2 在 8+ 次 run 里始终最慢；`.87` g1/g3、`.92` g0 各出现过），
  但**无法归因到任何可测硬件量**：时钟 2062/3996 MHz 全程无 throttle（`clocks_throttle_reasons=0x0`，
  功率 215–229 W，温度 31–32 °C）、152 SM、129.2 MB L2、HBM copy 13.4 TB/s（60 s 循环 p95 一致）、
  symm≈torch、权重物理位置无单调趋势、`COMPUTE_MINUS` 无变化、CTA 启动偏斜 ~3 µs、空闲 0%/0MiB。

**结论一句话**：shared 耗时不均是"**延迟受限的两相位结构 × 逐 GPU 的 TMA 供给延迟差异**"，
不是带宽争用、也不是 pull/routed/dispatch/barrier 的任何流量；软件侧唯一杠杆是**减少相位串行**
（融合 L1/L2 或细分 tile 让 max 摊平），或**加深 K-loop 流水**（受 SMEM 上限约束，当前已取最大）。

### 补充：内存类型不是根因；根因是 .72 的 GPU0–2 在"大块对称分配"下的执行异常（2026-11-22）

- **`GC_MOE_PLAIN_ALLOC=1`**（R=8 强制普通 `torch` 分配 + `peer_addresses` 自映射；探针下无 peer 访问，安全）：
  实测**更差**——总工时 spread 2.22×/2.29×（symm 是 1.38×/1.86×），且整体变慢。
  → **对称内存不是问题，反而比普通分配好**；之前"symm 映射是慢路径"的假设被否定。
- **`GC_MOE_ALLOC_PAD_MB=<N>`**（在 `get_symm_buffer_for_gc_moe` 之前垫 N MB，移动对称分配的物理落点）：

| pad | rank0 | rank1 | rank2 | rank3 | rank4–7 |
|---|---|---|---|---|---|
| 0 | 920 | 1125 | 872 | 820 | 813–826 |
| 64 | 1183 | 1505 | 1040 | 815 | 811–831 |
| 192 | 1164 | 1592 | 991 | 831 | 786–826 |
| 384 | 958 | 1230 | 829 | 815 | 772–823 |

  → **rank3–7（.72 g3 + .96 全部）恒在 772–831（±4%），对落点完全不敏感**；
  **rank0–2（.72 g0–g2）始终更高且随落点 920→1592→1230 大幅波动**。
- ECC / Retired Pages / Remapped Rows：两节点全部干净且一致；时钟、SM 数、L2、HBM copy 带宽也一致。
  → 这个退化**只有这个 workload 能看见**（延迟受限的 TMA 通路），常规健康检查全部漏检。

**最终根因（本集群口径）**：shared 耗时不均 = **两相位取 max 的结构放大器** ×
**.72 GPU0–2 在大块对称分配下的执行退化（1.2–1.9× 总工时，且对物理落点敏感）**；
健康的 GPU（.72 g3 + .96 全部）逐 rank 稳定在 ±4%。R=1 纯 shared 之所以均匀，是因为那次是
**小分配 + plain torch + 单 rank**，退化不出现。

**实用建议**：EP 级实验前用 `SHARED_DBG2` 的 `[0]+[2]`（shared 总工时）做**逐卡健康门禁**——
比 clocks / bandwidth / ECC 敏感得多；发现某卡总工时比其他卡高 >10% 就换卡。

### 重要更正（2026-11-22）：**插桩会抬高 shared 时长 ~23%，此前"多 rank 更慢"的结论作废**

`SHARED_DBG2 [20]/[22]`（K-block 等数据）的实现是**每个 K-block 一次 `read_globaltimer()` + 2 次 `atom_add_u64`**
（一个 call ~3000 次），实测把 shared span 从 15.5 µs 抬到 19.4 µs（**+23%**）。
所有"多 rank / 跨节点更慢"的数据都是在**带插桩的构建**上测的，而"R=1 很快"的数据是**无插桩**的旧构建
→ 那 30% 的差距是插桩差异，不是 rank 数差异。**该插桩已改为默认关闭（`GC_MOE_SHARED_DWDBG=1` 才开）**。
以后任何 A/B 必须保证**插桩开关一致**，否则结论无效。

**干净构建（无该插桩）的最终数据**：

| 配置（探针：NO_ROUTED+NO_PULL+NO_GRIDSYNC+SCAN_LOCAL） | 逐 rank（µs） | 中位 | 比值 |
|---|---|---|---|
| R=4，单节点 `.96`（两次） | 15.6 15.4 15.5 15.3 / 15.3 15.0 14.8 14.9 | 15.5 / 14.9 | **1.02× / 1.03×** |
| R=4，单节点 `.72`（两次） | **21.1 22.7** 15.6 14.9 / **20.6** 15.3 **17.0** 15.0 | 18.3 / 16.1 | **1.52× / 1.37×** |
| R=8，跨节点 `.72`+`.96` | **19.8 22.6 18.0** 15.8 \| 16.1 15.7 **18.0** 16.7 | 17.3 | 1.44× |
| R=1（4 个独立单 rank 进程/节点，无插桩） | 13.6–15.4（16 个 GPU） | 15.4 | 1.13× |

**结论（干净口径）**：
1. **只要 GPU 是健康的，shared 就是均匀的**（`.96` 四次测量：1.01–1.03×，14.8–15.6 µs，与 R=1 的 15.4 一致）；
2. 不均**全部来自特定 GPU**：`.72` 的 g0/g1/g2 是 17.0–22.7 µs，而 g3 与 `.96` 全部是 14.8–16.1；
3. **没有**"多 rank / 跨节点 / 分配器 / NCCL"效应——那些是插桩偏差 + run 噪声；
4. shared 的两相位（L1 波 → 每 M-block 门 → L2 波，每波取最慢 tile）把单卡慢 15–45% 放大成 rank 级 1.4–1.5×。

## EP16 shared gate 不稳定：根因 = CTA mod-8 共调度 stagger，修复 = GC_MOE_SHARED_DYNAMIC（2026-11-22）

问题：`.96`/`.82` 在 EP8 下 shared gate 时间稳定（19–21 µs，≤1.1×），EP16 下不稳定
（24–41 µs，最高 1.72×–2.6×）。以下全部实验在探针配置（`NO_ROUTED+NO_PULL+NO_GRIDSYNC+SCAN_LOCAL`）
下做，排除 pull/routed/barrier 干扰。

### 根因链（已逐级证实）

1. **现象**：EP16 下 shared L1 任务**启动时刻**按 cluster 索引呈**周期 8 的 4+4 分组**
   （`LLLLeeeeLLLLeeee…`，相位逐 rank 偏移），晚组比早组晚 8–16 µs 启动；而**完成时刻几乎相同**
   → gate 被最晚启动的 cluster 钉死。EP8 无此现象（全部 ~4 µs 启动）。
2. **触发条件**（四组对照）：`max(num_ranks, dispatch_sms) ≥ 16` 即触发；两者任一到达 16 都会：
   EP8+ds16（compute=136）不稳、EP16+ds8（compute=144，SM 划分与 EP8 完全相同）也不稳、
   只有 EP8 默认（ranks=8, ds=8）稳。**不是** compute_sms/workspace 尺寸本身。
3. **排除项**：散写 source_idx / count 交换（SELF_ROUTE=1 后 pattern 仍在）、workspace 撑大
   （e8pad86）、L2 cache 策略（evict_last）、NCCL 后端（gloo）、`GC_MOE_BALANCED_WORKQUEUES=0`。
4. **定位**（新增 per-cluster warp0 loop-entry 插桩，`shared_dbg2` flat u64 160+c，inverted-max 存储）：
   **stagger 完全在 consumer 侧 CTA 就绪**（prologue 完成、进入消费循环的时刻）：
   r6：loopE 3.6 vs 12.3 µs ≈ 100% 的 start stagger；r0/r5/r13：loopE 占 80–100%。
   任务本身早就发布好了（早组 pickup 延迟仅 ~0.5 µs）→ **调度器/publish 不是瓶颈**。
5. **硬件根源**：L20A `smCount=152, smCoscheduledAlignment=8` —— CTA 按 8-SM 对齐组共调度，
   交替组就绪时刻不同 → cluster（2 CTA/组×4）按 mod-8 分早晚。role1 足迹变大
   （rank 或 dispatch SM 到 16）把该 skew 从 ~1–2 µs 放大到 8–16 µs，进入 gate 可见区间。

### 修复：`GC_MOE_SHARED_DYNAMIC=1`（已实现 + 已验证）

把 shared 阶段的调度从"静态一 cluster 一任务"切到**原子全局抢占**（复用既有 dynamic claim 路径，
3 处 `publish_shared_phase` 调用点传入 `sh_sched_sms=0` 强制走 claim 循环）：
早就绪的 cluster 先抢任务，晚调度的 CTA 抢不到直接跳过，shared 不再被晚组钉死。

**EP16 探针实测（4 节点 16 rank，`routing_id16_t96`，bm96）**：

| 配置 | gate 范围 | 比值 | start pattern |
|---|---|---|---|
| baseline（static） | 24.1–41.5 µs | **1.72×** | `LLLLeeeeLLLL…` |
| `SHARED_DYNAMIC=1` | 19.6–25.9 µs（15/16 rank 在 19.6–21.5） | **1.32×** | **全 `eeee…`（mod-8 消失）** |

**bit-exact**：EP16 full mode（self_route+真实路由）`--check --dump-y`，static vs dynamic
16 个 rank 全部 **bitwise-identical**（max|diff|=0）；e2e 2573.2 → 2564.9 µs（略优）。

注意：修复针对的是"调度 stagger"这一层；单卡偶发慢（如 r14 一次 25.9 µs）仍可能由
不健康 GPU 造成，两类问题要用 per-cluster 数据区分（mod-8 pattern vs 孤立离群）。

### C1 实验（GC_MOE_DISPATCH_AFTER_READY=1，2026-11-22）：机制证实，但不是正确的修复

实现：软门（不动 graph 拓扑）——role1 照常启动（保留 per-call clears/epoch 次序），但在路由流量开始前
自旋等 role2 全部 CTA 过 prologue。握手字放在 x-order row 尾部（plain 路径无人使用，buffer 创建时
`zero_()` 保证初值为 0）：每个 role2 CTA 的 tid0 对 u64 ready 计数 atom_add；凑满 compute_sms 个的
CTA 用 `st_release_u32` 盖 release flag；role1 的 tid0 自旋 `ld_acquire` 直到 flag 离开入口快照，
然后 `sync_threads` 放行整个 CTA。

**EP16 探针四方对比（同机同配置，routing_id16_t96，bm96）**：

| 配置 | gate 范围 | 比值 | span 中位 | 任务落在哪些 cluster |
|---|---|---|---|---|
| baseline（static） | 24.1–41.5 µs | 1.72× | 18.6 | 固定 0–19 |
| C1 gate | 30.9–43.8 µs | 1.42× | **29.8** | 固定 0–19（start 干净 2–5 µs） |
| C1+DYNAMIC | 24.0–33.2 µs | 1.38× | 23.1 | 混合（0,2,3,9,10,11,18,19…） |
| **DYNAMIC（纯）** | **19.6–25.9 µs** | **1.32×** | **15.4** | **mod-8-early 位置（4,5,6,12,15,29,30,31…）** |

结论：
1. **机制证实**：C1 门让 role2 的 readiness stagger 完全消失（start 2–5 µs，mod-8 pattern 没了）——
   直接因果证明 role1 的 burst 拉伸 role2 的 CTA 就绪窗口（根因确认）。
2. **但作为修复失败**：role1 的 burst 释放点正好落在 shared GEMM 的权重流窗口上，全部 20 个 cluster
   的 span 均匀拉到 ~29–31 µs（与 SM 位置无关）——竞争没有消失，只是从"readiness 窗口"搬到
   "GEMM 窗口"，gate 绝对值反而更差（30.9–43.8 vs 24.1–41.5）。
3. **附带发现**：纯 DYNAMIC 的优势有两个来源——(a) 早就绪 cluster 先抢任务；(b) 抢到任务的 cluster
   恰好是 mod-8-early 的**快路径 SM 组**，其 GEMM 权重流对 role1 并发流量有韧性（span 15 µs 不变）。
   C1+DYNAMIC 把 readiness 抹平后，抢任务变成 atom_add 落地顺序的随机竞争，claimers 混合了
   快/慢路径组，span 居中（23.1）。
4. **最终排序：纯 SHARED_DYNAMIC >> C1+DYNAMIC > C1 > baseline**。e2e 四者持平（2496–2552 µs）。

残留问题（若未来要更进一步）：慢路径 SM 组（mod-8 late）的 GEMM 在 role1 流量下 span 拉伸的
微结构（L2 slice / 内存通道映射）未直接测到；C2（role1 重流量避开 shared 窗口、错到 routed 窗口）
是理论上的下一档，但 routed 相位的敏感性未知，且 e2e 已持平，收益存疑。

### DYNAMIC 干净环境复测（2026-11-22 晚：完美均衡 routing + 筛选后的 16 卡）

动机：此前 DYNAMIC 的 1.32× 里 r14 是单卡慢（span 20.2 vs 群体 15.x）——用逐卡门禁筛出的
干净 16 卡 + identity routing（imbalance **1.00×**，每 rank 精确 2304 对）重测，排除硬件混淆。

环境：9 台候选 R=4 门禁（三口径，阈值 ≤1.15×/end≤22µs）后选 **.96+.82+.92+.87**
（本轮 1.02/1.09/1.03/1.05×；.79/.87 从早前 FAIL 恢复——GPU 健康会漂移，门禁必须每次跑）。
同硬件同 routing 干净 A/B（probe，routing_id16_t96）：

| | STATIC | DYNAMIC |
|---|---|---|
| gate | 24.1–48.3 µs，**2.00×**（去最大 1.86×） | 19.4–24.5 µs，**1.26×**（去最大 **1.12×**） |
| span 中位 | 14.8–27.4（混合） | 15.0–21.0（15/16 在 15.0–16.0） |
| start pattern | `LLLLeeee`/`eLLLLeee`/`eeee`（mod-8 在） | **全 `eeee`** |
| start 范围 | 3.8–**32.7** µs | 2.8–**6.0** µs |

结论：
1. **同硬件同路由下，DYNAMIC 把 gate 从 2.00× 压到 1.26×（去单卡离群 1.12×）**，start 从
   32.7 µs 的 stagger 压到 6.0 µs 以内——干净环境下 DYNAMIC 的效果比之前（1.72×→1.32×）更显著；
2. 唯一残余仍是**单卡慢**：.92 g0 的 span 21.0 vs 群体 15.5（门禁时 1.03× 没抓到——单次 R=4
   筛查对边际卡的检出力有限，EP16 满载才显形）；排除后 **1.12× = EP8 健康基线同级（1.07–1.10×）**；
3. 完美均衡 routing 不改变结论（shared 与 routing 无关，m_blocks 只看 num_tokens）。

### #4 测量口径 + #1 任务细粒度化（2026-11-22 深夜，干净 16 卡 + DYNAMIC + identity routing）

**#4（bench 改动，已部署）**：`benchmark_gc_moe_eco.py` 的 eager 计时循环现在**逐 call 读 shared_dbg**
（6 次计时各读一次），三口径各取跨 call 的 **median/max/min**；新增输出行：
`shared gate (us, N-call median): min..max ratio | worst-call | best-call | per-rank[...]`，
JSON 增加 `eager_shared_end_worst/best_ranks` 与 `shared_calls` 序列。此前单 call 采样的噪声被消除。

**#1（任务细粒度化）**：读代码后发现 **N 维切细被 MMA 硬约束挡死**（`MixedMmaTile` 强制
BN128 × 2-CTA = 每任务固定 256 列，`n_cluster*2+cta_rank` 定位）。可行的是 **M 维**：
`--block-m` 从 96 降到 64/32 → `m_blocks` 1→2/3，任务数 20→40/60，且 L2(m=0) 与 L1(m=1) 流水。

EP16 干净 16 卡（.96+.82+.92+.87）+ DYNAMIC + identity routing（1.00×）实测：

| block_m | m_blocks | 任务数/相位 | gate 中位（跨 rank） | 比值 | worst-call | e2e |
|---|---|---|---|---|---|---|
| 96 | 1 | 20 | 19.7–24.0（中位 ~20.4） | 1.21× | 1.32× | 2492 µs |
| **64** | **2** | **40** | **17.8–22.3（中位 ~18.4）** | 1.25× | 1.37× | 2509 µs |
| 32 | 3 | 60 | 20.8–24.9（中位 ~23.0） | 1.19× | 1.19× | 2540 µs |

结论：
1. **bm64 是甜点**：两相位流水（40 个 cluster 各扛半尺寸 M-tile，L2(m0) 与 L1(m1) 重叠）把 gate
   绝对值降 ~2 µs（-10%）；bm32 过头（M=32 tile 的 MMA 效率损失吃掉粒度收益）。
2. **比值不随粒度改善**（1.19–1.25× 持平）——跨 rank 比值由**单卡漂移**决定（三次 run 的离群
   rank 各不相同：bm96 是 r8、bm64 是 r5/r10/r15、bm32 是 r5），这正是 #4 新指标暴露出来的：
   每卡 6-call 中位很稳，但卡与卡之间会漂。
3. N-split 方向（半宽 tile）硬件不可行；M-split（block-m 调参）已榨出可得的绝对值收益，
   对"跨 rank 不均"无进一步贡献——再次确认残余全部是单卡硬件层。

### +pull 层实验（2026-11-22 深夜）：activation pull 对 shared gate 零增量（结构性地不重叠）

设定：EP16（.96+.82+.92+.87，重扫空闲后沿用），`SHARED_DYNAMIC=1` + `ECO_TRACE=1`，
真实路由 `routing_r16.npy`（T=128，imbalance 1.61×），bm96（m_blocks=2，40 任务/相位），
NO_ROUTED=1 保持 routed 关闭。两个 arm 只差 pull 一个变量（均为真实 scan + gridsync，非 scan_local 探针）：

| arm | 配置 | gate 6-call 中位 | 比值 | worst-call | e2e |
|---|---|---|---|---|---|
| A | `NO_ROUTED=1 NO_PULL=1`（dispatch only） | 21.5–27.6（均值 ~24.9） | 1.28× | 1.53× | 2494 µs |
| B | `NO_ROUTED=1`（dispatch + pull） | **21.6–27.2（均值 ~24.3）** | **1.26×** | 1.38× | 2541 µs |

结论：
1. **pull 对 shared gate 零增量**（均值 24.9 → 24.3，噪声内）；e2e +47 µs 证明 pull 确实在跑。
2. **结构性原因**：role1 的相位序是 scan/count（~2µs 起）→ rank grid barrier（count 交换完成在
   ~100–200µs，trace f12）→ pull（barrier 之后）。而 shared 的窗口是 4–28µs——**pull 的流量
   落在 shared 完成之后 ~170µs**，物理上不可能干扰。pull 只拉长 role1 自身（+47µs e2e）。
3. 附带发现：**真实路由的完整 dispatch 流量**（peer 扫描 + source_idx 写 + count 交换，Arm A）
   比 scan_local 探针口径（identity，gate 均值 ~20.4）**贵 ~4µs**——dispatch 层在真实路由下更重。
4. 三个 run 的离群 rank 继续轮换（ArmA: r1/r5/r8/r10/r15；ArmB: r5/r13/r15）——间歇尖峰特征不变。

### +routed 层 + shared_first 开关（2026-11-22 深夜）：routed 零增量；shared_first 有害

设定：EP16（.96+.82+.92+.87），`SHARED_DYNAMIC=1` + `ECO_TRACE=1`，真实路由 `routing_r16.npy`
（T=128），bm96，完整流程（dispatch+pull+shared+routed）。两 arm 只差 `GC_MOE_SHARED_FIRST`：

| arm | 配置 | gate 6-call 中位 | 均值 | 比值 | e2e |
|---|---|---|---|---|---|
| C | 完整流程（shared_first 关，默认 routed∥shared） | 20.7–26.6 | **~22.9** | 1.29× | 2640 µs |
| D | 完整流程 + `SHARED_FIRST=1`（routed 等 shared） | 23.9–32.8 | **~27.2** | 1.38× | 2708 µs |

### 五层增量全表（shared gate 均值，同 16 卡、DYNAMIC 开）

| 层 | gate 均值 | 增量 | 结论 |
|---|---|---|---|
| 探针（scan_local + identity 路由） | ~20.4 | 基线 | dispatch 结构流量造成 stagger（已修复） |
| + 真实路由 dispatch（peer 扫描+source_idx+count） | ~24.9 | +4.5 µs | dispatch 层在真实路由下更重 |
| + activation pull | ~24.3 | ~0 | 时序错开（barrier 后才 pull） |
| + routed GEMM（完整流程，默认并行） | ~22.9 | ~0 | 48 个非 shared cluster 的 routed 权重流不伤 shared |
| + shared_first=1 | ~27.2 | **+4.3 µs（变差）** | **有害，勿开** |

### shared_first 有害的机制

`shared_first` 的实现是每个拿到非 shared 任务的 warp 先 `ld_acquire` 自旋等 `SHARED_DONE_WORD`
（单个全局字）。完整流程下 ~48 个非 shared cluster 的全部消费 warp（~400+ 线程）在整个 shared
期间**风暴式锤这一个 cacheline**，而 shared L2 的完成恰恰要对同一个字做 `red_release_gpu_add`——
读写对撞把 shared 的收尾拖慢（gate +4.3 µs），e2e 还要再付串行化代价（+68 µs）。两头都输。

### 最终结论（增量拼图收口）

- **shared gate 的全部干扰来自 dispatch 层**（scan/count 交换的早期窗口）；pull 和 routed 因
  时序/结构错开，零增量；
- 生产口径（完整流程 + DYNAMIC + 默认并行）：gate 20.7–26.6 µs（1.29×），残余是轮换的
  单卡间歇尖峰（环境性），与调度无关；
- `shared_first` 在本集群上有害（spin 风暴设计缺陷），默认关闭是对的。

### C2 实验（GC_MOE_DISPATCH_AFTER_SHARED=1，2026-11-22 深夜）：延后 dispatch 窗口不改善 gate

实现：把 role1 的整个路由 burst（scan+count 交换+source_idx 写）门到**全部 shared L2 完成之后**
（`SHARED_DONE_WORD >= shared_l2_tasks`，顺带启用该字的清零/递增机制）。与 C1（门到 readiness，
burst 砸 GEMM 窗口 span 2×）不同，C2 把整个 shared 窗口清空。完整流程 + DYNAMIC + 真实路由，
同 16 卡，跑两次：

| run | gate 6-call 中位 | 比值 | worst-call | e2e |
|---|---|---|---|---|
| Arm C（默认对照） | 20.7–26.6 | 1.29× | 29.4（1.36×） | 2640 µs |
| C2 #1 | 21.3–26.0 | 1.22× | **54.8（2.25×）** | 2732 µs |
| C2 #2 | 21.8–28.3 | 1.30× | 34.9（1.56×） | 2652 µs |

结果：
1. **中位 gate 无改善**（1.22×/1.30× vs 1.29×，噪声内）；
2. **机制上生效**：抽样 rank 的 per-cluster start 全部 5 µs（vs 默认的 3.6–12.3 散布）——窗口确实被清空了；
3. **为什么没用——这是本次最有价值的发现**：DYNAMIC 已经把 gate 与干扰**解耦**了。stagger 只影响
   晚就绪的 cluster，而它们在 DYNAMIC 下**本来就抢不到任务**；gate 由早就绪 cluster（~5 µs）的
   GEMM 工时（~16 µs）决定，那部分本来就不受干扰。清空窗口只是让晚 cluster "干净地闲置"，
   gate 不变。**绕开（DYNAMIC）与清空（C2）在 gate 上等价，但 C2 还要付串行化代价**；
4. 54.8 µs 尖峰未复现（第二次 worst 34.9，正常范围）——环境尖峰，非门自旋的系统性反馈；
5. full 模式下残余 stagger（门住 role1 后 start 仍散到 12.6 µs）来自 role2 内部流量
   （warp3 的 routed count-poll 等），但它对 gate 无影响（同第 3 条）。

### 三种方案最终排序（gate 视角）

| 方案 | 思路 | gate | 结论 |
|---|---|---|---|
| C1（门到 readiness） | 清空 readiness 窗口 | span 2×，gate 变差 | ✗ |
| C2（门到 shared done） | 清空整个 shared 窗口 | 持平 + e2e 代价 | 无增益 |
| **DYNAMIC（抢任务）** | **绕开干扰** | 最优且零代价 | ✓ 生产默认 |

### 16-rank full 模式可视化 profile（2026-11-22 深夜）：DYNAMIC 生产配置的完整时间线

文件：`prof/e16_full_dynamic_16rank_perfetto.json`（https://ui.perfetto.dev → Open trace file）。
生成：full 模式（无 probe/门/shared_first）+ `SHARED_DYNAMIC=1` + 真实路由 routing_r16 + bm96，
16 卡（.96+.82+.92+.87），`--profile-calls 1`（计时后 torch.profiler 下重放 1 次 captured call），
`merge_perfetto.py` 合并（本次扩展：sharedL1_first/last、sharedL2_last 字段名 + per-cluster
shared 条（tid 3）+ SHARED GATE 标记，来自 profile_shared_dbg_seq）。

结构：16 rank × 3 条 track（kernels / role1 phases (ring) / shared clusters）+ 1183 事件。
每个 rank 的 t=0 = 它自己的 role1 kernel 启动（跨节点时钟不同步，只看 rank 内相对时间）。

实测读数（单次 replay）：

| | 范围 |
|---|---|
| role1_dispatch（dispatch+pull 融合 kernel） | 37–173 µs（rank 间差 = barrier 等待） |
| role2_compute | 86–215 µs |
| role3_finalize | 7–43 µs |
| **SHARED GATE（相对各自 role1 start）** | **20.1–28.5 µs** |

关键画面：**SHARED GATE 全部落在 role1 尚在执行的区间内**（role1 最短 37µs > gate 最长
28.5µs）——shared 在 dispatch+pull 还没跑完时就已完成，DYNAMIC 下不被卡住的直观证据。
shared 条只出现在抢到任务的 cluster 上（DYNAMIC）；dispatch 相位标记 route_start→route_done→
grid_join→pull_entry→pull_exit 可在 role1 kernel 内部看到 pull 在 barrier 之后。

### 可视化 profile v2（重新生成，2026-11-22 深夜）：四阶段 span + 触发时刻标记

`prof/e16_full_dynamic_16rank_perfetto.json` 重新生成（同一次 run 的数据，纯后处理更新）。
结构：16 rank × 4 条 track：
- `kernels`：role1_dispatch / role2_compute / role3_finalize（kineto）
- `role1 phases (ring)`：dispatch 全相位（route_start/done、grid_join、pull_entry、first_bulk_issue、
  sf_done、x_arrive、s2g_done、**first_publish（第一个 block 拉取完成、l1_full 发布=触发 routed GEMM）**、
  last_publish、pull_exit、role1_exit）+ routed 的 l1_first/l1_last/l2_first/l2_last + **SHARED GATE**
- `shared clusters`：per-cluster shared 条（start→done，DYNAMIC 下只出现在抢到的 cluster）
- `phase spans`（新增）：**DISPATCH: route / DISPATCH: pull / SHARED EXPERT (full) / ROUTED GEMM (full)**
  四个汇总条

16 rank 实测（单次 replay，相对各自 role1 start）：

| span | 中位 | 范围 |
|---|---|---|
| DISPATCH: route | 10.3 µs | 4.7–20.7 |
| DISPATCH: pull | 18.3 µs | 11.8–44.6（barrier 之后才开始） |
| **SHARED EXPERT (full)** | **18.6 µs** | 15.2–22.8（起点 ~4-6 µs，GATE 20-28 µs） |
| ROUTED GEMM (full) | 76.8 µs | 70.5–103.1（从 l1_first ≈ first_publish+2µs 起） |

时间线（r0 例）：route 0.5–12.5 → count 交换/barrier 至 114.6 → pull 114.9–132.3 →
first_publish 120.6（触发）→ routed L1 first 122.7 → routed 至 ~195；**shared 4.7–23.5 完全
在前段完成**。first_publish → l1_first 间隔 1.0–5.9 µs（16 rank）。

**遗留诊断项**：profiled replay 的 ring 字段 f10/f2/f5（shared 时间线）与同次 replay 的
shared_dbg 矛盾（f10 反解 19.3µs vs dbg 首起点 4.7µs；f5=34.1 > gate=23.5 不可能）。同行的
dispatch/routed 字段链（counts→pull→publish→l1_first）自洽且 role1_exit(133.4) 与 kineto
role1 时长(136) 吻合——只有 shared 三字段异常，原因未明（待查 _eco_trace_ring 在 graph replay
下的行为）。已改为 shared 相位一律用 shared_dbg（全程验证过的数据源）。

### 可视化 profile v3（最终版，2026-11-22 深夜）：sse_tr1 风格六相位 + 修正后的计数口径

`prof/e16_full_dynamic_16rank_perfetto.json` 最终版。`phase spans` track 改为 sse_tr1 风格六相位，
**边界按最新（修正后）口径重算**：

| 相位 | 边界（新口径） | 旧脚本的错误 |
|---|---|---|
| dispatch: routing + count exchange | f16 route_start → **f12 counts_ready**（含 fabric 等待 peers counts） | 旧 `[f16,f17]` 漏掉了 count 交换（实测交换占 ~95µs，route_done 只是本地 scan） |
| dispatch: token pulls | f20 pull_entry → f27 pull_exit | 同 |
| shared expert L1+L2 | **shared_dbg** min start → SHARED GATE（已验证源） | 旧用 f10→f5+**硬编码 2.5µs 估计**，且 f5 是异常字段 |
| wait for routed tokens | SHARED GATE → f13 l1_first（role2 视角等 token 到位） | 旧的 shared_end 是估计值 |
| routed GEMM | f13 l1_first → f8 role2_exit | 旧拆 L1/L2 两段 |
| combine | f1 role3 entry → f9 role3 done | 同 |

16 rank 实测（中位 / 范围，µs）：routing+count exchange **118.3**（18.0–128.5，大头是 fabric 等
peers）；token pulls 18.3（11.8–44.6）；**shared expert 18.6（15.2–22.8）**；wait for routed tokens
104.7（2.1–113.8，与 count 交换/pull 重叠——role2 视角）；routed GEMM 73.9（68.2–100.7）；
combine 31.4（4.7–39.9）。自洽性：wait 最短 2.1µs 的 rank 正是 count 交换最快（18µs）的 rank。

### native vs self_route 六相位对比（2026-11-22 深夜，16 卡 full+DYNAMIC+真实路由，单次 replay）

文件：`prof/e16_full_dynamic_16rank_perfetto.json`（native）与
`prof/e16_full_dynamic_selfroute_16rank_perfetto.json`（self_route），同口径同 merge 脚本。

| 相位（中位 µs） | native | self_route | Δ |
|---|---|---|---|
| dispatch: routing + count exchange | **116.2** | **12.5** | **-103.7** |
| dispatch: token pulls | 18.1 | 20.1 | +2.1 |
| shared expert L1+L2 | 18.5 | 23.1 | +4.6 |
| wait for routed tokens | **102.0** | **4.2** | **-97.8** |
| routed GEMM | 73.6 | 80.4 | +6.7 |
| combine | 31.1 | 32.0 | +0.8 |
| **role1 kernel（kineto）** | **140.3** | **37.1** | -103.2 |
| **role2 kernel（kineto）** | **183.8** | **91.2** | **-92.6** |
| e2e（bench） | 2690 | 2662 | -28 |

结论：
1. **count 交换的 fabric 等待被彻底消除**（116→12.5µs）——self_route 用 bulk topk 拷贝替代散写
   + 免 count 交换，backlog 不再存在；
2. **routed 关键路径整体前移 ~98µs**：wait for routed tokens 102→4.2µs，routed GEMM 提前启动，
   role2 kernel 总时长 183.8→91.2µs（GPU 管线 -92.6µs；e2e 只降 28µs 因为 bench 的 host 开销
   掩盖了 GPU 侧收益）；
3. **shared 不受影响**（18.5 vs 23.1，单样本噪声；6-call 中位口径：native 21.2-25.8/1.22×，
   self_route 21.5-37.0/1.72×——self_route 的 shared gate 离群更多，与其共享 bulk 拷贝流量
   有关，但都在 DYNAMIC 修复后的量级）；
4. routed GEMM 本体 +6.7µs（80.4 vs 73.6）——self_route 的路由派生开销/不同的 pull 模式
   略增 GEMM 前准备，远小于换来的 -98µs。

### profile 文件跨 rank 对齐修复（2026-11-22 深夜）

现象：合并后的 perfetto 文件里各 rank track 起点不一（rank0 从 0 开始，其他节点 rank 从
+248/+543/+914µs 开始）。原因：**跨节点 kineto 时间戳来自不同步的 host 时钟**（四台机器
时钟基点差 248-914µs），而 profile 前 bench 的 dist.barrier() 使真实启动偏差只有 2.5-11.6µs
（节点内）。修复：`merge_perfetto.py` 归一化每个 rank 的事件到它自己的 role1 kernel 起点
（t=0 = 本 rank role1 start），两个文件（native/self_route）均已重新生成，16 条 track 全部
左对齐。注意：归一化后跨 rank 仍不可比绝对墙钟（时钟不同步），但 rank 内相对时间线完全可比，
且相位标记本来就是 rank 相对的。

### self_route + 均衡 routing + PULL_AFTER_SHARED 实验（2026-11-22 深夜）：两个假设都被否定

**假设 1：routing 不均衡是 shared 抖动的主因 → 否定**

| 配置 | routing | gate 比值 |
|---|---|---|
| self_route + DYNAMIC + 真实路由（1.61×） | r16 | 1.72× |
| **self_route + DYNAMIC + identity（1.00×）** | id16_t96 | **1.65×** |

identity routing 下 gate 仍 1.65×，6/16 rank 出现 LLLL pattern。**routing 不均衡不是主因**。

**假设 2：PULL_AFTER_SHARED 可以修复 self_route 的 shared 抖动 → 否定**

| 配置 | gate 比值 |
|---|---|
| self_route + DYNAMIC（无 PAS） | 1.65× |
| **self_route + DYNAMIC + PULL_AFTER_SHARED** | **1.60×**（span 散布更大 10.5-24.1） |

PULL_AFTER_SHARED 只延迟 activation pull，**不延迟 self_route 的 bulk topk 表读取**——那些
fabric 读发生在路由派生阶段（pull 之前），仍然撞 shared 窗口。

**根因修正**：self_route 下 shared 抖动的干扰源不是 activation pull（PAS 已延迟它），
而是 **bulk topk 表的 fabric 读**（self_route 必须从 15 个 peer 批量拷贝 topk_idx 来本地
派生路由，这些读紧接 role1 启动开始，正好落在 shared 的 readiness/权重流窗口）。
这与 routing 分布无关（identity 也一样），也与 activation pull 无关（PAS 无效）。

### self_route + shared 完整方案排序（最终）

| 方案 | gate | routed 路径 | 建议 |
|---|---|---|---|
| native + DYNAMIC | **1.22×** | 慢（count 交换 ~95µs） | shared 稳定优先 |
| self_route + DYNAMIC | 1.65-1.72× | **快（-98µs）** | routed 速度优先 |
| self_route + PAS | 1.60× | 快 | PAS 无效，勿开 |
| **待做**：self_route + 延迟 bulk topk 读到 shared 完成后 | ? | 稍慢 | 需改 kernel |

如果需要同时保住两头（shared 稳定 + routed 快），需要把 self_route 的 bulk topk 读也门到
shared 完成后——即把 SHARED_DONE_WORD 的等待从 PULL_AFTER_SHARED 扩展到路由派生阶段。
但这会让路由更晚完成，routed 路径回吐 ~15-20µs（仍然远优于 native 的 -98µs）。

### ROUTE_AFTER_SHARED 实验（2026-11-22 深夜）：mod-8 LLLL pattern 消除，gate 1.65→1.49×

实现：`GC_MOE_ROUTE_AFTER_SHARED=1`——把 self_route 的 bulk topk 表读取门到全部 shared L2
完成后（SHARED_DONE_WORD 自旋），同时启用该字的清零/递增机制。

identity routing（1.00×），同 16 卡，DYNAMIC 开：

| 配置 | gate 比值 | start pattern | span_med |
|---|---|---|---|
| native + DYNAMIC | 1.22× | `eeee` | 紧 |
| self_route + DYNAMIC（无门） | 1.65× | 6/16 `LLLL` | 13.6-19.1 |
| self_route + PULL_AFTER_SHARED | 1.60× | 仍有 `LLLL` | 10.5-24.1 |
| **self_route + ROUTE_AFTER_SHARED** | **1.49×** | **基本全 `eeee`（仅1 rank 1个L）** | 9.9-22.0 |

结论：
1. **LLLL pattern 被消除**（bulk topk 读被成功门住）——readiness stagger 源头被切断；
2. **gate 1.65→1.49×**（改善 0.16×）但仍未到 native 的 1.22×；
3. 残余 1.49× 来自 **span 散布**（9.9-22.0µs）而非 start stagger——可能原因：
   (a) 门自旋本身对 SHARED_DONE_WORD 的 cacheline 竞争（16 CTAs × 1 线程自旋
   ~15-20µs，shared L2 的 release-add 要写同一个字）；(b) 单卡方差；
4. e2e 2407µs（vs 无门 self_route 2353µs，+55µs = shared 完成才开始读 topk 的代价）。

### 完整方案排序（最终更新）

| 方案 | gate | routed 路径 | e2e | 建议 |
|---|---|---|---|---|
| native + DYNAMIC | **1.22×** | 慢（-0µs） | ~2500 | shared 稳定优先 |
| self_route + DYNAMIC | 1.65× | **快（-98µs）** | ~2350 | routed 速度优先 |
| self_route + ROUTE_AFTER_SHARED | **1.49×** | 快（-40µs） | ~2407 | **折中：可接受** |
| self_route + PAS | 1.60× | 快 | ~2450 | 无效，勿开 |

## 完整 EP16/EP32 MoE 性能测试（2026-11-23，原生方法论 CUDA graph + self_route + DYNAMIC + identity routing）

### 配置
- benchmark_gc_moe_eco.py，CUDA graph 模式（`--iters 30 --inner-iters 100 --warmup 10`，
  与原生 _se bench 同 methodology：背靠背 graph replay、inner_iters 摊销、无 host 同步）
- `GC_MOE_SELF_ROUTE=1 GC_MOE_SHARED_DYNAMIC=1`
- identity routing（T=128/rank，imbalance 1.00×）
- `--mode plain384:plain:384 --num-shared-experts 1 --block-m 96`
- EP16：.96+.82+.92+.87（4 节点）；EP32：+.79+.88+.76+.72（8 节点）
- 两遍：第一遍纯 timing，第二遍 `--profile-calls 1` + torch.profiler

### 整体延迟（CUDA graph，慢 rank 中位，µs/call）

| 规模 | e2e |
|---|---|
| EP16 | **93.31** |
| EP32 | **66.70** |

EP32 更快：每 rank 本地专家更少（12 vs 24），routed GEMM 工作量减半。

### 六相位详细数据（单次 profiled replay，µs）

| 相位 | EP16 med | EP16 min | EP16 max | EP32 med | EP32 min | EP32 max |
|---|---|---|---|---|---|---|
| dispatch: routing + count exchange | 15.7 | 8.2 | 27.7 | 12.7 | 9.8 | 25.4 |
| dispatch: token pulls | 17.4 | 14.5 | 28.7 | 14.7 | 13.1 | 25.8 |
| **shared expert L1+L2** | **18.7** | **9.1** | **30.7** | **16.3** | **15.1** | **19.4** |
| wait for routed tokens | 2.5 | 0.0 | 14.4 | 3.6 | 0.5 | 10.9 |
| routed GEMM | 75.1 | 71.1 | 81.9 | 45.8 | 41.4 | 60.3 |
| combine | 4.4 | 3.9 | 4.8 | 4.9 | 4.3 | 5.5 |
| role1 kernel | 36.6 | 27.0 | 47.0 | 33.5 | 28.5 | 45.5 |
| role2 kernel | 91.7 | 86.9 | 107.0 | 60.4 | 56.0 | 67.3 |
| role3 kernel | 6.9 | 6.0 | 7.8 | 7.2 | 6.1 | 8.5 |

### shared expert 跨 rank 稳定性

| 规模 | med | min | max | max/min | 判定 |
|---|---|---|---|---|---|
| EP16 | 18.7 | 9.1 | 30.7 | **3.37×** | **不稳定**（self_route bulk 读干扰，与之前发现一致） |
| **EP32** | **16.3** | **15.1** | **19.4** | **1.28×** | **稳定** ✓ |

EP32 下 shared expert 明显更稳定（1.28× vs 3.37×）——可能原因：
(a) 每 rank 本地专家更少（12 vs 24），bulk topk 读量更小；
(b) 32 rank 的 self_route 路由派生窗口更短（12.7 vs 15.7µs），与 shared 的重叠更少；
(c) DYNAMIC 抢占在更多 cluster 下更均匀。

### Profile 文件

- `prof/e16_cg_sr_dynamic_16rank_perfetto.json`（EP16，16 rank，968 事件）
- `prof/e32_cg_sr_dynamic_32rank_perfetto.json`（EP32，32 rank，1946 事件）
- 每个 rank 4 条 track：kernels / role1 phases (ring) / shared clusters / phase spans
- 所有 rank 归一到各自 role1 start（t=0）

### 与原生 _se benchmark 的对齐说明

使用 `benchmark_gc_moe_eco.py` 的 CUDA graph 模式（非 eager），methodology 与
`benchmark_gc_moe_se.py` 一致：背靠背 graph replay、inner_iters 摊销、无 host 同步。
`benchmark_gc_moe_se.py` 原版在节点上因 deep_gemm 初始化挂起无法直接运行。

### 三配置对比（2026-11-23，EP16/EP32，identity routing T=128，CUDA graph）

| 规模 | 配置 | 一次 MoE call（µs） |
|---|---|---|
| EP16 | native | **93.98** |
| EP16 | self_route only | **92.21** |
| EP16 | self_route + DYNAMIC | **93.31** |
| EP32 | native | **66.97** |
| EP32 | self_route only | **65.74** |
| EP32 | self_route + DYNAMIC | **66.70** |

identity routing 下三配置基本持平（差 ≤2 µs）——native 的 count 交换弱点在 identity
routing 下不暴露（零跨 rank token，count 行全部为零，交换即刻完成）。self_route 的
优势（消除散写风暴）在此场景无差别。真实路由下的差异见之前实验。
