# x-order dispatch：首块专家优先发送方案

状态：阶段 A 已实现待验证（2026-11）。EP8 验证环境；EP32/EP64 集群待 sat_app 权限。
实验日志（fused dispatch、publish 优化、探针）见 AGENTS.md「pull dispatch 优化」章节。

## 方案概要

专家权重重排时按路由热度为每个 dst 生成单副本专家的热度排序表（greedy
max-coverage 选前 K 个进早集，其余按 mean 补齐）；src 侧在 fused pass +
grid_sync 后按**自身相位旋转的 per-dst 序**发布 count——第 j 轮发每个 dst 的
第 j 热专家，src i 从 dst (i+1)%R 起轮转，把"所有 src 同时打同一 dst"的网络
突发摊薄到 R 个 dst；dst 侧按同一 per-dst 序**渐进等待 count、逐个排产
pool_block**。发送序 ≡ 池序 ≡ 消费序统一为一条有序流：最热专家 count
~5-7µs 到达 → 首块 GEMM 提前 ~6-9µs（l1_first ~18-20µs → ~10-14µs）。
关键操作只有三个：**重排时生成表、src 按旋转序发布、dst 按序渐进消费**。

## 1. 动机

EP8 bm32 实测分解（trace probes slots 37-40）：fused pass ~1.1µs、grid_sync ~0.4µs、
count 发布 ~3-6.5µs、传播+轮询 ~6µs → count_ready ~16µs → l1_first ~18-20µs →
call ~194µs（pull 阶段 ~39µs 为带宽 bound 大头）。

**问题**：dst 首块 GEMM 需要某 expert 的全部 src 行（count+index+x），而所有 expert
的 count ~同时到达 → 首块被迫等 bulk。目标：让最热 expert 的元数据最先到，
首块提前 ~6-9µs。

## 2. 核心思想：统一 x-order

**发送序 ≡ 池序 ≡ 消费序**：每 dst 一张热度序表，src 发布序是它的相位旋转视图，
dst 消费序就是它本身——三者由同一组 per-dst 表 + 旋转规则隐式统一：

```
src i: grid_sync → 按 xorder_i（per-dst 表的旋转交错）发 count（前缀=热专家先发）
 dst d: count 按 per_dst[d] 序到达 → pool_block 前缀按 per_dst[d] 算
      → 最热专家占 pool_block 0..k → 任务/pull/GEMM 按此序推进
```

早到的 count 直接对应早的 pool_block——单一有序流，无 straggler、无热区/冷区错位。

### 2.1 x-order 的确定（数据已验证）

**生命周期**：每次专家权重重排（EPLB/replan）时由路由统计生成一次，复用至下次。
bench 从 routing_off.npy 聚合注入；生产接重排钩子（后续工作）。

**结构**：
- 每 dst 内：候选 experts 按热度降序 → 得到 per-dst 本地序 `per_dst[d]`（长度 L_d）
- 全局 publish 序：**per-src 相位旋转**（防网络突发）
  - src i 的发布序列：第 j 轮依次取 dst `(i+1+k)%R`（k=0..R-1）的第 j 号 expert：
    `xorder_i = [d_{i+1}#1, d_{i+2}#1, ..., d_i#1, d_{i+1}#2, ...]`
  - **轮结构必须保留**：若按全局热度序纯排序，冷 dst 首个 expert 会排到 ~300+ 位 →
    该 dst 首块不早 → makespan 受损；round-robin 保证所有 dst 首块 ~同时早到
  - **旋转防突发**：若所有 src 用同一全局序，任意时刻 R 个 src 的远端写同时打向
    同一 dst → `recv_count_sum[le]` 同地址 atomic 串行化 + 入口突发；旋转后任意时刻
    R 个 src 写向 R 个不同 dst，每 dst 每轮 ~1 路写
  - 代价：dst d 第 j 轮最后到达位置 ≤ `j*R+(R-1)`（vs 对齐序 `j*R+d`），
    延迟 ≤ R 个 publish step（EP8 ~60ns）可忽略
  - **每 dst 的 expert 仍按 `per_dst[d]` 序到达**（轮结构不变，轮内偏移不同）→
    dst 渐进调度的到达序假设不受影响
  - 实现：过程化生成，位置 p → 轮 j=p/R、偏移 k=p%R →
    `dst=(src_rank+1+k)%R`，`expert=per_dst_local[dst][j]`；无需存 R 张表
  - 起始偏移 (i+1) 任意（从 dst i 开始等价）；ECO 变长 L_d 时按 per-dst 长度处理
- 不再需要单一全局 xorder 数组——统一性 = per-dst 表 + 旋转规则，
  所有 src 确定性推导同一组序列 = 隐式商定保持
- 副作用保留：远端 atom 目的地在 dst 间均摊，避免单 dst L2 突发

**热度必要性**：GEMM 块按 block_m=32 补齐——16-token 块与 32-token 块占 TC 时间
几乎相同但只产一半有效输出。热度无关序 → 首块 ~16 tokens（random 均值）→ 早窗口
被半空块填满；热度序 → 首块 ~90-125 tokens → 2-4 个满块，早期 TC 吞吐高 ~2-4×。
热度不是让块"更早到"（早集内 count ~同时到），而是让早到的块"值得先排"。

**候选集 = 单副本 slots（ECO 约束，已验证）**：
- fixed slots（`slot_idx < eco_num_fixed[R]`，per-rank，EcoMeta）：单副本，count 走
  原生 `recv_count_sum` 到达协议 → x-order 完全适用
- replica slots（`>= num_fixed`）：count 来自 `eco_quota`（plan role ~15-35µs），
  scheduler 对其到达数期望=0、不读 dispatch count → 早发布无意义
- plain 模式全 slot 单副本 → 无此限。ECO 候选池 = num_fixed/dst，K 相应下调

**热度稳定性**（routing_off 1200 calls，4 段验证）：相邻段 top-8/dst 重叠 ~90%
→ 单表足够；单 argmax 命中仅 ~5% → 必须 top-K 集合兜底，不能押单个。

**选集键 = greedy max-coverage**（直接优化目标 E_c[max_{e∈S} cnt]，子模最大化
(1-1/e) 近似保证；实测优于 mean 排序）：

```
S={}; cur[c]=0; repeat K: e*=argmax_e mean_c(max(cur,cnt[:,e])); S+=e*; cur=max(cur,cnt[:,e*])
```

routing_off 实测（K=8/dst）：mean 排序 E[best]=84.9/P≥32=75% → **greedy 89.9/78%**。

**K 值权衡**（greedy）：K=4→62.9/53%（不推荐）；**K=8→89.9/78%（默认）**；
K=12→105.9/90.6%；K=16→115.5/96.9%（近 oracle 125，发布成本×2）。
发布成本 ≈ K×R×2 个远端 op（早段 ~0.5-1µs/8 experts）。

### 2.2 已证伪的替代方案（勿重试）

| 方案 | 结果 |
|---|---|
| 各 src 按本地 top-K 独立选首块 expert | 8 src 交集≈0 → 无 expert 凑齐全 src 行 |
| 在线商定（读对端上轮 count/广播）| 成本否决 |
| 静态 top-1 argmax | 跨 call 命中 ~5% |
| 两阶段 v1（热/冷相位）| 被统一渐进规则取代——相位边界是往状态机插接力，更复杂 |
| per-src 按本地热度发 | 全局热 expert 只对部分 src 热 → 完成时间≈bulk |

结论：必须跨 src 一致的顺序，唯一零通信来源 = 重排时生成的静态表。

## 3. 协议不变量（正确性）

1. **count⇒index**：expert count 只在其全部 index 行写完后发布；fused pass ~1.1µs
   完成全部 index 写 → grid_sync 后任何发布顺序都满足 ✓
2. **到达计数**：`recv_count_sum[e]` 高 32 位 = `dispatch_sms×num_ranks` 次 atom 到达；
   每 (e,src) 发布 = matrix 行 store + sum atom，格式不变，dst 等待逻辑不变
3. **零计数也要发**：src 未路由到早集 expert 时 count=0 照常发布，否则 dst 到达
   计数凑不齐死等；dst 见 0 自动跳过 ✓
4. **双缓冲 parity / epoch 门**：沿用现有 call 序协议，不新增信号
5. **src 一致性**：所有 src 共享同一组 per-dst 表 + 同一旋转规则 →
   确定性推导各自发布序列，发布同一批早集 expert（隐式商定，零通信）

## 4. Src 侧改动（role1 dispatch，~40 行）

```
fused pass（不变）→ grid_sync（不变）
发布循环按本 src 的相位旋转序迭代（位置 p → 轮 j=p/R、偏移 k=p%R）：
  dst = (src_rank + 1 + k) % R;  le = per_dst_local[dst][j]
      st recv_count[src][le]（matrix 行）
      red.release.sys recv_count_sum[le] += (dispatch_sms<<32)|count[e]
  —— 前 K 轮（热集）先到 dst，bulk 照旧；发布总成本不变（~3µs，
     red.release.sys + CTA 分块已在树）
```

- **index 行写序不改，也不需要改**：按地址寻址，fused pass 已全部完成；
  dst 按地址读、对到达序零要求。写模式逐 route 远端 `st.u32`、由 routing
  决定目的地——天然打散（token 序 × 随机 routing → 目的 (dst,e) 跨 dst 跳变；
  各 src fused pass 不同步）。热 expert 行 ~100 次写同区 = 400B 命中少数
  L2 line、写在 pass 不同时刻流水化，slot 全局 atomic 同地址串行 ~300ns
  与其他地址并行——实测 pass ~1.1µs 已含此效应，非瓶颈。
  原版非 fused 路径同样逐 route 分散写（smem atomic+merge 两 pass），
  聚集写是协议固有、非 fused 引入；消除需桶排序 ~300 行，无下游收益，不做。
- `send_count[e]` 终值直接给出 count[e]，无需额外直方图
- dst 不感知发布顺序细节——只看每个 expert 的 sum 到达与否；协议零改动；
  ECO 下迭代集只含 fixed slots（§2.1）
- publish 的 8-CTA 分块按位置区间切分即可（同 src 内所有 CTA 用同一序）

## 5. Dst 侧改动（scheduler，核心手术 ~150-250 行）

### 5.1 现状约束

- count 等待 per-lane 独立（lane 轮询 `wave*32+lane` 的 sum word），**但 warp 在
  wave 末尾收敛** → wave 完成时间 = 最慢 expert → 早到的 count 用不了
- `locate_routed_pool_block` 按 expert-id 序做块前缀；任务循环是单状态机
  （L1/L2 交织+warmup），`routed_m_blocks` 需全量 count

### 5.2 主线：x-order 驱动的渐进调度（一条规则）

**count 按 x-order 到一个、排一个。** 替换"等全部 count"块：

```
routed_m_blocks = 0; xpos = 0
while xpos < local_experts:
    e = xorder_local[xpos]                # 本 dst 第 xpos 热 expert
    等 recv_count_sum[e] 到达 → count
    expert_counts[e 槽位] = count
    routed_m_blocks += ceil(count/block_m); xpos++
    # 新可调度 pool_block ∈ [旧total, routed_m_blocks) 立即进任务循环
    # 批量阈值：累计新块 ≥ l1_clusters 再统一 publish（摊薄 warp 同步）
任务循环终止：xpos==L 且 claims 耗尽（counts_done 标志替代固定 routed_m_blocks）
```

- **count 落位**：到达时 scatter 到 `cnt_by_xpos[]`（~L×4B）+ arrival 计数；
  pool 前缀按 x-order 扫此数组 → 前缀天然 = 池序
- **expert-id 保持 canonical**：x-order 只是遍历置换表，不动编号/ECO/push
- **冷 expert count=0**：照常占 x-order 位次，前缀正确跳过（0 块）
- **L1/L2 兼容**：`required_l1=(pool_block+1)*l1_clusters` 全局保序不变
- **warmup**：l1_warmup 原按总量规划；渐进下用上 call 总量或已知+期望估计——
  只影响 L1/L2 排序效率，required_l1 guard 保证正确，估错只是亚优
- 改动集中在 schedule_tasks 中段（wait 块 → 摄入循环；终止条件+warmup 调整）

### 5.3 pull 侧残余（暂缓）

per-(e,src) 渐进 pull 有**布局 blocker**：dense pool 按 expert-major 拼接，
src s 的 token 偏移 = `sum(count[e,0..s-1])` 需全 src count → 写 pool 仍须等 sum；
需 staging 缓冲（多一份拷贝）或 pair-window 布局（改动大）。性价比不高，
待渐进调度落地按实测缺口再定。index 行 ~1.5µs 即完成，pull 不缺 index。

## 6. 性能预期（诚实版）

| 指标 | 现状 | 方案后 | 物理下限 |
|---|---|---|---|
| 首块 count 就绪 | ~16µs | ~5-7µs | ~4-5µs |
| **l1_first** | **~18-20µs** | **~10-14µs** | ~8-11µs |
| call 总时延 | ~194µs | ~186-190µs（-3~4%）| — |
| pull_end/尾部 | ~55µs | ~持平略优 | 带宽 bound |

首块耗时达物理下限 ~85-90%；残余 ~2-3µs = 更细批量 + 渐进 pull staging。
count 下限 ~4-5µs（全 src 扫描+同步+发布+传播+skew）不可再降，除非动协议本身。
pull 吞吐不变（带宽 bound）——吞吐优化是正交工作线。

## 7. 分阶段实施

| 阶段 | 内容 | 验证 |
|---|---|---|
| A | x-order 表生成（greedy，routing_off.npy）+ src 按序发布 | bit-exact；trace 看 ready 梯度 |
| B | dst 渐进调度（§5.2）| bit-exact + l1_first A/B |
| C | （可选）渐进 pull staging / 更细批量 | 按 B 实测缺口定 |

- 回退：`GC_MOE_XORDER=0/1`（默认关）、`GC_MOE_XORDER_K` 调 K
- bench 表注入：kernel 参数/env；生产接重排钩子为后续

## 8. 风险

1. **x-order 质量**：greedy K=8 最坏 ~74-82 tokens（≥2 块）；冷 pick 损失有界，
   bulk 不受影响。K=8 起步、K=12 对比
2. **热度漂移**：top-K 集合 ~90% 稳定兜底；重排周期内分布平稳假设成立
3. **warmup 估计**：渐进下 routed_m_blocks 增长中，估错只影响 L1/L2 排序效率
4. **批量粒度**：逐 expert publish ~48 次 warp 同步开销；按新到块数阈值摊薄
5. **EP32/EP64**：expert/rank ~12-16，渐进更简单、收益更大（32/64 src 发布更慢）；
   集群验证待权限
6. **ECO**：早集限 fixed slots（§2.1）；表生成读 EcoMeta num_fixed/phy2log 过滤；
   push_dispatch 分支不动

## 9. 关键代码位置

- fused dispatch：`gc_moe/impls/sm100_fp8_fp4_gc_moe.py` ~L1855（route loop）、~L2360（publish）
- count 等待：`gc_moe/scheduler/gc_moe.py` ~L428-445（plain per-lane wait）
- pool 前缀：`locate_routed_pool_block` ~L158；任务循环 `schedule_tasks` ~L490-620
- index 布局：`(local_e*num_ranks+src)*capacity+slot` @ `SIGNAL_OFFSETS['size']`
- count 协议：`recv_count[src][local_e]` matrix + `recv_count_sum[local_e]`（高32=到达数）
- trace：`_eco_trace_rel` slots 37-40 + ring fields

## 10. 状态

- [x] fused dispatch、publish red.release.sys+CTA 分块、探针分解、热度分析
- [~] 阶段 A：表生成 + src 相位旋转发布已实现（GC_MOE_XORDER/_ROT，tensor_maps 槽位传表）；
      待 GPU 空档 bit-exact + trace 验证
- [ ] 阶段 B（dst 渐进调度 §5.2）/C；生产侧表生成钩子；EP32/EP64 验证

---

## 附录：x-order 表生成（离线参考实现）

```python
import numpy as np
# routing: [calls, ranks, T, topk] 全局 expert id；L = experts_per_dst
calls, R, T, K = routing.shape
L = num_experts // R
e = routing.reshape(calls, R, -1)
cnt = np.zeros((calls, R, L), dtype=np.int32)
for c in range(calls):
    for d in range(R):
        sel = e[c][(e[c] // L) == d] % L
        cnt[c, d] = np.bincount(sel, minlength=L)

Ktop = 8  # 每 dst 早发布 expert 数
per_dst = []
for d in range(R):
    X = cnt[:, d, :].astype(np.float64)     # [calls, L]
    S, cur = [], np.zeros(calls)
    for _ in range(Ktop):                    # 贪心前 Ktop 个
        gains = np.maximum(cur[:, None], X).mean(0) - cur.mean()
        eb = int(np.argmax(gains)); S.append(eb); cur = np.maximum(cur, X[:, eb])
    rest = [i for i in np.argsort(-X.mean(0)) if i not in S]
    per_dst.append(S + rest)                 # 长度 L 的全序

# 产物 = per_dst[R][L] 的 local 序表（uint16×R*L），即每 dst 的 x-order。
# dst 侧消费序 = per_dst[self]；src i 的发布序由旋转规则过程化生成：
#   位置 p → 轮 j=p//R、偏移 k=p%R → dst=(i+1+k)%R, expert=per_dst[dst][j]
# （E=R*L 时 p 遍历 0..E-1 恰好覆盖所有 (dst,expert) 一次）
```

注：贪心只选前 Ktop 个进早集；其余按 mean 降序（只影响 bulk 相对先后，不影响正确性）。
生产侧在权重重排时对当次校准 routing 重算，输出 `per_dst` 表
（uint16×num_experts）作 kernel 常量参数；dst 用 per_dst[self]，
src 用旋转规则现算，不需要物化 R 张表。

**ECO 变体**：候选集先按 EcoMeta `num_fixed`/`phy2log` 过滤为 fixed slots（单副本），
再跑同一 greedy；K_top ≤ num_fixed/dst；L_d 若不均则 src 侧按 per-dst 长度处理轮边界。

