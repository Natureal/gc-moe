# self-route 实施记录（已实现 + 实测）

开关 `GC_MOE_SELF_ROUTE=1`（默认 0，保留旧路）。plain pull 模式专用（eco / push / xorder /
prog_native / shard_counts 全部互斥）。

## 一、最终实现（role1 = dispatch 角色内部完成）

关键认知修正：**pull 是 role1（dispatch kernel）做的，不是 role2**。role2 只消费 ring。
所以 self-route 全部在 role1 内部完成，没有任何跨 role 协议改动。

1. **src 归属**：每个 dispatch CTA 独占整条 source：`src = block + k * dispatch_sms`，
   `waves = ceil(R / dispatch_sms)`（EP32: dispatch_sms=16, R=32 → waves=2）。
   于是每个 `(expert, src)` 的 slot 游标只有一个写者，无需跨 CTA 协调。
2. **传输**：每个 owned src 一次 `bulk_g2s_1d`（`gc_moe/ptx/tma.py:55`，裸地址、无 tensormap），
   从 `mapped_peer_base(peer_ptrs, src) + topk_offset_words*4` 读
   `num_tokens*topk*8` 字节（T=128, topk=6 → 6144 B）到 **空闲的 pull staging 区**
   （`shared_layout.pull`，零新增 SMEM 预算），每 src 一个缓冲 + 一个 mbarrier。
   waves 个副本同时 in-flight。
3. **扫描**：全部 768 线程按 `dispatch_warps*32` 步长并行；命中本 rank 的 expert 时
   `atom_shared_add_u32` 分配 slot，本地写 `source_idx[(e*R+src)*capacity+slot] = token_topk`。
   计数表复用 `expert_token_count`（R*L u32 = 1536 B，self-route 下路由直方图不用它）。
4. **计数发布**（warp0，每 CTA 一次）：
   - `expert_recv_count[gen][(src*L+e)] = count`（本地）
   - `expert_recv_count_sum[gen][e] += (dispatch_sms*owned)<<32 | subtotal`
     （GPU-scope release atomic，本地内存），使原 `dispatch_sms*num_ranks` 到达门自然闭合。
5. **role1 跳过**：两遍路由循环的远端 `source_idx` 写（原 2016-2030）、count 远端发布、
   count 交换。**保留 `grid_sync`**：把本 rank 各 CTA 的本地 `source_idx`/count 行发布给
   自己的 pull 循环（per-rank 屏障，很便宜）。
6. pull 循环、`schedule_tasks`、role2/role3 一字未改。

## 二、实现中踩到的三个坑（都很隐蔽）

1. **capacity ≠ T**：`self.capacity = sym_buffer.num_max_tokens_per_rank = 1920`（不是 T=128）。
   第一版按 capacity 定缓冲（1920*6*8 = 92 KB/src）→ 184 KB > pull 区 122 KB → 守卫静默
   禁用 self-route（日志里只有一行 `GC_MOE_SELF_ROUTE=1 ignored`，跑出来的"on"其实是旧路）。
   修正：按 **`self.num_tokens*topk*8`**（constexpr，6144 B）定缓冲，只扫前 `num_tokens*topk`
   条（其余是上一 call 的陈旧数据）。
2. **计数发布必须在单 warp 内**：最初发布循环写成了所有 warp 都执行 → 24 个 warp 各发一遍
   → sum 高 32 位变成 24×512 = 12288，role1 的 `dispatch_sms*num_ranks=512` 门永不闭合
   → `GC recv-count timeout` → `device_trap()` → 表现为 `LAUNCH_FAILED`。
3. **cute-dsl 不允许 dynamic for 套在 dynamic if 里**（`TYPE_UNSTABLE_JOIN`）。
   改为把循环提出 if：先算 `scan_stop`（0 或 valid），循环本身无条件。

## 三、实测（EP32 hole 8 节点 32 rank，T=128, topk=6, E=384, bm96, cold L2, 20 calls）

### bit-exact
`--check --dump-y` 两路各跑一遍，32 rank × 655,360 个 bf16 全部 **max|diff| = 0**，
0 个不一致元素（20,971,520 个值）。

### e2e（每次 20 calls，`slowest_rank_median_us` 的中位数；off/on 交替跑）
| rep | off | on |
|---|---|---|
| 1 | 172.16 | — |
| 2 | 166.23 | 152.88 |
| 3 | 175.33 | 152.02 |
| 4 | 168.95 | 154.10 |
| 5 | 163.42 | (崩，见稳定性) |
| 6 | 161.71 | 156.26 |
| 7 | 164.40 | 155.19 |

off n=7：mean 167.5 / median 166.2；on n=5：mean 154.1 / median 154.1
→ **−13 µs（−8%）**，两组完全不重叠（Mann-Whitney p≈0.0013）。
（首轮估计的 −19 µs 是拿早期 off 值比的，off 后来整体漂低了，故以全量数据为准。）

### 相位（GC_MOE_ECO_TRACE=1 + --profile-calls 3；中位数 over calls，max over ranks，µs）
| 相位 | off | on | Δ |
|---|---|---|---|
| route_start（最早 CTA，含扫描） | 0.83 | 2.98 | +2.1（扫描成本） |
| grid_join | 39.52 | 31.26 | −8.3 |
| pull_entry（最早 warp） | 47.81 | 32.26 | −15.6 |
| pull_exit | 109.25 | 83.65 | −25.6 |
| role2 进任务循环 | 38.21 | 4.86 | **−33.3** |
| role2 scheduler 入口 | 33.34 | 2.11 | −31.2 |
| role2 cluster barrier1 放行 | 33.15 | 1.92 | **−31.2** |
| gemm_done | 168.61 | 140.22 | −28.4 |
| role3_done | 173.38 | 145.15 | −28.2 |

**机制确认**：role2 启动段的 cluster barrier 从 ~33 µs 塌到 ~2 µs —— 正是三轮 trace 推断的
"24.5K 笔散乱 4B 远端写饱和 L2/NoC 拖慢 role2 启动"被直接消灭。role1 的 pull 因此提前
15.6 µs 开始、25.6 µs 结束，GEMM/role3 尾巴跟着提前 28 µs。

注意：trace 里的 `route_done`(27 µs) / `grid_join` 等字段被插桩自身污染（每 CTA 24 warp ×
2 次 volatile 全局读 + atom_max 打同一地址），只能横向比较，不能当绝对值看。

### 稳定性
5 次 on run 里 1 次（v5）崩：rank8 首发 `LAUNCH_FAILED`，其余 rank
`GC token readiness timeout ... peer=8`。与 AGENTS.md 记录的本集群历史 flaky 故障同形态
（随机 rank 首发 IMA/ILLEGAL_INSTRUCTION，其余 rank 等它超时）。off 侧 5 次未复现，
样本还不足以归因，需要更多重复才能判定是否与 self-route 有关。

## 四、后续可做（按价值排序）

1. **稳定性采样**：on/off 各跑 10 次以上统计崩溃率，确认 v5 是否只是历史 flaky。
2. **role1 的 CTA 启动错峰存疑**：f16（最早 CTA）0.8→3.0 µs vs f17（最晚 CTA）27→29 µs，
   两路都是这个形态。但 f16/f17 都是每 CTA 24 个 warp 各打一戳 = 384 次 atom_max 打同一地址，
   与之前 f32 全 warp 版自扰 ~12 µs 同源，**f17 的 27 µs 大概率是插桩自身串行化**，不是真实错峰。
   验证需要把 f16/f17 改成每 CTA 单 warp（本轮试过把 ring 扩到 40 字段加干净戳，结果 trace 模式
   连续两次挂（token readiness timeout），退回 36 字段后 trace 恢复正常 → **不要再扩 ring**，
   要测就改现有字段的发射者）。
3. **扫描成本 2.1 µs**：可用 4 个缓冲把 2 个 wave 的副本都提前发；收益很小。
4. **大 T**：T=1024 时 `waves*T*topk*8 = 98 KB` 仍能塞进 pull 区，但 T≥2048 会被守卫禁用，
   需要分块流水。
5. 默认值：等稳定性采样后再考虑把 `GC_MOE_SELF_ROUTE` 默认打开（ECO/xorder 仍走旧路）。
