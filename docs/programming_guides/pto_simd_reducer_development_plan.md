# PTO SIMD Reducer 开发计划

## 目标与基线

按 reviewer《TileLang PTO SIMD Reducer 实现方案》第 1–4、7、8 章实现
`alloc_reducer → reducer_init → reducer_update → finalize_reducer(acc, result)`。
输入直接读取 UB，accumulator/result/seed 使用普通 local；结果保持 one-lane VMI，
支持单元素 UB 写回及后续 Parallel 按需广播。第 5、6 章的外围普通算术不纳入实现。

- 依赖 PR：https://github.com/PTO-ISA/tilelang/pull/38
- 固定起点：`841d2336b04f9693d2a9e339c660772c5270af6e`
- 开发分支：`feat/pto-simd-reducer`
- PR 仓库：`liuzidi/tilelang`
- PR base：`feat/pto-parallel-private-alloc`，不是 upstream main。
- 本地 worktree：`/home/liuzidi/.codex/worktrees/pto-simd-reducer/tilelang`

只提交相对 PR38 的 reducer 增量。发布前复核 PR38 head；若依赖更新，先检查差异，
再更新开发分支并重新运行受影响测试，不修改或强推 PR38 分支。

## 支持契约

单输出、恰好一个静态 update、float16/float32 × sum/max/min，VL=64/128/256，
只接受一步完整归约 `plan_steps=[VL,1]`。允许条件更新、尾块、重复动态 epoch、
合法 scalar 或 one-lane seed 快照。多输出、分组、多步、空计划、VL seed 明确拒绝。

普通 scalar 直接读写、直接别名和已有 direct reduce 保持原契约；Parallel 外普通
数据算术（包括 seed RHS）在 Verify 拒绝。Parallel 内允许 one-lane 子表达式，
在消费点扩大宽度，不把 one-lane VMI 降格为 scalar。

## 实现工作与完成标准

|工作|主要文件|实现与原因|完成标准|
|---|---|---|---|
|依赖核对|PR38 存储、控制流、direct reduce、测试环境|核对实际接口及缺口，记录 TileLang/TVM/PTOAS 和加载路径；不以 PR 标题代替能力验证|基线回归通过，缺口有最小复现及归属|
|贯穿示例|examples 与 reducer 专项测试|UB 输入，独立 local result，结果同时单元素写回和后续 Parallel 消费|源码解析及每个 pass 的产物可追踪|
|布局|src/ascend/transform/layout_inference.cc|从 PTO update 与 loop layout 推导虚拟 reducer PartialFragment；result 不进入 Fragment 布局|replicate/combine=VL；zero/multi-update 到 materializer 明确诊断|
|物化|src/transform/reducer_plan_materialize.cc|按 epoch 分派独立 PtoReducerPlan；普通 local accumulator/seed；原位置初始化和捕获|真实静态 update 数先于布局检查；allocation 与引用一致|
|内部 finalize|src/ascend/op/pto_finalize_reducer.h/.cc|独立 tileop，acc 读/result 写/seed 可选读，显式 plan_steps|解析、访问区域和非法 plan 检查；遗漏消费报错|
|身份与值分类|parallel_to_pto_utils.h/.cc、attr.h|metadata 仅 Array<Var> accumulator；共享 reducer helper、ValueInfo 与宽度规则|scalar/VMI1/VMIVL 在 Verify 与 Vectorize 一致，无重复生命周期状态机|
|准入与规范化|verify_parallel_to_pto.cc、legalize_parallel_to_pto.cc|复用源生命周期检查；检查 local slot、plan、seed、外围算术；条件更新识别 local|错误在规定阶段报告；合法 direct reduce/scalar 读写不误拒绝|
|向量化|vectorize_parallel_to_pto.cc|accumulator data_array 跨 chunk；result local.var/data；保存 seed 实际快照；VF Buffer 映射与词法 Bind 环境|初始化位置不移动、inactive lane 保留旧值、one-lane provenance 不丢失|
|finalize 与消费|vectorize_parallel_to_pto.cc|完整 VL mask 横向 reduce；one-lane seed combine；无强制 VL 广播；单元素 vstore|one-lane 算术→group=1 广播→chunk-mask 运算；旧 finalize 完全消费|
|下游集成|allocation marker、lowering/codegen|复用 marker carrier；检查版本替换只修实证缺口|完整 PTO codegen 和 PTOAS lowering；USE_ASCEND=OFF 不依赖 Ascend 头文件|

实施顺序：固定示例及依赖 → 内部 op/布局/物化 → 共享分析及 Verify → Vectorize 与
消费者 → 条件/tail/seed 组合 → 完整验证 → 提交与 PR。顺序不形成代码中的阶段开关。

## 关键不变量

1. 一个逻辑 slot 承载 VL 份部分和，跨所有 chunk 保留，外层每次执行 init 时重置。
2. update 使用 tail/条件 mask；finalize 使用完整参与 mask，不能复用最后一轮 tail。
3. seed 在 init 保存实际值，横向归约后合并一次；来源覆盖不改变快照。
4. result 与 accumulator 独立，one-lane VMI 的身份不能由标量 dtype 推断。
5. VF lane 数不随 one-lane 算术改变；mask_1 与 mask_VL 分别生成并在 VF 内支配使用。
6. Buffer 映射按 data Var 身份，Bind 按词法作用域；退出 unit 保留外层绑定、移除内部绑定。
7. source lifecycle 复用 VerifyReducerEpoch；不新增 materialized lifecycle 状态机。
8. 通用 materializer 仅用 Op::Get/Call 构造内部 op，避免非 Ascend 构建链接依赖。

## 验证计划

### 主机结构与诊断

- VL=64/128/256，E=1、17、VL、2*VL+1；float16/float32 × sum/max/min。
- 捕获实际 pipeline：PartialFragment、local allocation、metadata、内部 finalize、
  完整 mask、跨 chunk 状态、one-lane result、seed snapshot、消费点广播及 marker 交接。
- 负例：zero/multi-update、缺失/错误 metadata、非法 region/slot/dtype/plan、VL seed、
  unit 外算术（Bind/store/seed/别名/VF 首尾）；与前置生命周期测试不重复建设。
- 回归：普通 Parallel、PR38 local/var、cast、direct reduce、原 reducer 路径。
- 构建：USE_ASCEND=ON；检查 USE_ASCEND=OFF 通用 materializer 编译。

### PTOAS 与设备数值

- VL128：4096 个不同贡献；257 个 1 得 257；seed=7 后来源改 19 得 264。
- max 全负、min 全正；极值分别位于早期 chunk 与 tail。
- 条件部分/全部为假、外层重复 epoch、scalar 与前一 reducer 的 one-lane seed。
- result 直接写回及后续 Parallel/别名消费。
- Parallel 内 result/4096：E4096 和 E257 输出分别为 1 和 257/4096。
- 与现有 direct reduce、scalar 访存混合，核对真实读写依赖。
- 短输入 E1/17；VL64/256 的 E=2*VL+1；VL256 必须经过 PTOAS lowering。
- 输入补齐并使用干扰 padding，输出使用 sentinel；按 dtype 明确数值误差标准。
- 不把 skip、编译成功或文本检查当成设备数值通过。

独立 build/cache/artifact 目录；记录源提交、TVM/PTOAS 提交、Python/native 实际加载路径
和生成物指纹。设备验证遵循本机 task-submit 规则。NaN/有符号零先核对源与后端约定，
未证明前不宣称语义一致。

## PR 交付与进度记录

PR 描述说明行为、支持边界、PR38 依赖、测试命令及实际结果。遇到缺少前置能力时，
记录复现、影响和解决方式；不静默缩减完整验收，不把依赖问题标记为已通过。

- [x] 已核实 PR38 head 并创建独立 worktree/开发分支。
- [x] 开发计划落盘。
- [x] 依赖基线验证与缺口定位。
- [x] reducer 编译链与示例实现。
- [ ] 主机回归、PTOAS 与设备数值验证。
- [ ] 提交代码并创建以 PR38 分支为 base 的 PR。

当前记录：PR38 文档明确内部控制流尚未集成，条件 local 更新不能假定已经可用。
需核对控制流 PR 与当前基线的具体差异；在本分支完成必要集成或明确阻塞证据。

## 实现说明与依赖集成

### 布局和物化

PTO Parallel 先求解自己的 lane layout，再用 `AnalyzeReducerUpdateSite` 推导虚拟
`PartialFragment`。完整布局表写回 SIMD_VF；普通 local result 不进入 fragment 推导。
该 PartialFragment 仅描述参与归约的关系，不是最终存储。

通用 `ReducerPlanAndMaterialize` 根据 target 与 VF 上下文选择 PTO 分支。静态 update
计数独立于布局是否存在；零次/多次调用先报支持范围错误，缺布局单独诊断。
原 reducer allocation 替换为一个 shape=(1,) 的普通 local accumulator；seed 也使用
独立 local。所有 PartialFragment 条目在物化后清理，避免污染下游布局重映射。

init 在原位置写入 sum 的 0、max 的负无穷或 min 的正无穷，并在该位置保存 seed。
update 物化为 local load/combine/store。finalize 保留为独立内部 tileop：accumulator
读、result 写、可选 seed 读，`plan_steps` 用 `[VL,1]` 编码。解析器逐项检查数组元素
确为 IntImm，避免错误 annotation 导致无效类型解引用。

### finalize 和跨 Parallel 数据流

VF 层维护原 data Var 到新 Buffer 的映射：accumulator 为 `TxVL` local data_array，
result 为 `T` local.var data。one-lane VMI 与普通 scalar 在 TIR dtype 上相同，因而
使用独立 `ValueInfo{kind,logical_lanes}` 记录来源；不能通过 `dtype.lanes()==1`
把 result 判为 scalar。转换结果通过 `ConvertedValue` 传递分类，临时 Bind 的分类
同时记录在以持有引用的 Var 为 key 的表中，避免临时对象回收后的指针重用。

每个 Parallel 生成 chunk loop；accumulator 在 VF 层分配、在原 init 位置初始化，
因此所有 chunk 共用状态。尾块先做 masked combine，再用 vsel 保留 inactive lane
的旧 partial。条件 update 规范化为 select，未执行分支读回旧 partial，叠加 tail
处理后保留历史贡献。

finalize 读取完整 accumulator，以完整 VL mask 调用 vcadd/vcmax/vcmin，得到
one-lane VMI。scalar seed 先广播到 1 lane，VMI seed 直接读取，在 mask_1 下合并一次。
result 始终独立保存。直接 UB 写回生成无 mask 的单元素 vstore；后续 Parallel
中的 `result/4096` 先在 mask_1 下除法，再在乘法消费处 group=1 广播至 VL。

词法 Bind 环境与值环境随 Attr/Block/For 恢复；转换 unit 后恢复地址、mask、loop
mapping 等上下文。完整 mask 在 VF 内定义，并按宽度缓存；one-lane 运算不会修改
VF 的 lanes 或 unit 的地址布局。最终消费 reducer metadata 和内部 finalize，
下游复用 PR38 的 allocation marker 与 codegen。

### 当前依赖边界

PR38 head 固定为本文开头的提交。它不包含 PR11 的完整条件控制流，也不包含 PR39
的 high-level direct reduce。为使本分支在 PR38 上独立可编译、可验证：

- 条件更新仅补入 reducer 所需的单个 store 分支规范化；不包含嵌套条件、多语句分支、
  分支内 Bind 或普通条件 UB store。没有扩大为 PR11 的完整控制流实现。
- direct reduce 的检查、计划与 lowering 适配自 PR39
  `dae783c88d764ea20bccd4dca4a06a25b574dc16`，在代码中保留来源。涵盖独立 VF 与
  混合 VF、serial/unrolled 包装、buffer version 前缀；if/while/Parallel 内调用拒绝。
  未引入该 PR 的 fill 等无关能力，纳入其 direct reduce 契约测试。
- PR11/PR39 后续合入时应合并共享 helper 与入口，消除重复实现；本分支不修改它们
  的远端分支。普通 unit 外数据算术仍由 Verify 拒绝。

### 数值语义与本版边界

sum 允许 reassociation，不承诺与串行逐元素求和逐位一致。验收中的状态/mask/seed
数据采用可精确表示的数值；一般浮点输入另按 dtype 容差检查。max/min 使用已有
vmax/vmin 与 vcmax/vcmin；有限非零输入精确比较。源 TIR Max/Min 和 VMI 文档没有在
本次依赖中共同明确 NaN payload、NaN 传播及正负零 tie 的逐位规则，因此本版不声称
这些特殊值逐位一致。需要严格特殊值约定的扩展，应先固定源与后端契约再增加测试。

可运行示例：`examples/ascend/example_pto_simd_reducer.py`。专项测试同时涵盖结果
单元素写回、尾块外 sentinel、跨 unit 别名、one-lane 除法、scalar/VMI seed 来源覆盖、
重复 epoch 和 mixed direct reduce。缺少 NPU 的设备用例会失败，不将 skip 算作通过。

## 验证记录（2026-11-11）

- PR38 发布前复核：head 仍为 `841d2336b04f9693d2a9e339c660772c5270af6e`。
- TVM 子模块：`4211874e9e4b8770fd20ad0d98ab80afd1df5028`。
- 主机构建：`USE_ASCEND=ON` 完整构建通过；`USE_ASCEND=OFF` 单独配置并编译通用
  `reducer_plan_materialize.cc.o`，证明该文件不依赖 Ascend-only 实现（不是完整非
  Ascend 构建验证）。
- 主机专项及回归：348 passed。命令见下；4 个设备用例明确 deselect，未计入通过。
- 额外尝试的 CUDA reducer 套件在本环境因无 CUDA、目标默认 Ascend 与旧式无 VF
  写法不兼容而失败；14 个 CUDA scalar-candidate 用例 skip。两者均不列入通过项。
- PTOAS 运行包从 `ptoas-shared-memory-analysis-20261110` 的构建树独立复制并逐文件
  校验，源 HEAD=`476e8ce513273e364b7b22ab403f5be57bf054fd`。源工作树有其他任务的
  未提交修改，因此 **HEAD 不能单独代表此构建**；保留 source.patch、运行包 SHA256
  manifest 和实际动态库加载路径，以冻结包为验证身份。未改动其他任务的源码。
- CANN 9.2.0；Python 3.11 / tilelang-a5。缓存、临时目录和 PTOAS 运行包均独立于
  日常开发目录。设备运行前验证 TileLang 源码/动态库与 PTOAS 全包指纹。

```bash
python -m pytest \
  testing/ascend/language/pto/test_pto_simd_reducer.py \
  testing/ascend/language/pto/test_pto_parallel_private_alloc.py \
  testing/ascend/language/pto/test_pto_direct_reduce_contract.py \
  testing/ascend/language/pto/test_pto_parallel_vectorize_structure.py \
  testing/ascend/language/pto/test_pto_parallel_verify_contract.py \
  testing/ascend/language/pto/test_pto_parallel_cast_matrix.py \
  testing/ascend/language/pto/test_parallel_pipeline_routing.py \
  testing/ascend/language/simdvf/test_simdvf_reduce.py \
  -k 'not dim1_sum and not dim1_accumulate' -q
```

本机可追溯证据目录 `/home/liuzidi/reducer-validation/`：
`host-release.log`、`tilelang-manifest.json`、`ptoas-frozen/manifest.json`、
`ptoas-frozen/source.patch`、`final-env.sh`、`check-frozen.py`。
设备最终结果与 PR 链接在发布时补充。
