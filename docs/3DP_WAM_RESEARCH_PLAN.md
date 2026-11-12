# 3DP-wam：资源准备、研究设计与实验记录

更新日期：2026-11-12。项目目录：`/data/haoyang/3DP-wam`。

## 1. 当前范围与结论

本轮按用户最后要求仅复制 infra、准备官方源码/预训练权重、检查资源和撰写设计文档。**不启动实验：没有执行 baseline smoke、模型 forward/backward、特征提取、缓存构建、训练、offline evaluation 或 simulator rollout。** CPU 上构造官方模型并严格加载 checkpoint 属于权重准备校验，不能视为实验通过。

最初终端因 `bwrap: Failed to make / slave: Permission denied` 无法启动；权限环境恢复后已继续实际准备。原空目录 `3DP-WAM` 已调整为用户要求的项目名 `3DP-wam`。

源 infra 为 `/data/fcy/code/openwam-exp`，复制时 commit 为 `d5785d1cdb92c027785e861daca06c6e65c0dd99`，工作树干净。通过本地 Git clone 复制 **1,829 个受版本管理文件**，逐文件比较 SHA-256/符号链接目标全部一致；后续仅修改项目说明和新增准备工具，没有改动 `openwam/`、已有训练/评测代码或实验 YAML。详细来源见 [infra_copy_manifest.json](../preparation/infra_copy_manifest.json)，原始结果见 [infra_copy.log](../preparation/logs/infra_copy.log)。

源项目的 `README.md`、`result.md`、`CHANGELOG.md`、`EXP_RESULTS_LOG.md` 和既有实验文档保留历史内容；其中成绩全部属于源项目，不能当作 3DP-wam 本轮结果。没有复制源 `outputs/`、历史 `reports/`、大规模 feature cache、无关 `genfirst/`、`RoboScope-Kit/` 或源虚拟环境。

## 2. 本地代码检查结果

优先复用的现有接口：

| 能力 | 现有文件/入口 | 后续使用方式 |
|---|---|---|
| 训练、cache、评测统一入口 | `scripts/experiment.py` → `openwam/experiments.py` | 保持原入口，新增配置/数据字段 |
| 基线及公共协议 | `configs/experiments/E01.yaml`、`configs/protocol.yaml` | 保留源 DINOv3-H+ 对照及原 DINOv3 YAML |
| Encoder contract | `openwam/model/video_backbone/encoder/base.py` | 严格遵循 `VideoEncoderProperties` 和五维 latent |
| Encoder registry | `openwam/model/video_backbone/encoder/registry.py` | 新 wrapper 按原 registry 注册，不改工厂语义 |
| 原 DINOv3 | `openwam/model/video_backbone/encoder/dinov3.py` | 支持 framewise、可选因果时间池化、冻结和部署资产 |
| 已有 VGGT-Omega | `encoder/vggt_omega.py`（相对上述 encoder 目录） | 参考同时间多视图及防未来泄漏做法；与原始 VGGT 分开报告 |
| 现有 frozen cache | `encoder/vggt_feature_cache.py`、`scripts/precompute_image_features.py` | 复用身份校验/原子提交；新增按 episode 分片层 |
| LIBERO reader | `openwam/dataloader/libero.py`、`libero_native_stereo.py` | 保留原动作格式和采样规则，显式增加几何元数据 |
| 资源管理 | `assets/registry.json`、`scripts/download_assets/download_visual_encoder.py` | 新下载复用 resumable downloader、`encoder/` 和 `assets/pretrained/` 布局 |
| 条件输入 | `openwam/model/architectures/base.py::_append_proprio_context_token` | 已有追加 context token 并扩展 bool mask 的模式 |
| Video backbone 条件接口 | `openwam/model/video_backbone/wan_backbone.py::prepare`、`base.py::BlockLoopState` | 已处理 `context/context_mask`，是 sidecar 首选接入候选 |
| Action 条件接口 | `openwam/model/action_backbone/separate_action_dit.py::_prepare_context` | 已接受 raw context 和有效 mask，需检查选定 architecture 路由 |

现有 `VideoEncoder.batch_encode` 只有 RGB tensor 输入：`[B,3,T,H,W] → [B,C,T_lat,H_lat,W_lat]`。Utonia 所需 RGB-D、点云、相机和 mask 无法通过这个签名单独表达，因此接入点云需要独立 representation adapter 和明确的数据路由，不能把缺少输入的 wrapper 注册成“已兼容”。本轮没有新增空壳 `utonia.py`/`vggt.py` 或宣称四组已能 forward。

源 M29/E01 本身使用 `from_scratch`。后续 sidecar 保留已有 WAM 权重时，应从已验证 baseline checkpoint 出发；“复用冻结 DINOv3”和“复用预训练 Video DiT”是不同状态，必须分别记录。

## 3. 模型资源与可追溯性

### 官方来源

- Utonia 官方源码：[Pointcept/Utonia](https://github.com/Pointcept/Utonia)，实际源码 commit `5fce241d3fd943bdef2db14fdd41fff8b090d98e`；权重仓库 [Pointcept/Utonia](https://huggingface.co/Pointcept/Utonia)，固定 revision `9457f025e36c7001ee962a542f0d27a5b600e1e7`。优先下载 **Tiny**，不下载含训练状态的 7 GB pretraining 包。
- VGGT 官方源码：[facebookresearch/vggt](https://github.com/facebookresearch/vggt)，实际源码 commit `a288dd0f14786c93483e45524328726ab7b1b4ce`；原始权重 [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B)，固定 revision `860abec7937da0a4c03c41d3c269c366e82abdf9`。源码使用 sparse checkout 提取模型代码及根目录说明，避免下载大体积演示数据。
- DINOv3 H+/L 使用源 infra 的本地资产、config、许可证及参考 hash，不重新下载，也不以默认 DINOv3-B 替换原 H+ 对照。

| 模型 | 项目内权重路径 | 字节数 | 官方/源 registry SHA-256 |
|---|---|---:|---|
| Utonia Tiny | `encoder/utonia-tiny/utonia_tiny.pth` | 37,390,848 | `e49a1944403d4ec3df459537fb7d5b132326736d0ebdb7c5a74868f3fa50d137` |
| VGGT-1B | `encoder/vggt-1b/model.safetensors` | 5,026,367,224 | `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e` |
| DINOv3 H+ | `encoder/dinov3-hplus/model.safetensors` | 3,362,432,800 | `3e1d4d18b9bfa9f28fad8e9de6a783f1313532d3460efa4cd0b12521d81d1a4d` |
| DINOv3 L | `encoder/dinov3-l/model.safetensors` | 1,212,559,808 | `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179` |

VGGT 在 `/data/fcy/model/vggt/model.safetensors` 已有本地文件，只有与官方 LFS hash 和大小完全一致才复用；项目内建立资源链接，避免重复传输 5 GB。DINOv3 H+/L 链接至源 `openwam-exp/encoder/` 对应目录。共享模型和共享文本 cache 仅用于读取，后续新增几何 cache 与训练输出必须位于本项目，不能写回源仓库。

Utonia 源码为 Apache-2.0，而其权重 model card 标注 **CC-BY-NC-4.0**。原始 VGGT-1B 权重同为 **CC-BY-NC-4.0**；VGGT 源码许可与商业 checkpoint 许可不能替代原始权重许可。各新模型目录保存 model card、权重许可文本、源码许可副本和 `manifest.json`；下载锁定规格见 [geometry_registry.json](../assets/geometry_registry.json)。

下载器使用源 infra 的 `.part` + HTTP Range 续传机制，失败保留部分文件，完成后校验官方大小和 SHA-256；全部资源校验完才将 manifest 标记 ready。许可证站点对 urllib 返回 403，准备工具对该站点使用 curl，保留首次失败与重试日志。缺少权重或不匹配直接失败，不使用随机权重回退。

### 准备命令（不包含实验启动）

```bash
cd /data/haoyang/3DP-wam
.venv-geometry/bin/python scripts/download_assets/download_geometry_encoders.py --name utonia-tiny
.venv-geometry/bin/python scripts/download_assets/download_geometry_encoders.py \
  --name vggt-1b --reuse-from /data/fcy/model/vggt

# 以下只在 CPU 构造与严格加载，明确禁止 forward/backward。
.venv-geometry/bin/python scripts/download_assets/validate_geometry_weights.py --name utonia-tiny
.venv-geometry/bin/python scripts/download_assets/validate_geometry_weights.py --name vggt-1b
.venv-geometry/bin/python scripts/download_assets/validate_geometry_weights.py --name dinov3-hplus
```

新增依赖放在 `.venv-geometry`，以源 OpenWAM Python 的 system site-packages 为基础，避免改动源环境。本机原环境为 PyTorch `2.7.1+cu128`、Transformers `5.17.0`；PyPI 不提供 `spconv-cu128`，准备阶段安装 `spconv-cu126==2.3.8` 和 `cumm-cu126==0.7.11`。CPU 严格加载成功也不能证明 CUDA 稀疏算子 forward 兼容，该项必须在后续 P1 独立验证。Torch Scatter 使用 PyG 的 `torch_scatter==2.1.2+pt27cu128` wheel；其余新增依赖包括 `addict==2.4.0`。完整安装输出与验证依赖版本保存在 preparation。

初次用当前 Transformers 的模型直接 `load_state_dict` 加载旧 DINOv3 文件，遇到 `layer.*` 与 `model.layer.*` 键名差异。最终沿用源 infra 的 `AutoModel.from_pretrained`，执行 Transformers 自带键名迁移，再严格检查 loading info：missing、unexpected、mismatched、error 均为空；未修改或重写 checkpoint，也未用 `strict=False` 掩盖缺失权重。首次失败日志与最终成功日志同时保留。

资源检查还发现源 registry 中 `assets/pretrained/wan21-i2v-14b-480p` 的目标目录原本就不存在；本轮保留其引用并在 `PREPARATION_STATUS.json` 列出。这是未准备的独立 Wan-I2V 特征分支，不是本轮 A/B/C/D 的新增权重。源配置仍含服务器绝对路径和少量占位路径，运行时需通过公共入口解析实际资产，不能据此宣称所有旧分支都已可运行。

## 4. 四组核心实验与公平比较

以下全部为待实现设计，当前没有启动：

| 组别 | 冻结 encoder | 输入及表示 | trainable 部分 |
|---|---|---|---|
| Baseline | 原 DINOv3 H+ | RGB → visual semantics → 原 WAM | 原 backbone/action；保留 E01 与原 DINOv3 YAML |
| A | VGGT | RGB → visual geometry | 必要投影/representation adapter + WAM |
| B | Utonia Tiny | RGB-D → point cloud → explicit 3D tokens | 点云 representation adapter + WAM |
| C | Utonia Tiny + DINOv3 | point geometry + visual semantics | fusion/interaction adapter + WAM |
| D | VGGT + Utonia Tiny | visual geometry + explicit point geometry | fusion/interaction adapter + WAM |

所有组锁定同一 episode split、数据版本、action backbone、World DiT 结构容量、训练步数、global batch、action horizon、目标采样 offset、评估 seeds/reset/renderer 和 checkpoint selection。增加 adapter 带来的参数量单独报告；通过公共投影宽度、query token 数和 adapter 层数约束额外容量。

RGB 与 RGB-D 是输入信息差异。真实/仿真深度下的 B/C/D 与仅 RGB 的 A/baseline 必须标注模态差异；VGGT-predicted depth 下的 B/C/D 则标注额外几何 teacher 成本，不混成同一“真实 RGB-D”实验。不改变数据切片来绕过缺深度，也不让不同组的有效训练 episode 悄然变化。

源协议为 10 Hz 数据、action horizon 32、历史/current + 8 future visual timestamps，offset `[-4,0,4,8,12,16,20,24,28,32]`；是否保持这些设置以 P0 通过后锁定版本为准。不得把“相同 epoch”替代相同更新步数和实际样本预算。

## 5. Encoder replacement：严格 latent contract

计划新增：

- `openwam/model/video_backbone/encoder/vggt.py`：官方 VGGT aggregator features，明确去除 camera/register special tokens，保留真实图像 patch 位置及相机身份；冻结并禁止训练模式漂移。
- `openwam/model/video_backbone/encoder/utonia.py`：官方 frozen point/superpoint 特征提取。若原 VideoEncoder 接口不足以传入点云，wrapper 只作为独立 provider，由 representation adapter 路由，不破坏现有 RGB 接口。
- `openwam/model/representation/`：位置、valid mask、batch/frame index、相机、inverse mapping 与 tokenizer 元数据，以及投影/重采样/时间对齐模块。

点云表示首先使用 `[B,T,N,D]`（或 packed features + batch/frame offsets）、`xyz[B,T,N,3]`、`valid[B,T,N]`，坐标为统一 frame 的米制值。3D voxel/superpoint 聚合保留原点到 token 的映射；确定性采样或随机种子进入 provenance。

只有建立明确空间意义后才输出 `[B,C,T,Hg,Wg]`：优先使用已标定相机的 image-aligned rasterization，在有效深度处对点特征投影/聚合，空洞有独立 occupancy mask；可选固定体素/BEV 网格必须明确其空间轴含义。现有 `spatial_compression` 是图像像素压缩比，不能用虚构整数描述任意 3D/BEV 网格。无法表达时保持独立 adapter，而不是假冒 VideoEncoder。

官方原始 VGGT 为 patch14，源 DINOv3 为 patch16；相同 224 输入会自然产生不同 native grid。公平 token budget 如需统一，应显式做基于像素坐标的固定 grid resampling，再投影到统一通道数，记录重采样及其开销，不把两者 grid 当作天然相同。不同 native resolution 的结果单独列出。

每组正式训练前必须检查：latent shape、真实有效 token、temporal compression、first-frame/clean-prefix 语义、dit patch geometry、`VideoEncoderProperties`、input/output projection channels、位置编码和 target normalization、mask 是否同时进入 attention 与 world loss。现有 RGB contract 没有点云有效 mask 返回值，新增数据路由需要贯通 mask，不能用全零 padding 冒充有效 world target。

replacement 使用源 `from_scratch=true` 机制重新初始化 DiT 时，在所有对应 replacement 对照中使用相同结构、种子规则和预算；投影/重采样参数分别统计。不可仅用 `strict=False` 跳过 checkpoint 形状冲突。加载/部署同时保存 architecture、tokenizer、geometry/schema、adapter state 和所需 config；保存后应做严格恢复校验。

## 6. Sidecar geometry conditioning：后续主路线

保留原 visual encoder、视频 latent geometry、DiT patch embedding 和已有 WAM checkpoint。冻结 Utonia tokens 经 trainable adapter 后，优先沿现有 **raw `context` + `context_mask`** 接口注入；源代码已经用该模式追加 proprio token，Wan backbone 和 action context 也有相应投影/mask 机制。adapter 输出维度对齐已配置的 context width，而非修改冻结视频特征通道。

具体接入需在 P0 后验证选定 dual-system variant 的训练/生成路由、条件丢弃、CFG 分支、context padding 和 action/video 两条路径。首版只追加推理时可见的观察几何 tokens，以维持预训练接口；context-only sidecar 不改变原视频 encoder 的 `VideoEncoderProperties`。mask 的 True/False 语义在模块边界统一。

需要依赖候选动作的 residual conditioning 时，优先在 architecture 管理的 token/condition 扩展点实现独立 gated residual adapter，关闭时恒等；不直接修改官方 attention 内部实现。两种方案按本地路由和验收结果选择，本轮不预先硬编码。

## 7. Hybrid action-conditioned interaction adapter

独立模块计划为 `openwam/model/representation/interaction_adapter.py`。输入包含 frozen point tokens/xyz/mask、frozen visual/geometry tokens 及对应空间位置、robot state、可选 goal、可选 candidate action/action context；输出 fused spatial tokens、interaction-aware tokens、可供 dynamics head 使用的 features 及 valid mask。

三种模式使用同一接口和可比较容量：

1. Geometry-only：位置编码、masked pooling/query attention 与投影。
2. Simple multimodal fusion：几何和视觉经投影与 modality/view embedding 融合，支持空/缺失模态 mask。
3. Action-conditioned fusion：加入 state、goal 和允许读取的 action context，采用轻量 Transformer/cross-attention。

**信息隔离是结构约束**：world prediction 可在明确时间段使用真实动作监督条件；policy/action generation 只读取观测历史和推理时自身的 noisy/candidate action，不能读取未来真实动作。M29 joint attention 中，真实未来动作条件若进入共享 context 会泄漏至 policy，必须使用 branch-specific routing/mask 或分开的 world-only 调用。禁止把 world-only teacher features 缓存后直接提供给 policy。

同样限制 VGGT 的多帧聚合：观测 token 不能联合未来 RGB 计算。优先同一 timestamp 的多视图联合、跨时间独立提取；要用历史多帧则编码窗口身份和截止时间进入 cache key。future target encoder 与 policy conditioning 使用可审计的独立可见窗口。

## 8. LIBERO 数据接口与深度来源

实际检查文件为 `/data/fcy/code/OpenWAM/data/libero/meta/info.json`、`conversion.json`，快照见 [dataset_audit.json](../preparation/dataset_audit.json)。转换元数据记录 source `lerobot/libero` revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`，40 tasks、1,693 episodes、273,465 frames，10 Hz；本轮仅检查元数据，没有迭代训练数据。

声明字段包括两路 RGB video、10D action、10D robot state、episode/frame/timestamp 等。两路视频的 `video.is_depth_map=false`。**当前元数据未声明 depth、point cloud、camera intrinsics/extrinsics、object pose 或 contact labels。Robot EEF pose 不等于 object pose。** 原始 10D 为 native delta EEF10 控制表征，再映射至统一 80D 模型接口；其中 action xyz 是归一化控制指令，不能当作米制位移。

后续复用原 sampler，增加可选数据字段并显式标注：

| 来源模式 | 所需条件 | 报告方式 |
|---|---|---|
| recorded/real depth | 检查原始记录、标定和同步字段 | 仅真实存在时启用 |
| simulator supplemented depth | 匹配场景版本、状态、相机及重放时间点 | 标记仿真补充；先验证 RGB/轨迹重建一致性 |
| VGGT-predicted depth | 冻结 VGGT 输出及置信度、可见窗口 | 标记预测深度；不作为同源评估 ground truth |
| no depth | 保留原 RGB 数据 | 支持 baseline/A，禁止悄悄用常数/随机点云跑 B/C/D |

每个样本携带 episode_id、frame_index、timestamp、action interval、camera_id；RGB/depth/point/state/action 对齐原采样 offset。深度通过 intrinsics 反投影，经明确 `T_world_from_camera` 或 `T_base_from_camera` 变换，统一米制单位。resize/crop 更新 intrinsics，失效深度、遮挡和投影越界进入 mask。

VGGT 单目/多视图重建的尺度和 reference frame 不能假设等于机器人基座的真实米制坐标。必须用已知标定/尺度锚或显式对齐；无尺度时使用 scale-aware 单独条件，不将未标定值冒充 metric。每帧随机中心化/缩放不能抹掉机器人与物体相对位置，必要时保留可逆变换。

## 9. Feature cache 设计

保留现有逐帧/内容 hash cache，实现时增加 **episode 分片、按需索引** 的 representation cache。单个 episode 过大可再分 frame chunks，训练仅加载当前 episode/chunk，采用有上限 LRU；严禁读取全部 episode feature 到 RAM。

identity/manifest 至少包含：数据 split/版本/hash、encoder checkpoint hash 与代码 commit、dtype、预处理、点采样/tokenizer 版本、相机标定/hash、坐标单位/变换、帧时间戳及可见窗口。保存 features、xyz、valid、frame offsets、camera/view、inverse correspondence 与来源标签，不缓存 trainable adapter 输出。冻结模型 `eval()` + `requires_grad_(False)`，encoder 输出 detach；不能把 trainable adapter 包在 `no_grad` 内。

分片采用 lock、临时文件和原子 rename，manifest 只标记已完整验证的 shard；中断恢复跳过完整 shard、重建缺失或损坏 shard。cache identity 变化时独立 namespace。新 cache 位于本项目 `cache/geometry/`；当前复用的 `cache/text` 链接只用于读取。

## 10. Representation benchmark 与统计口径

复用现有训练和 LIBERO rollout；新增离线 metric dispatcher，对每个指标记录有效样本数和标注来源。

| 指标 | 定义/所需真值 | 缺少条件时 |
|---|---|---|
| Future object pose error | 对象平移米制误差 + SO(3) 旋转角误差；匹配 object id | N/A，不能以 EEF state 替代 |
| 3D motion/flow EPE | 同一物理点的预测位移与真值位移差，考虑遮挡/有效性 | N/A 或单列明确标注的伪监督实验 |
| Future contact F1 | 固定 horizon/阈值；报告 precision/recall/support | 缺 contact labels 时 N/A |
| Latent temporal consistency | 基于追踪/投影/几何匹配后的对应关系；显式处理真实物体运动 | 禁止无序点云按索引 MSE |
| Action prediction error | 原控制表征和物理有效分量的误差；排除 80D padding | 保留 normalized 与反归一化口径 |
| LIBERO rollout success | 与源一致的 task/reset/seed/action chunk/renderer | 环境不可用时标记未执行 |

不以 VGGT 自己生成的伪深度或预测 pose 作为独立 ground truth 宣称几何精度提升。跨帧 correspondence 可用 calibrated reprojection、object tracks、scene flow 或受约束匹配；Chamfer 集合距离可作辅助指标，但不能替代对应点 flow EPE。动态场景不能简单要求跨时 latent 恒定。

每组分别记录 frozen encoder feature extraction 时间、端到端时间、实际 token 数/有效 token 数、adapter/WAM trainable parameters、GPU allocated/reserved peak、world metrics、action metrics、缓存命中率与冷/热缓存成本。先冻结 encoder，再单独安排 partial fine-tuning，不混入同一主表。

## 11. 阶段进度与验收

| 阶段 | 内容 | 本轮状态 |
|---|---|---|
| 资源准备 | 复制 infra、锁定官方源码与权重、复用 DINOv3、说明文档 | 已完成本轮目标；四个模型均通过 CPU 权重加载校验 |
| P0 | 原 OpenWAM/DINOv3 baseline smoke、训练与评估入口验证 | 未执行，按用户要求暂不启动 |
| P1 | wrappers、真实 forward、shape/mask/坐标、冻结及 CUDA 算子验证 | 未实现/未执行；权重 CPU 加载不等于 P1 完成 |
| P2 | A/B/C/D、投影、episode cache 保存/读取/恢复、统一 benchmark | 设计完成，未实现/未执行 |
| P3 | sidecar 和三种 interaction adapter；adapter backward/信息隔离 | 设计完成，未实现/未执行 |
| P4 | ≥20 steps world/action、checkpoint 保存/恢复、offline/rollout | 未执行 |

“既有配置不受影响”本轮仅有静态证据：`openwam/`、`configs/`、原训练/评测脚本保持源文件内容；运行时兼容性要由 P0 实测，不能提前宣称通过。没有产生本项目的新训练 checkpoint。

## 12. 本轮实际校验、日志与显存

资源准备已经完成，机器可读总表见 [PREPARATION_STATUS.json](../preparation/PREPARATION_STATUS.json)。以下只有资源校验结果，不包含实验成绩。

| 项目 | 状态/证据 |
|---|---|
| infra 原样复制 | 通过，1,829 文件 SHA-256/链接核对，`preparation/logs/infra_copy.log` |
| Utonia 官方权重 | 下载、官方 SHA-256、源码/权重许可、官方 loader CPU strict load 均通过；[加载报告](../preparation/utonia-tiny_load_validation.json)、[原始日志](../preparation/logs/load_utonia-tiny.log) |
| VGGT 官方权重 | 本地复用、官方 SHA-256、源码/权重许可、完整模型 CPU strict load 均通过；[加载报告](../preparation/vggt-1b_load_validation.json)、[原始日志](../preparation/logs/load_vggt-1b.log) |
| DINOv3 H+ 复用 | 原 infra hash 相符，官方 loader 键名迁移后零缺失/多余/不匹配，CPU 加载通过；[报告](../preparation/dinov3-hplus_load_validation.json)、[日志](../preparation/logs/load_dinov3-hplus_verified.log) |
| DINOv3 L 复用 | 原 infra hash 相符，CPU 完整加载通过；[报告](../preparation/dinov3-l_load_validation.json)、[日志](../preparation/logs/load_dinov3-l.log) |
| 准备工具静态检查 | 两个新增 Python 脚本 py_compile、git diff --check 通过；核心代码/配置与复制基线无差异 |
| 原 baseline smoke、四组 forward、backward、cache、20 steps、checkpoint roundtrip | 全部未执行 |

日志目录为 `preparation/logs/`：`download_utonia*.log`、`download_vggt*.log`、`geometry_dependencies*.log`、`load_*.log`；模型严格加载记录为 `preparation/*_load_validation.json`。最终准备检查见 [final_preparation_checks.log](../preparation/logs/final_preparation_checks.log)。保留失败尝试以便追溯环境问题，最终状态以成功报告为准。

| 模型 | 实际参数量 | 校验时 trainable parameters | CPU 构造/加载与检查耗时（包含 hash 等，不是 feature extraction） |
|---|---:|---:|---:|
| Utonia Tiny | 9,326,106 | 0 | 15.105 秒 |
| VGGT-1B | 1,256,537,516 | 0 | 15.361 秒 |
| DINOv3 H+ | 840,592,640 | 0 | 15.320 秒 |
| DINOv3 L | 303,129,600 | 0 | 13.816 秒 |

四者均为 eval 模式，所有参数处于 CPU、无 meta parameter/buffer，`requires_grad=False`。这只证明准备时的模型冻结状态；训练期间 encoder 不更新及 adapter 能 backward 的验收仍未执行。无随机初始化参数残留作为预训练结果使用。

| Encoder | GPU peak memory | Feature extraction cost | 本项目 world/action 结果 |
|---|---|---|---|
| DINOv3 H+/L | 未测量 | 未测量 | 未执行 |
| Utonia Tiny | 未测量 | 未测量 | 未执行 |
| VGGT-1B | 未测量 | 未测量 | 未执行 |
| C/D 融合组合 | 未测量 | 未测量 | 未执行 |

本机可见 4 张 H100 80GB，已有其他任务使用大量显存。本轮未分配 GPU；`nvidia-smi` 中整卡占用不归因于任何 encoder。本轮 CPU load 时间也不作为 feature extraction 时间。

## 13. 实际修改文件与后续建议

复制的全部原文件及 hash 见 `preparation/infra_copy_manifest.json`。新增/修改文件的逐路径清单（含实际检出的官方源码）见 [file_inventory.json](../preparation/file_inventory.json)。本轮相对复制基线修改/新增：

- `README.md`：项目命名及新说明入口，历史结果归属说明。
- `.gitignore`：独立准备环境忽略项。
- `assets/geometry_registry.json`：官方模型固定 revision、文件大小/hash、license 和源码位置。
- `scripts/download_assets/download_geometry_encoders.py`：复用源 downloader，支持续传、官方 hash 校验、本地等价资产复用、license/provenance。
- `scripts/download_assets/validate_geometry_weights.py`：CPU 严格加载及冻结状态检查，无 forward/backward。
- `docs/3DP_WAM_RESEARCH_PLAN.md`：本文。
- `third_party/utonia/`、`third_party/vggt/`：新增官方源码 checkout；源码版本见 manifest。
- `preparation/`：复制清单、数据元数据审计、环境与加载记录、原始日志、准备摘要。
- ignored 本地资源：`encoder/` 模型目录/复用链接、`assets/pretrained/` 链接、只读复用的 `cache/text`、`.venv-geometry/`。

下一阶段先执行 P0，确认原训练与评估正常，再实现 P1。小数据 debug 建议保留源 DINOv3-H+ 作为主对照，Utonia Tiny 起步、短可见窗口、microbatch=1；如另用 DINOv3-L 降低成本，必须单独明确该 debug baseline。公共 representation 维度、token budget、batch size 和小模型规模在实测后锁定，不提前编造可运行配置。

正式研究优先 sidecar：从已验证 baseline 权重继续，保留原视频输入和 DiT 几何，先 geometry-only → simple fusion → action-conditioned fusion。replacement 四组作为独立 from-scratch 对照；相同训练预算和初始化规则，不与保留 checkpoint 的 sidecar 混报。深度来源、时间对齐及独立 3D 真值未解决前，不开始 B/C/D 的正式排名实验。
