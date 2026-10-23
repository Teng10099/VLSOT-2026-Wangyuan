# 望远队 · VLSOT 完整时序候选

**2026 VLP 挑战赛参赛作品**

这是 2026-10-23 完整 30 帧候选的已公开源码。望远队已完成报名，代码仓库：[Teng10099/VLSOT-2026-Wangyuan](https://github.com/Teng10099/VLSOT-2026-Wangyuan)。完整源码已公开；整仓许可证依据仍待补，本版本未指定整仓许可证。师兄允许参赛、公开源码及分享权重已获用户确认，具体权利人、版本、授权范围尚待补录；主框架上游整体许可仍待补。见 `submission_docs/02_THIRD_PARTY_PROVENANCE.json`。本源码用于方法审阅和复现；尚未正式提交 VLP 评测，没有官方评分。

## 对应交付版本

实际分享包为 `望远队_VLSOT模型与ONNX.zip`，解压根目录是 `VLSOT_Wangyuan/`。下载：[百度网盘](https://pan.baidu.com/s/1bVVBg2RImI3-x0BulhllCg?pwd=2qm4)，提取码 `2qm4`。已核验百度分享页面：显示文件名与本包一致、大小 2.2G、分享时间 2026-10-23 19:17，过期时间为“365天后”，满足至少 90 天的有效期要求。尚未独立完整下载或校验远端文件 SHA256。该包总计 13 个文件，含原始模型、ONNX、4 个 RoBERTa 配置/词表文件、便携配置及 6 份说明/核验记录。ZIP 为 2,350,802,747 bytes，SHA256 `7e299a7c6281afe1d784b5aaceff45e5f7f93938a74d5964db33999769ce81e0`。本地原始审计候选目录 `VLSOT_提交候选_完整时序_20261023` 保留：

- `03_weights/`：完整时序续训第 2 轮，1033 个 state keys，去除优化器后张量逐值一致。
- `04_ONNX/model_official_v2.onnx`：同一版本导出，已完成下述图与数值验证。
- `01_raw_inference/result.zip`：原始 v2 输出，仅供比较。
- `01_prior_candidate/result.zip`：在原始输出上加入描述中的初始距离和尺寸；当前拟评审候选。
- `05_验证记录/`：内部验证证据；不是官方评分。

固定提交快照的源码 ZIP 不包含权重、ONNX、预测结果、数据、标定或 PPT，需另行取得对应包。分享包用 `share_manifest.json` 校验文件配对；原始审计目录另外保留 `artifact_manifest.json`，不能只凭文件名混用版本。主模型 SHA256 为 `2d66f572b50b69b05937761c6a3595e7e168dfccf278d86395435f909217ad10`，ONNX 为 `44ccbc8c89da96897f47be10a58531af6aa98fee3d587620d0250e8dbb2568e8`。

技术报告：[望远队 VLSOT 技术报告（PPTX，已署名，共 5 页）](reports/望远队_VLSOT技术报告_5页.pptx)。报告涵盖方法、贡献、实验与消融，供评审下载。

## 方法、贡献与边界

已有基线包含 Mono3DVG/Mono3DVLT 相关结构、Swin、RoBERTa、差分注意力、TTM 和师兄提供的训练代码/初始化 checkpoint，不能归为本次从零原创。本次工程修改包括官方数据隔离、显式几何监督与可逆解码、padding-mask 修复、时序残差、完整时序续训、严格权重契约、数值描述先验、无测试标签推理和 ONNX 适配验证。未经独立公平对照的改动不宣称具有单独精度增益。

v2 使用标定将官方 TRAIN 数值 XYZ 编码为投影中心与相机 Z，监督正值 H/W/L，解码后恢复 L/W/H 与原标签数值 XYZ。不加半高度偏移，不再变换到另一个世界坐标。`camera_geometric_center_v2` 是当前工程约定；标签物理中心和平台坐标解释尚未独立核实，格式检查和图执行通过不能证明该解释正确。

数值先验只解析输入描述中明确、无歧义的初始距离与尺寸：保持网络预测方向，将初始距离半径用于全段，并覆盖有明确依据的尺寸；不使用方位角。模糊、区间或否定描述回退原始值。距离固定可能伤害运动目标，不能宣称对所有场景有效。TEST 预测只读图像、描述、标定及模型，不读取目标 XYZ、尺寸、朝向或其他测试标签，不用 GT 回退。

## 环境与 CUDA 算子

实际运行环境：Linux、Python 3.8.20、PyTorch 2.4.1+cu118、CUDA 11.8、V100 32GB；依赖版本在 `requirements.txt`。这不代表 RTX 3060 的实测性能。先激活 Python 环境，再执行：

```bash
python -m pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
cd lib/models/mono3dvg/ops
bash make.sh
cd ../../../..
```

`make.sh` 实际执行 `python setup.py build install`，因此此处 `python` 必须与之后的 `$PY` 指向同一环境。需要匹配的 CUDA Toolkit 与 C++ 编译工具链。V100 编译架构可用 `TORCH_CUDA_ARCH_LIST=7.0`；3060 为 8.6，需要在目标机器重新构建。主模型含 CUDA 代码，不支持以 CPU 直接运行整套 PyTorch 模型；ONNX Runtime 已验证 CPU 路径。

## 资产与数据布局

```text
VLSOT_Wangyuan/
  model.pth
  model_official_v2.onnx
  portable_config.yaml
  share_manifest.json
  weights_verification.json
  onnx_verification.json
  portable_config_verification.json
  training_summary.json
  使用说明.txt
  roberta-base/
    config.json
    tokenizer.json
    vocab.json
    dict.txt
DATA/
  train/json/<六位视频ID>.json
  train/img/<图像>.jpg
  test/json/<六位视频ID>.json
  test/img/<图像>.jpg
CALIBRATION/
  camera_intrinsic/<图像ID>.json
  virtuallidar_to_camera/<图像ID>.json
```

`--source-project` 指向解压后的 `VLSOT_Wangyuan`，不是源代码目录；`--config` 使用其中 `portable_config.yaml`。入口从本地 RoBERTa 配置构建，再从完整 `model.pth` 严格恢复编码器，不需另附 `pytorch_model.bin`、`merges.txt` 或 `tokenizer_config.json`。便携配置只保留原模型结构、开关与种子，编码器路径由入口按 `--source-project` 覆盖。已用它重新推理全部 10 段/300 帧，原始 TXT 和 result ZIP 均与原候选逐字节相同；ZIP SHA256 为 `55cc7136556effa6902eca9a79928d29c20c3f060745abc617306c7d23a00386`。证据是分享包 `portable_config_verification.json`。因此复现推理不要求私有 `run_manifest.json` 或 `resolved_config.yaml`。

`tools/prepare_weights.py` 仅打包张量、模型契约和 tokenizer，不自动复制训练配置/记录。若自行重新打包，需另行提供对应便携配置、真实训练摘要及哈希记录，不能假定这些文件自动产生。原始审计目录的完整训练配置/manifest 只用于内部追溯，不随公开源码提供私人绝对路径。

官方包共 227 段 TRAIN、10 段 TEST，每段 30 帧；不包含标定。当前标定来自与全部 7110 张官方图像逐字节匹配的原始图像数据。使用者须合法取得数据与对应标定；数据和原始标签不得随源码或模型包二次分发。固定内部划分是 `configs/official_split_20261023/train.txt`（185 段）及 `val.txt`（42 段），按原始序列与共享帧隔离。序列 0042 保留在 train，0050 在 val；初始化权重更早的完整训练数据仍未查明。

## 复现当前预测：先验证，再推理，再后处理

在本源码根目录运行。将路径改成合法取得的实际资产位置，所有输出路径必须尚不存在。

```bash
PY=python
MODEL=/path/to/VLSOT_Wangyuan
DATA=/path/to/DATA
CAL=/path/to/CALIBRATION
OUT=/path/to/new_reproduction
SPLIT=configs/official_split_20261023
mkdir "$OUT"

$PY tools/evaluate_official_v2.py \
  --config "$MODEL/portable_config.yaml" --source-project "$MODEL" \
  --checkpoint "$MODEL/model.pth" --data-root "$DATA" \
  --calibration-root "$CAL" --val-ids "$SPLIT/val.txt" \
  --output "$OUT/full30_validation.json" --frames 30 --gpu 0 \
  --audit-numeric-priors

$PY tools/infer_official_v2.py \
  --config "$MODEL/portable_config.yaml" --source-project "$MODEL" \
  --checkpoint "$MODEL/model.pth" --data-root "$DATA" \
  --calibration-root "$CAL" --output "$OUT/raw" --frames 30 --gpu 0

$PY tools/postprocess_v2_numeric_priors.py \
  --source "$OUT/raw" --output "$OUT/initial_range_and_dimensions" \
  --mode initial_range_and_dimensions
```

模型契约为 `camera_geometric_center_v2`、输入宽高 960×544、两个模型开关 `official_correct_padding_mask=true` 和 `official_temporal_residual=true`。加载不允许静默切换开关或编码。每段种子 444，段首重置记忆及 phase，段内顺序传递状态。原始输出不可覆盖；先验输出写新目录并记录输入哈希。验证入口只访问给定 TRAIN val，不访问 TEST。

候选 ZIP 是 `$OUT/initial_range_and_dimensions/result.zip`，包含十个 `result/<六位ID>.txt`，各 30 行、7 列 `frame_index l w h x y z`，帧号 1..30，尺寸与坐标单位米。数量、有限性、正尺寸与哈希检查只证明封装正确。不得把 raw 与 prior 同时当成一个提交，也不要用旧 `export_vlsot.py` 再给 v2 结果施加世界变换。

## 内部结果与训练历史

下表使用相同 42 段、每段 30 帧的独立推理；中心误差是原标签数值坐标下逐帧三维欧氏距离均值。

| 模型 | 原始中心误差/m | 初始距离及尺寸先验后/m |
|---|---:|---:|
| 前 3 帧训练 5 轮候选 | 12.2360 | 6.5386 |
| 再完整 30 帧续训 2 轮，选第 2 轮 | 10.7594 | 6.3588 |

完整时序候选尺寸 MAE 为原始 0.6409 m、先验后 0.2292 m。这里没有官方 APE/ACE/SR、IoU、归一化得分或平台排名；公开资料没有给出可直接复现的完整官方公式。这些内部结果不能当成官方评分，也不能单独证明 residual 或 padding-mask 的因果收益。当前训练文本截断后填满 110 tokens，padding 修复由受控回归验证。历史旧 checkpoint 的严重定位异常属于已隔离旧流程，不能与本候选 1033-key 权重和内部结果混用。

前 5 轮历史配置如下；须另有当时合法授权的原始初始化权重，当前分享包 `model.pth`（原审计 `03_weights/model.pth`）不能代替它重放原历史：

```bash
INIT=/path/to/authorized_original_initialization.pth
RUN5=/path/to/new_first5_run
$PY train_official.py --config configs/mono3dvg.yaml \
  --source-project "$MODEL" --checkpoint "$INIT" \
  --data-root "$DATA" --calibration-root "$CAL" --output "$RUN5" \
  --label-center-reference geometric_center --gpu 0 --seed 444 \
  --input-width 960 --input-height 544 --trainable fusion \
  --epochs 5 --lr .0001 --head-lr .003 --head-init-std .01 \
  --correct-padding-mask --temporal-residual \
  --train-ids "$SPLIT/train.txt" --val-ids "$SPLIT/val.txt" \
  --max-frames-per-clip 3
```

完整时序续训从该 5 轮候选权重初始化，使用新输出目录与新优化器，不用 `--resume`：

```bash
FIRST5=/path/to/selected_first5_checkpoint.pth
RUN30=/path/to/new_full30_run
$PY train_official.py --config configs/mono3dvg.yaml \
  --source-project "$MODEL" --checkpoint "$FIRST5" \
  --data-root "$DATA" --calibration-root "$CAL" --output "$RUN30" \
  --label-center-reference geometric_center --gpu 0 --seed 444 \
  --input-width 960 --input-height 544 --trainable fusion \
  --epochs 2 --lr .00005 --head-lr .0003 --head-init-std .01 \
  --correct-padding-mask --temporal-residual \
  --train-ids "$SPLIT/train.txt" --val-ids "$SPLIT/val.txt" \
  --max-frames-per-clip 30
```

fusion 解冻跨模态 transformer、投影与新几何头，冻结大视觉/语言骨干；逐帧 loss 反传，帧间记忆梯度截断。best 按内部验证 loss 选择，latest 会覆盖；需要具体轮次时须保存固定快照。再次训练会受硬件与非确定性算子影响，不保证逐比特复现已交付权重。

## 同版本 ONNX

分享包 `model_official_v2.onnx` 与原审计目录 `04_ONNX` 模型字节不变，已验证：严格加载 1033 keys、epoch 2；原始 TTM 多 phase 与参考实现比较、rank-safe pooling 精确比较、CUDA 与 GridSample 参考路径比较、ONNX checker 标准算子检查、ORT 两个连续 recurrent 步与 Torch 比较均通过。相对/绝对容差为 0.003 / 0.0003，opset 17，batch 1，文本长度 110。此验证证明限定输入与容差内的执行一致性，不证明物理坐标和比赛精度。分享包的 `share_manifest.json` 将模型和 ONNX 哈希绑定；原导出 manifest 本身未包含 checkpoint 哈希，原审计目录另有外层 artifact manifest。

复导出须使用同一权重与配置：

```bash
$PY tools/export_official_v2_onnx.py export \
  --config "$MODEL/portable_config.yaml" --source-project "$MODEL" \
  --checkpoint "$MODEL/model.pth" --data-root "$DATA" \
  --partition test --clip-id 000237 --output-dir "$OUT/onnx" \
  --gpu 0 --seed 444 --opset 17 --rtol .003 --atol .0003

$PY tools/export_official_v2_onnx.py run \
  --model "$MODEL/model_official_v2.onnx" \
  --input-npz "$OUT/onnx/verification_step0_inputs.npz" \
  --output-npz "$OUT/ort_step0.npz" --ort-provider CPUExecutionProvider
```

图输入为 `images,input_ids,attention_mask,memory_tokens,memory_phase,read_noise,write_noise,memory_noise`；输出为 `pred_logits,pred_boxes,pred_3d_dim,pred_depth,next_memory_tokens,next_memory_phase`。段首 memory 为零 `[1,128,256]`、phase 为零 int64 `[1]`，后续帧回送两个 next state。噪声按每段 NumPy RandomState(444)，每帧 read/write/memory 顺序生成；图内缩放分别 .02/.02/.2。具体张量、预处理、tokenizer 与状态协议以导出脚本/manifest 为准。分享包不带可能包含原图信息的 NPZ；上面 `run` 示例使用先执行本地 `export` 后新生成的验证输入。分词、图像预处理、标定解码和数值先验在图外；`run` 子命令只运行保存的单步张量，不能声称已提供完整 ONNX 数据集端到端入口。

## 公开与正式提交待办

本地四类资料不代替平台提交：需当前任务 result ZIP、公共 GitHub/Gitee 代码 URL、≤20GB 的模型网盘链接/提取码且有效≥90天（PyTorch 同附 ONNX），以及≤5页技术报告（方法、创新、实验和真实消融）。模型链接、提取码和公共仓库 URL 已记录；完整源码已公开，首次公开源码提交为 `dbe4a1b701842360da987aa493ea88d6d253f6f6`，后续文档更新以 Git 历史为准。需完成主框架实际采用部分的许可衔接后再确定符合 MIT/Apache-2.0/BSD-3-Clause 要求的整仓发布方式。已核实组件许可归档在 `third_party_licenses/`，它们不自动覆盖整个项目。

报名截止 2026-10-25 来自通知，不等于最终结果提交/榜单冻结时限；最终时限尚待主办方确认。提交后保留版本、时间、回执与实际得分。当前尚无官方评分；这里不宣称模型可以直接定为最终参赛版本。

旧入口说明见 `README.legacy_pipeline.md`，原始基线说明原样保存在 `README.upstream.md`。本暂存版只清理旧路径/旧分割副作用与文档；主训练、v2 推理、几何解码、先验、评估和 ONNX 核心源码与本地实测候选一致。
