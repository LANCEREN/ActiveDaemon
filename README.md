# ActiveDaemon (model lock🔒)

通过数据隐写授权（StegaStamp）与数据投毒训练实现神经网络模型的主动授权保护：
只有携带正确隐写水印的"授权数据"才能获得正常推理精度，未授权数据的输出被锁定为无效分布。

## 环境要求

**本项目锁定在 Python 3.8 + CUDA 11.3 运行**，不要使用其他 Python 版本，原因是依赖版本的硬性交集：

| 依赖 | 版本 | Python 限制 |
|---|---|---|
| torch | 1.10.0 | 仅提供 ≤ 3.9 的 wheel |
| numpy | 1.22.3 | 要求 ≥ 3.8 |
| opencv-python | 4.1.2.30 | 仅提供 ≤ 3.8 的 wheel |

三者交集 = **只有 Python 3.8 可以装齐全部依赖**。

GPU 方面：requirements.txt 中的 torch 已指向 CUDA 11.3 构建（`+cu113`），支持到 Ampere
架构（RTX 30 系 / A100）。**RTX 40 系（sm_89）和 H100（sm_90）无法被 torch 1.10 支持**，
必须按下文迁移方案升级。

### 安装

```bash
conda create -n activedaemon python=3.8 -y
conda activate activedaemon
pip install -r requirements.txt
# StegaStamp 水印 token 生成子模块有独立依赖：
pip install -r stegestamp_tokens_generation/requirements.txt
```

### 训练与测试

```bash
# 多卡 DDP 训练（按 nvidia-smi 自动选卡，--ngpu 指定卡数）
python ./playground/poison_exp.py --experiment=poison --type=cifar10 --ngpu=3 \
    --epochs=100 --lr=0.001 --poison_flag --trigger_id=15 --poison_ratio=0.5 \
    --rand_loc=1 --rand_target=1

# 测试
python ./tests/poison_exp_test.py --pre_experiment=poison --pre_type=cifar10
```

更多示例见 `scripts/` 目录。注意：

- 数据根目录默认指向 `--data_root=/mnt/data03/renge/public_dataset/image/`，请按实际环境传参。
  MNIST/FMNIST/CIFAR/SVHN 可自动下载，GTSRB 用 `scripts/download_gtsrb_dataset.sh`，
  ImageNet 与 StegaStamp 隐写数据集需自行准备。
- `trigger_id` 10–29 需要 `<data_root>/triggers/trigger_*.png` 触发器素材。
- 显存紧张时可加 `--empty_cache_interval=N`（每 N 个 batch 清一次 CUDA 缓存）；
  默认 0 表示仅在 epoch 间清理，以保证多卡训练吞吐。

## 依赖四件套升级技术方案（torch / torchvision / numpy / opencv）

> 当前代码**保持在 torch 1.10 / Python 3.8 上运行，暂不升级**。
> 以下方案供将来迁移到新硬件（RTX 40 系 / H100）或新 Python 时执行。

### 目标版本组合（建议）

| 依赖 | 当前 | 目标 | 说明 |
|---|---|---|---|
| Python | 3.8 | 3.10 / 3.11 | torch 2.1 支持 3.8–3.11 |
| torch | 1.10.0+cu113 | 2.1.x（cu118/cu121） | cu118 起支持 sm_89/sm_90 |
| torchvision | 0.11.1+cu113 | 0.16.x | 与 torch 2.1 配套 |
| numpy | 1.22.3 | 1.24+ | 见下方 API 变更 |
| opencv-python | 4.1.2.30 | 4.8+ | API 基本兼容 |

### 升级时需要修改的代码点

1. **`np.bool` 别名移除**（numpy ≥ 1.24 直接报错）
   - `playground/bubble_exp.py:171` 的 `np.bool` → 内置 `bool` 或 `np.bool_`。
   - `np.fromstring`/`.tostring()` 已在当前分支修复为 `frombuffer`/`tobytes`，无需再改。
2. **`torch.meshgrid` 缺少 `indexing` 参数**（torch ≥ 1.10 警告，2.x 仍默认 'ij' 但应显式）
   - `utee/utility.py:440` 改为 `torch.meshgrid(array1d_x, array1d_y, indexing='ij')`，行为不变。
3. **`torch.utils.model_zoo` 弃用**
   - `NNmodels/model.py`、`utee/misc.py:260` 的 `model_zoo.load_url` → `torch.hub.load_state_dict_from_url`，签名兼容。
4. **`torch.autograd.Variable` 弃用**
   - `utee/misc.py:239` 的 `Variable(torch.FloatTensor(data))` → 直接用 tensor。
5. **`torch.load` 默认行为变更**（torch ≥ 2.6 默认 `weights_only=True`）
   - 本项目部分 checkpoint 用 `torch.save(model)` 保存整个模型对象，
     加载处（`NNmodels/model.py` 的 `_load_pretrained`、各 test 脚本）需显式传 `weights_only=False`，
     或统一改为只存取 `state_dict`（推荐，`misc.model_snapshot` 默认就是 state_dict）。
6. **AMP 写法**：升级后引入混合精度训练时用 `torch.amp.autocast('cuda')`（2.x 推荐），
   对本项目 KLDivLoss 的小 loss 值需配合 `GradScaler` 验证数值稳定性。
7. **DataLoader 增强**（torch ≥ 1.11 可用）：`persistent_workers=True` 避免每个 epoch 重建 worker；
   `DistributedSampler(..., drop_last=True)` 消除验证集 padding 导致的精度偏差。
8. **torchvision 内置 GTSRB**（≥ 0.12）：`datasets.GTSRB` 可替换 `dataset/clean_image_dataset.py` 中的自定义实现（可选）。
9. **配套依赖**：升级 torch 后需同步放开 `kornia`（≥ 0.7）、`torchmetrics` 等的版本钉死；
   `tensorboard==2.0.0` 升级到 2.x 最新即可，SummaryWriter 调用方式不变。

### 建议的迁移步骤

1. 新建 Python 3.10 环境，安装目标版本四件套，`pip check` 确认依赖树无冲突。
2. 按上面 1–5 修改代码（均为机械替换，互不耦合，可逐条提交）。
3. 用 CIFAR-10 小规模任务跑通 `poison_exp.py` 单卡 → 多卡 DDP，对照旧环境验证：
   授权/非授权精度曲线趋势一致、checkpoint 可双向加载。
4. 回归 `tests/` 下的各攻击测试（fine_tune / prune / neural_cleanse / grad_cam）。
5. 全部通过后更新 requirements.txt 并替换本节版本表。
