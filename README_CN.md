## `run_qwen_image_edit.py` **启动参数文档**

### **一、基础参数**


| **参数**               | **类型** | **必填** | **默认值**                     | **说明**                                                                                                                            |
| -------------------- | ------ | ------ | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `--model-path`       | str    | 否      | `Qwen/Qwen-Image-Edit-2511` | 模型路径，支持 HuggingFace 仓库 ID 或本地目录。可选值：`Qwen/Qwen-Image-Edit`（基础版）、`Qwen/Qwen-Image-Edit-2509`（增强版）、`Qwen/Qwen-Image-Edit-2511`（最新版） |
| `--image-path`       | str    | **是**  | —                           | 待编辑的输入图片路径，支持本地文件路径或 HTTP URL                                                                                                     |
| `--prompt`           | str    | **是**  | —                           | 编辑指令文本，例如 `"Turn it into a watercolor painting."`                                                                                 |
| `--negative-prompt`  | str    | 否      | `None`                      | 负面提示词，用于抑制不希望出现的特征                                                                                                                |
| `--lora-path`        | str    | 否      | `None`                      | LoRA 适配器权重路径，例如 `prithivMLmods/Qwen-Image-Edit-2511-Anime`                                                                        |
| `--output-dir`       | str    | 否      | `outputs/`                  | 生成图片的保存目录，不存在时自动创建                                                                                                                |
| `--output-file-name` | str    | 否      | `None`（自动生成）                | 自定义输出文件名（不含扩展名），不指定则根据 prompt + 时间戳 + 参数哈希自动命名                                                                                    |


### **二、生成参数（Generation parameters）**


| **参数**          | **类型** | **默认值**          | **说明**                                                                         |
| --------------- | ------ | ---------------- | ------------------------------------------------------------------------------ |
| `--height`      | int    | `None`（模型自动决定）   | 输出图片高度（像素），不指定时由模型根据输入图片自动计算                                                   |
| `--width`       | int    | `None`（模型自动决定）   | 输出图片宽度（像素）                                                                     |
| `--steps`       | int    | `None`（模型默认值）    | 去噪步数。`Qwen-Image-Edit` 默认 50 步，`Qwen-Image-Edit-2509/2511` 默认 40 步。步数越多质量越高但越慢 |
| `--cfg-scale`   | float  | `None`（模型默认 4.0） | Classifier-Free Guidance 引导强度。越大越忠实于 prompt，但可能降低多样性                           |
| `--seed`        | int    | `42`             | 随机种子，用于结果复现。相同参数 + 相同种子 = 相同输出                                                 |
| `--num-outputs` | int    | `1`              | 每个 prompt 生成的图片数量                                                              |


### **三、性能参数（Performance options）**


| **参数**                       | **类型** | **默认值** | **说明**                                                                              |
| ---------------------------- | ------ | ------- | ----------------------------------------------------------------------------------- |
| `--num-gpus`                 | int    | `1`     | 使用的 GPU 数量，多卡并行可加速推理                                                                |
| `--text-encoder-cpu-offload` | flag   | `False` | 将文本编码器卸载到 CPU 运行，节省 GPU 显存。适用于显存不足的场景（如 4090 24GB）                                  |
| `--pin-cpu-memory`           | flag   | `False` | 将 CPU 内存设为 pinned memory，加速 CPU 与 GPU 之间的数据传输。建议与 `--text-encoder-cpu-offload` 配合使用 |


### **四、性能分析参数（Profiler options）**


| **参数**                    | **类型** | **默认值**             | **说明**                                                                                       |
| ------------------------- | ------ | ------------------- | -------------------------------------------------------------------------------------------- |
| `--profile`               | flag   | `False`             | 开启 `torch.profiler`，对整个生成过程进行 CPU/CUDA 算子级别的性能追踪                                             |
| `--profile-dir`           | str    | `./profiler_traces` | Profiler trace 文件的保存目录                                                                       |
| `--profile-warmup`        | int    | `0`                 | 正式 profiling 前的预热运行次数。预热不记录 trace，目的是稳定 CUDA 缓存和 JIT 编译，避免首次运行的额外开销影响测量结果                    |
| `--profile-repeat`        | int    | `1`                 | 带 profiling 的重复运行次数。每次运行生成一个独立的 trace 文件（`trace_run0.json`、`trace_run1.json`...），多次运行可观察性能波动 |
| `--profile-with-stack`    | flag   | `False`             | 记录 Python 和 C++ 完整调用栈。会显著增大 trace 文件体积，但可以定位到具体源码行                                           |
| `--profile-memory`        | flag   | `False`             | 追踪 CUDA 显存的分配和释放事件，用于排查显存泄漏或优化显存占用                                                           |
| `--profile-summary`       | flag   | `False`             | 在终端打印算子级别的汇总表格，分别按 CUDA 总耗时、CPU 总耗时排序。开启 `--profile-memory` 时额外输出按显存占用排序的表格                  |
| `--profile-summary-top-n` | int    | `30`                | 汇总表格中显示的 Top N 算子数量                                                                          |


### **五、Profiler 输出产物**

开启 `--profile` 后，会在 `--profile-dir` 目录下生成：


| **文件**                      | **格式**            | **说明**                                                                                |
| --------------------------- | ----------------- | ------------------------------------------------------------------------------------- |
| `trace_run{N}.json`         | Chrome Trace JSON | 可视化 trace 文件，用浏览器打开 `chrome://tracing` 或 [Perfetto UI](https://ui.perfetto.dev/) 加载查看 |
| `trace_run{N}.metrics.json` | JSON              | 结构化的算子耗时数据（仅在 `--profile-summary` 开启时生成），方便程序化分析                                      |


### **六、典型使用场景**

**场景 1：快速生成一张编辑后的图片**

```
python run_qwen_image_edit.py \
    --image-path photo.jpg \
    --prompt "把背景换成海边日落"
```

**场景 2：低显存 GPU 运行**

```
python run_qwen_image_edit.py \
    --image-path photo.jpg \
    --prompt "转换为水彩画风格" \
    --text-encoder-cpu-offload --pin-cpu-memory
```

**场景 3：使用 LoRA 适配器生成动漫风格**

```
python run_qwen_image_edit.py \
    --image-path photo.jpg \
    --prompt "Transform into anime." \
    --lora-path prithivMLmods/Qwen-Image-Edit-2511-Anime
```

**场景 4：快速定位性能瓶颈**

```
python run_qwen_image_edit.py \
    --image-path photo.jpg \
    --prompt "添加雪花效果" \
    --profile --profile-summary
```

**场景 5：完整性能基准测试（预热 + 多次采样 + 显存追踪 + 调用栈）**

```
python run_qwen_image_edit.py \
    --image-path photo.jpg \
    --prompt "添加雪花效果" \
    --profile --profile-warmup 1 --profile-repeat 3 \
    --profile-memory --profile-with-stack --profile-summary
```

