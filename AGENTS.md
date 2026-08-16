# MLEvolve agent instructions

本文档适用于仓库根目录及其全部子目录。执行任务前先阅读本文件；用户在当前任务中的明确要求优先级更高。

## 仓库与 Git 约定

- 仓库根目录：`E:\MLEvolve`。
- 默认工作分支：`main`。开始修改前确认 `git status --short --branch`，正常情况下直接在 `main` 上工作；除非用户明确要求，不创建功能分支或 PR。
- 默认推送目标仅为 `origin/main`，其远端是 `git@github.com:violetljj/MLEvolve.git`。`upstream`（`InternScience/MLEvolve`）只用于读取和同步上游，禁止向它推送。
- 修改前先检查工作树。保留用户或其他任务已有的改动，只编辑、暂存本任务拥有的文件或代码块；不要为了得到干净状态而回滚、覆盖或删除无关内容。
- 完成实现后做与改动范围相称的验证，然后自行执行暂存、提交和推送，不需要等待用户再次说“提交推送”。提交信息应简洁说明结果，不创建空提交。
- 推送前至少执行 `git diff --cached --check`，审阅 `git diff --cached --stat` 和暂存内容。推送后执行 `git rev-list --left-right --count HEAD...origin/main`，预期为 `0 0`。
- 如遇未明确归属的工作区改动、当前分支与 `origin/main` 分叉、推送会携带本任务之外的既有提交、凭据不可用或非快进推送，停止并向用户报告，不要强制推送或自行合并。

## 本机环境（2026-08-17 快照）

- 操作系统：Windows 11 家庭中文版，64 位；PowerShell 7.6.4。命令示例默认使用 PowerShell。
- CPU：Intel Core Ultra 7 251HX，18 个逻辑处理器；内存约 15.4 GiB。
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU，约 8 GiB 显存；驱动 610.88。
- 可用磁盘空间会变化；本快照为 `C:` 约 203.6 GiB、`D:` 约 415.8 GiB、`E:` 约 93.4 GiB、`F:` 约 375.8 GiB。大型数据集、模型和长时间运行产物开始前必须重新检查空间，避免写入 Git。
- 当前没有可用的 WSL/Bash。`run_single_task.sh` 和 `launch_server.sh` 不能直接作为本机默认入口；优先从 PowerShell 调用 `python run.py ...`，只有在另行确认 Bash 环境后才使用 shell 脚本。
- `config/config.yaml` 默认 `cpu_number: 21`，高于本机 18 个逻辑处理器。真实运行前根据并行负载调低，且不要把仅适用于本机的实验路径或密钥提交进仓库。
- GPU 只有约 8 GiB 显存。启用 `agent.memory_embedding_device: cuda`、多分支搜索或参赛模型训练前先评估显存；必要时将嵌入模型切到 `cpu`、降低并行度或使用外部算力。

## 本机工具与依赖位置

| 工具 | 当前版本/位置 | 使用说明 |
| --- | --- | --- |
| Git | 2.55.0；`D:\Git\cmd\git.exe` | PATH 中可直接调用 `git` |
| Python | 默认 3.14.5；`E:\codex-tools\bin\python.cmd` | 项目依赖较重，优先用独立 Python 3.11 虚拟环境，不要污染全局环境 |
| Python 3.11 | `E:\codex-tools\uv-python\cpython-3.11.9-windows-x86_64-none\python.exe` | 推荐作为本项目 Windows 环境的基础解释器 |
| uv | 0.11.15；`E:\codex-tools\bin\uv.cmd` | 用于创建和维护隔离环境 |
| Node.js | 24.16.0；`E:\codex-tools\tools\nodejs\node.exe` | Codex CLI 的稳定启动器 |
| npm | 11.13.0；`E:\codex-tools\bin\npm.cmd` | 全局包根位于 `E:\codex-tools\tools\node-global` |
| NVIDIA 工具 | `C:\Windows\system32\nvidia-smi.exe` | 长运行前检查显存、驱动和进程占用 |
| Docker 包装器 | `E:\codex-tools\bin\docker.cmd` | 使用前先确认 Docker Desktop/daemon 已启动 |

推荐用仓库内 `.venv` 隔离依赖：

```powershell
Set-Location E:\MLEvolve
& E:\codex-tools\bin\uv.cmd venv --python E:\codex-tools\uv-python\cpython-3.11.9-windows-x86_64-none\python.exe .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --no-deps -r requirements_base.txt
python -m pip install --no-deps -r requirements_ml.txt
python -m pip install --no-deps -r requirements_domain.txt
```

三个 requirements 文件含大量固定版本及部分偏 Linux/CUDA 的包。不要把“一次完整安装成功”当作理所当然；先按任务所需安装最小依赖，若完整安装失败，记录具体包、平台和错误，不要随意改版本掩盖兼容性问题。

## Codex CLI 调用

命令选项以 [OpenAI Codex CLI 官方参考](https://developers.openai.com/codex/cli/reference/) 为准；本节记录的是这台机器已经验证过的启动路径。

不要在自动化或 Python 子进程中直接执行 PATH 中的 `codex`。它当前解析到 WindowsApps：

`C:\Program Files\WindowsApps\OpenAI.Codex_26.810.7004.0_x64__2p2nqsd0c76g0\app\resources\codex.exe`

该路径在本机子进程中会报“拒绝访问”/WinError 5。应显式通过独立 Node.js 启动已安装的 Codex CLI：

```powershell
$CodexNode = 'E:\codex-tools\tools\nodejs\node.exe'
$CodexCli = 'E:\codex-tools\tools\node-global\node_modules\@openai\codex\bin\codex.js'

# 每次用于实验或自动化前先确认版本；当前已验证为 codex-cli 0.147.0。
& $CodexNode $CodexCli --version

# 非交互式调用示例。
& $CodexNode $CodexCli exec `
  --cd E:\MLEvolve `
  --sandbox workspace-write `
  '检查当前仓库并完成指定任务'
```

需要结构化输出时使用 `codex exec --output-schema <schema.json>`；需要事件流时使用 `--json`。不要默认使用 `--dangerously-bypass-approvals-and-sandbox`，权限必须与任务风险相称。

MLEvolve 的 Codex 后端要求把命令编码为 JSON 字符串数组：

```powershell
$env:MLEVOLVE_CODEX_COMMAND = @(
  'E:\codex-tools\tools\nodejs\node.exe',
  'E:\codex-tools\tools\node-global\node_modules\@openai\codex\bin\codex.js'
) | ConvertTo-Json -Compress
$env:MLEVOLVE_CODEX_REASONING_EFFORT = 'medium'

python run.py `
  agent.code.model=codex:gpt-5.6-sol `
  agent.feedback.model=codex:gpt-5.6-sol `
  '<其他 MLEvolve/Hydra 参数>'
```

该适配器会自行追加 `exec`、只读沙箱和结构化输出参数；不要把 `exec` 写进 `MLEVOLVE_CODEX_COMMAND`。认证由 Codex CLI 的本机登录状态处理，无需为此设置 API key。正式长实验前先做一次短的 CLI 和认证冒烟测试，并把实际 CLI 版本记入实验记录。

## 项目运行与验证

- 主配置是 `config/config.yaml`；数据集目录、模型缓存、API 地址和密钥应通过本地配置或命令行覆盖，不得提交真实密钥、私人数据路径或竞赛私有数据。
- `runs/`、`dataset/`、`mle-bench/`、模型权重和日志已被忽略。新增大文件前仍要检查 `git status`，避免误提交未覆盖的产物格式。
- 小型文档或配置修改至少执行 `git diff --check` 并人工核对相关内容。
- Python 局部修改优先执行对应模块的最小导入、定向测试或编译检查；只有跨模块或公共基础设施改动才扩大验证范围。
- 长时间 MLEvolve 运行不是普通代码修改的默认验收项。只有用户要求真实实验，或改动涉及搜索/执行/评估主链且短测不足以证明正确性时才启动；开始前确认数据、时间预算、磁盘、CPU/GPU 和终止策略。
- 不把成功启动、单个 seed、短冒烟或生成候选误报成算法效果、榜单提升或可复现实验结论。结论强度必须受实际证据约束。
