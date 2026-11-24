# 950 双卡 HIXL 激活直传验证包

本包用于在服务器上回答：**建链后创建的 PyTorch NPU tensor，能否直接 WRITE 到远端 HBM，并在释放源 tensor 后 READ 到新 tensor？**

本包是独立实验，不修改 DeepSeek-V4/FSDP2 训练。先回传实验结果，再决定训练端的注册、释放和预取设计。

## 1. 实施顺序与环境

1. 提交本目录到代码仓，在有两张空闲 950 的同一台服务器拉取该提交。
2. 加载服务器实际安装的 CANN 环境；使用与该 CANN 配套的 Python、PyTorch、torch_npu、NumPy 和 HIXL Python 包。
3. 先执行 preflight，再执行 smoke。检查 smoke 结果后执行 stress。
4. 回传每次运行产生的 `artifacts.tar.gz`，失败结果也需要回传。

不需要复制开发机的 `/home/hjy/project/hixl`，也不需要 Mooncake 或完整模型权重。实验调用服务器安装的 HIXL。HIXL 源码分支或提交号不能替代实际加载的二进制版本信息。

下面命令均在 MindSpeed-LLM 仓库根目录执行。`source` 路径和 Python 可执行文件按服务器安装调整：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 tests/tools/fsdp2/hixl_offload_probe/run_probe.py \
  --preflight --devices 0,1 --output /tmp/hixl-probe/preflight-001
```

`--devices` 是**当前进程可见的逻辑设备号**，不是必然等于 `npu-smi` 的物理卡号。本包不修改 `ASCEND_RT_VISIBLE_DEVICES`；请结合环境中的设备映射选卡。两个进程分别使用第一张和第二张卡。

preflight 只检查导入、设备初始化和接口能力，不验证注册或链路。没有 NPU 的本地环境执行时应失败，并输出诊断包，不能记为实机通过。

## 2. 服务器执行命令

### 正确性验证

```bash
python3 tests/tools/fsdp2/hixl_offload_probe/run_probe.py \
  --suite smoke --devices 0,1 --transport default \
  --output /tmp/hixl-probe/smoke-001
```

默认参数：池端 256 MiB；尺寸 4 KiB、128 KiB、2 MiB、16 MiB、64 MiB；每个尺寸 3 轮，无预热；同步每次 1 个请求，异步最多 4 个请求。源、目标及扰动 tensor 额外消耗客户端 HBM，CPU 内容校验也需要内存。

每种注册方式和同步/异步模式使用独立双进程及 HIXL 会话。固定 ACL 同步基线失败后，其余依赖用例跳过；单个动态注册用例失败不阻止其他独立用例。

### 压力与性能采样

```bash
python3 tests/tools/fsdp2/hixl_offload_probe/run_probe.py \
  --suite stress --devices 0,1 --transport default \
  --case-timeout 3600 --output /tmp/hixl-probe/stress-001
```

stress 每尺寸先预热 5 轮，再测量 100 轮；增加池进程退出测试。CPU 正确性校验可能使整组实验较慢，因此不要把总运行时间当作通信耗时。

调试时可缩小测试规模，仍会验证所有注册方式：

```bash
python3 tests/tools/fsdp2/hixl_offload_probe/run_probe.py \
  --suite smoke --devices 0,1 --sizes 4096,2097152 --loops 2 \
  --inflight 2 --pool-mib 16 --timeout 30 --case-timeout 300 \
  --output /tmp/hixl-probe/debug-001
```

输出目录必须为空，每次使用新目录。每请求/控制消息默认超时 30 秒，每组双进程默认硬截止时间 1800 秒。硬截止会终止本包拥有的工作进程，避免错误路径无限等待。不会停止其他服务或重置整张 NPU。

退出码：`0` 表示本次要求的用例通过；`1` 表示环境/用例失败；`2` 表示有显式不支持或跳过的用例；`130` 表示中断。地址复用等观察性覆盖不足单独写入报告，不改变退出码。结果中个别成功传输不能替代整组用例的清理成功。

### 传输配置

首轮保持 `--transport default`，不把 950 的默认协议称为 HCCS。`--host` 默认使用单机回环地址；若运行环境要求实际管理网 IP，可显式传入本机 IPv4 地址。HIXL endpoint 端口由启动器选取空闲 TCP 端口；极小概率被其他进程抢占时，保留失败日志并换新目录重跑。

支持读取以下环境变量并转换成 HIXL 初始化 option：

| 环境变量 | HIXL option |
|---|---|
| `ASCEND_LOCAL_COMM_RES` | `LocalCommRes` |
| `ASCEND_GLOBAL_RESOURCE_CONFIG` | `GlobalResourceConfig` |
| `ASCEND_BUFFER_POOL` | `BufferPool` |

也可通过 `--options-json /path/to/options.json` 提供字符串到字符串的 JSON 对象，覆盖上述映射。显式 `--transport` 最后覆盖 `GlobalResourceConfig` 中的 `comm_resource_config.protocol_desc`，保留其他字段。`AutoConnect=1` 被拒绝，防止掩盖建链前后注册差异。

为了隔离版本/后端影响，可以在首轮结果回传后，另建一轮使用服务器确认支持的 CS 配置，例如：

```json
{
  "LocalCommRes": "{\"version\":\"1.3\"}",
  "BufferPool": "0:0"
}
```

这不是首轮必需配置，也不是所有安装版本都支持。库可能读取其他选路环境变量（如 `HCCL_INTRA_ROCE_ENABLE`），因此必须结合日志判断实际路径。`ASCEND_ENABLE_USE_FABRIC_MEM` 仅记录，不由本包自动翻译成 HIXL option；需要时应明确在 options JSON 中配置。

## 3. 实验和内存生命周期

| 用例 | 本地地址 | 注册时机 | 主要目的 |
|---|---|---|---|
| `acl_fixed` | ACL HUGE_ONLY 固定内存 | 建链前，普通注册 | 基础设备注册和链路 |
| `torch_fixed` | 固定 PyTorch tensor | 建链前，普通注册 | PyTorch 地址兼容性 |
| `local_only_fixed` | 固定 PyTorch tensor | 建链前，local_only | 本地专用注册能力 |
| `torch_dynamic` | 每轮新建 tensor | 建链后，普通注册 | 普通动态注册支持范围 |
| `local_only_dynamic` | 每轮新建 tensor | 建链后，local_only | 激活原地址直传候选路径 |
| `pool_exit` | 动态 tensor | 建链后，local_only | 池端退出后的失败/超时返回 |

动态用例按以下顺序执行：

1. 创建源 tensor，注册其原始 `data_ptr()`；填入按迭代和 slot 区分的 PCG64 伪随机字节。
2. 用 NPU event 等待生产完成，再 WRITE 到远端 slot。
3. 同步返回或异步查询确认完成后，由池端独立读取内容并校验 SHA-256。
4. 注销并释放源 tensor，记录 allocated/reserved 及注册数。
5. 新建扰动 tensor 并写入不同内容，增加暴露旧地址误用的机会。
6. 新建并注册回载目标，先写入污染值，再 READ；完成后在客户端验证 SHA-256。
7. 注销目标并释放本轮数据，确认没有活跃注册或未完成请求。

固定用例在整个会话中复用源、目标地址。每个异步 group 最多持有 `--inflight` 个请求，每个请求对应一个不重叠的远端 slot。第一版没有为了吞吐合并大包，不包含模型前反向或计算通信重叠。

使用 uint8 连续 tensor 验证字节级传输，不宣称已验证 autograd、BF16 计算或共享 storage 的行为。不得将示例里的地址对齐方式用于任意激活：本包不会将 PyTorch 的 `data_ptr()` 向上取整并跳过 tensor 内容。

**失败时的释放策略：** 未确认完成的 DMA 不能通过普通 Python 析构提前释放内存。worker 写入诊断后直接结束隔离进程，父进程先停止客户端再停止池端；不在失败会话中复用地址、不重新连接并继续传输。正常退出严格先断开客户端、注销本地内存，再通知池端清理。

`pool_exit` 仅在动态 local_only 异步用例通过后执行。父进程终止自己创建的池端后，客户端发起 READ；明确失败或应用侧超时可通过，若返回成功则保留为失败结果，不能据此认定远端已安全释放。若原生调用一直不返回导致父进程硬超时，该用例为 FAIL。

## 4. 输出与回传

```text
output/
  environment.json          # 版本、模块位置、加载库、设备、白名单环境变量
  config.json               # 实际选用的实验参数和初始化 option
  results.jsonl             # 全部用例与迭代结果
  summary.md                # 状态、传输耗时、动态显存观察及限制
  client.log / pool.log     # Python 和原生 stdout/stderr
  preflight_device_*.json
  cases/<case>_<mode>/      # 每个隔离用例的配置、原始结果和日志
  manifest.json            # 回传文件的 SHA-256
  artifacts.tar.gz
```

`results.jsonl` 包含注册方式/时机、字节数、迭代、warmup 标记、原始返回码、操作耗时、WRITE/READ 校验、地址复用、显存和活跃资源数量。

- `PASS`：该阶段实际成功。完整用例还要求正常清理成功，或符合池退出测试的预期错误。
- `FAIL`：运行错误、校验不一致、原生崩溃或硬截止。
- `UNSUPPORTED`：接口缺失或 HIXL 明确返回 `UNSUPPORTED`；不将任意注册错误解释为不支持。
- `SKIP`：前置条件未满足。

通信 `complete_ms` 不含 CPU 数据生成、远端校验和哈希。同步 `submit_ms` 已包含阻塞等待；异步另外记录 `wait_ms`，其结果包含 Host 查询间隔（0.5 ms）的影响，不能当作纯链路延迟。`end_to_end_ms` 包含整轮控制和验证开销。注册、注销、初始化和建链耗时保存在 `operations` 中。

PyTorch reserved 可能因缓存不下降；需要看 allocated 变化、活跃句柄与长期趋势。ACL 池内存不计入 PyTorch allocator 统计，池大小单独记录。未实际观察到地址复用时，报告为覆盖不足，不编造通过结论。

本包设置 `application_staging=false`：传输提交直接使用业务地址，不增加激活中转。数据初始化和校验确有 Host/Device 拷贝，放在通信计时外。**不据此断言库内部零中转。** 实际协议、完成通知路径与 Host 异常回调缺少可靠证据时保持 `unknown`。

关于 HIXL `d83c2dd3`：新增的 device completion flag 是内部控制信息；ValueWait 是否启用还受机型、协议、HCOMM 和 Host 回调影响。公开 API 仍通过 Host 查询获取完成状态。本包不读取或等待私有 slot flag。

若需要补充原生日志，可在新一轮运行前配置服务器支持的 CANN 日志选项，例如 `ASCEND_SLOG_PRINT_TO_STDOUT=1`。日志等级会影响耗时，诊断轮与性能轮应分开。不要把日志中仅出现的 `ValueWait` 名称或安装版本当作路径已启用的证据。

回传 `artifacts.tar.gz`；环境检查失败时也会打包。没有自动收集仓库、全量环境变量或外部 CANN 日志目录。若运行器被 SIGKILL，最后打包可能未执行，可重新生成：

```bash
python3 tests/tools/fsdp2/hixl_offload_probe/report.py /tmp/hixl-probe/smoke-001
```

## 5. 本地检查与后续决策

无需 NPU 的检查：

```bash
python3 -m unittest discover -s tests/tools/fsdp2/hixl_offload_probe -p 'test_probe.py' -v
python3 tests/tools/fsdp2/hixl_offload_probe/run_probe.py --help
```

mock 测试验证请求只消费一次、在途请求禁止注销、超时保留资源所有权、断链失败不释放、句柄复用、CLI 参数和归档输出；不能替代板上验证。

收到结果后按以下规则决定训练设计：

1. 动态直接路径正确、allocated 可回收：进入 DeepSeek-V4 主干重计算输入的接入。
2. 只有固定注册地址可用：设计受控注册分配区域，验证激活能否直接产生在其中及其显存收益。
3. 直接路径不可用：先排查版本、选路、注册和分配约束，再评估中转对照方案。
4. 正确但协议/完成路径未知：先补诊断，不能宣称零拷贝或 ValueWait 性能收益。
