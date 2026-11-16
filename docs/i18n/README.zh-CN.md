# WEX（简体中文简介）

本文是英文 [README](../../README.md) 的简短摘要，日期为 2026-11-15。英文 README 是参考文本：两者不一致时，以英文版为准。本译文并非由母语者撰写，欢迎指正（提交 issue 或 pull request）。

- **是什么。** WEX 是一系列容器格式，用于存放混合专家（Mixture-of-Experts）模型的专家（expert）：一个文件由按 4096 字节对齐的 record 组成，每个 routed expert 一个 record，引擎在生成过程中通过一次读取将其从 NVMe 读入。WEX4 增加了 header page 和索引（index），其中含有每个 record 的 SHA-256；WNR3 是 record 的无损 rANS 编码。WEX4（格式版本 4.0）和 WNR3（v1）已冻结；这里的代码是参考实现，属于研究级别（research-grade），并且优先面向 Windows。
- **目前运行的内容**（2026-11-13 的摘要）。DeepSeek V4.1 REAP-192 读取含 WNR3 record 的 WEX4 文件（大小为未压缩时的 0.854），Qwen3.8-Flash-Next 读取含未压缩 MXFP4 record（RAW）的 WEX4 文件；两者都从两块 NVMe 硬盘读取（U: 和 C:，读取按 38/62 分配）。最新状态见 [STATUS.md](../../STATUS.md)。
- **如何试用。** `bash scripts/check_all.sh` 会编译并运行 CPU 检查（格式测试、WEX4 读取器和转换器、一致性测试向量、快速 WNR3 解码器）以及仓库检查，不需要 GPU，也不需要模型；成功时最后一行是 `all checks passed`，退出码为 0。前提条件见 README 的 [Try it in ten minutes](../../README.md#try-it-in-ten-minutes) 一节。CI 在 Linux（x86-64 和 ARM64）、macOS（Apple silicon，分别使用 GNU gcc 和 Apple clang）和 Windows 上运行它；sanitizer、模糊测试（fuzzing）以及 Go 和 Rust 读取器是单独的 CI 作业。
- **从哪里开始。** 整体结构见 [docs/ARCHITECTURE.md](../ARCHITECTURE.md)；字节布局见 [docs/WEX4-SPEC.md](../WEX4-SPEC.md) 和 [docs/WNR3-SPEC.md](../WNR3-SPEC.md)；仅依据规范编写的 Go 实现见 [impl/go-wex4/](../../impl/go-wex4/README.md)；接入引擎见 [docs/COLIBRI-INTEGRATION.md](../COLIBRI-INTEGRATION.md)；待完成的任务见 [docs/HELP-WANTED.md](../HELP-WANTED.md)；参与贡献见 [CONTRIBUTING.md](../../CONTRIBUTING.md)；报告安全漏洞见 [SECURITY.md](../../SECURITY.md)。
- **许可证。** Apache 2.0（[LICENSE](../../LICENSE)、[NOTICE](../../NOTICE)）。本仓库不包含任何模型权重。`docs/reports/` 中的部分带日期的报告为意大利文。
