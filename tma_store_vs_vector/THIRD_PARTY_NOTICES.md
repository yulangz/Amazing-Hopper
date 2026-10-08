# 来源与许可

这两个诊断样例提取自 WeftMoE commit `8606637`，按 Apache-2.0 分发，见 [LICENSE](LICENSE)。
`ring_stage_repro.cu` 仅替换本地 header/namespace；`store_repro.cu` 仅替换说明。
`hopper.cuh` 从原 primitive header 提取并删除不使用的适配器、调参宏和 include。

`hopper.cuh` 的 descriptor/address 与 array adapters 源自 DeepGEMM，保留其 MIT 许可如下。
CuTe/CUTLASS 通过外部 headers 使用，没有复制其实现；其许可随 CUTLASS 分发。

MIT License

Copyright (c) 2025 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
