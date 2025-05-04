# TidalForceSimulator

## 项目概要

TidalForceSimulator 是一个基于 CUDA 和 OpenGL 实现的天体潮汐破碎模拟器，支持多体引力、刚体碎片化与可视化渲染，可用于科研和教学演示。

## 目录结构
```plaintext
TidalForceSimulator
├── CMakeLists.txt     # 构建脚本
├── include/           # 头文件目录
├── src/               # 源代码目录（.cpp, .cu）
├── Render/            # 渲染相关代码和着色器
│   └── glsl/          # GLSL 着色器文件
├── lib/               # 第三方库 (.lib/.dll)
│   ├── glfw3.dll
│   ├── glew32.dll
│   └── ...
└── build/             # 构建输出目录（由 CMake 生成）

```

## 依赖环境

- **CMake** ≥ 3.18
- **C++14**
- **CUDA 支持**（架构例如 75,80,86）
- **OpenGL**
- **第三方库**：GLFW, GLEW, GLM（已将 DLL 放在 `lib/`，头文件在 `include/`）
- **Windows 平台**：Visual Studio 2019/2022

## 快速开始

1. 克隆仓库并进入构建目录：

```bash
git clone https://github.com/pph-hpp/TidalForceSystem.git
cd TidalForceSystem
mkdir build && cd build
生成 Visual Studio 解决方案：

bash
Copy
Edit
cmake .. -G "Visual Studio 16 2019" -A x64
编译并运行：

使用 VS 打开 build/TidalForceSimulator.sln，选择“生成”或“重建”。

点击“本地 Windows 调试器”（F5）运行。

注意：构建时会自动将 lib/ 中的 glfw3.dll、glew32.dll 和 Render/glsl/ 目录复制到可执行文件输出目录。

运行示例
启动后，程序会打开一个窗口并显示模拟结果，按键操作：

W/A/S/D：移动视角

鼠标拖拽：旋转视图

Space：暂停/继续模拟

常见问题
缺少 DLL：请确认 lib/ 下包含 glfw3.dll 和 glew32.dll。

着色器加载失败：检查 Render/glsl 是否完整复制至输出目录。