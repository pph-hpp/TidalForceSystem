# TidalForceSimulator

## ��Ŀ��Ҫ

����Ŀ���� CUDA �� OpenGL��ʵ���������ڳ�ϫ���������𲽽��岢�γ��ǻ����������ģ�⡣ϵͳ֧�ֶ����������㡢������Ƭ��ģ���Լ�ʵʱ���ӻ���Ⱦ��

���������У������������ž����ƽ������˥����������һ����������崦��ǿ����Դ����ʱ���俿����Զ������Դ���������ܵ�������С��ͬ��������ϫ�����ó�ϫ�����������˺�����ã�һ��������ṹ���ޣ���ᵼ��������壬��Ƭ�ڹ�������·ֲ������տ����γ������ǻ��Ľṹ��

<img src="./img/tidal.png" width="50%" style="display: block; margin: auto;">

## Ŀ¼�ṹ
```plaintext
TidalForceSimulator
������ CMakeLists.txt     # �����ű�
������ include/           # ͷ�ļ�Ŀ¼
������ src/               # Դ����Ŀ¼��.cpp, .cu��
������ Render/            # ��Ⱦ��ش������ɫ��
��   ������ glsl/          # GLSL ��ɫ���ļ�
������ lib/               # �������� (.lib/.dll)
��   ������ glfw3.dll
��   ������ glew32.dll
��   ������ ...
������ build/             # �������Ŀ¼���� CMake ���ɣ�

```

## ��������

- **CMake** �� 3.18
- **C++14**
- **CUDA**
- **OpenGL  (GLFW, GLEW, GLM)**

## ������Ŀ��
- **CMake����**
```bash
git clone https://github.com/pph-hpp/TidalForceSystem.git
mkdir build
cd build
cmake ..

```
## ����ʾ��
W/A/S/D���ƶ��ӽ�

�����ק����ת��ͼ

- **�����𽥽����γ��ǻ�**
<img src="./img/disintegration.gif" width="50%" style="display: block; margin: auto;">

- **�����ӹ����Ǹ���**
<img src="./img/Grazing.gif" width="50%" style="display: block; margin: auto;">