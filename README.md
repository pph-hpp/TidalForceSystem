# TidalForceSimulator

## ��Ŀ��Ҫ

TidalForceSimulator ��һ������ CUDA �� OpenGL ʵ�ֵ����峱ϫ����ģ������֧�ֶ���������������Ƭ������ӻ���Ⱦ�������ڿ��кͽ�ѧ��ʾ��

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
- **CUDA ֧��**���ܹ����� 75,80,86��
- **OpenGL**
- **��������**��GLFW, GLEW, GLM���ѽ� DLL ���� `lib/`��ͷ�ļ��� `include/`��
- **Windows ƽ̨**��Visual Studio 2019/2022

## ���ٿ�ʼ

1. ��¡�ֿⲢ���빹��Ŀ¼��

```bash
git clone https://github.com/pph-hpp/TidalForceSystem.git
cd TidalForceSystem
mkdir build && cd build
���� Visual Studio ���������

bash
Copy
Edit
cmake .. -G "Visual Studio 16 2019" -A x64
���벢���У�

ʹ�� VS �� build/TidalForceSimulator.sln��ѡ�����ɡ����ؽ�����

��������� Windows ����������F5�����С�

ע�⣺����ʱ���Զ��� lib/ �е� glfw3.dll��glew32.dll �� Render/glsl/ Ŀ¼���Ƶ���ִ���ļ����Ŀ¼��

����ʾ��
�����󣬳�����һ�����ڲ���ʾģ����������������

W/A/S/D���ƶ��ӽ�

�����ק����ת��ͼ

Space����ͣ/����ģ��

��������
ȱ�� DLL����ȷ�� lib/ �°��� glfw3.dll �� glew32.dll��

��ɫ������ʧ�ܣ���� Render/glsl �Ƿ��������������Ŀ¼��