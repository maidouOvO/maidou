@echo off
setlocal enabledelayedexpansion

echo PS绘本文本处理工具 - 一键安装脚本
echo ====================================
echo.

:: 检查是否以管理员身份运行
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo 请以管理员身份运行此脚本！
    echo 右键点击此脚本，选择"以管理员身份运行"。
    pause
    exit /b
)

:: 设置颜色
color 0A

:: 检查Python是否已安装
echo 正在检查Python...
where python >nul 2>&1
if %errorLevel% neq 0 (
    echo Python未安装，正在下载安装程序...
    
    :: 下载Python安装程序
    powershell -Command "& {Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe' -OutFile 'python-installer.exe'}"
    
    if not exist python-installer.exe (
        echo 下载Python安装程序失败！
        echo 请手动下载并安装Python 3.10或更高版本。
        echo 下载地址: https://www.python.org/downloads/
        pause
        exit /b
    )
    
    echo 正在安装Python...
    start /wait python-installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    
    :: 删除安装程序
    del python-installer.exe
    
    :: 检查安装是否成功
    where python >nul 2>&1
    if %errorLevel% neq 0 (
        echo Python安装失败！
        echo 请手动安装Python 3.10或更高版本。
        pause
        exit /b
    )
    
    echo Python安装成功！
) else (
    echo Python已安装。
)

:: 检查pip是否可用
echo 正在检查pip...
python -m pip --version >nul 2>&1
if %errorLevel% neq 0 (
    echo 正在安装pip...
    python -m ensurepip --upgrade
    
    :: 检查安装是否成功
    python -m pip --version >nul 2>&1
    if %errorLevel% neq 0 (
        echo pip安装失败！
        echo 请手动安装pip。
        pause
        exit /b
    )
    
    echo pip安装成功！
) else (
    echo pip已安装。
)

:: 安装依赖库
echo 正在安装依赖库...
python -m pip install -r requirements.txt

if %errorLevel% neq 0 (
    echo 安装依赖库失败！
    pause
    exit /b
)

echo 依赖库安装成功！

:: 检查是否存在可执行文件
if exist dist\ps_text_remover.exe (
    echo 可执行文件已存在，跳过构建步骤。
) else (
    :: 生成图标
    echo 正在生成图标...
    python placeholder_icon.py
    
    :: 构建可执行文件
    echo 正在构建可执行文件...
    pyinstaller ps_text_remover.spec
    
    if not exist dist\ps_text_remover.exe (
        echo 构建可执行文件失败！
        pause
        exit /b
    )
    
    echo 可执行文件构建成功！
)

:: 检查NSIS是否已安装
echo 正在检查NSIS...
where makensis >nul 2>&1
if %errorLevel% neq 0 (
    echo NSIS未安装，正在下载安装程序...
    
    :: 下载NSIS安装程序
    powershell -Command "& {Invoke-WebRequest -Uri 'https://sourceforge.net/projects/nsis/files/NSIS%203/3.08/nsis-3.08-setup.exe/download' -OutFile 'nsis-installer.exe'}"
    
    if not exist nsis-installer.exe (
        echo 下载NSIS安装程序失败！
        echo 请手动下载并安装NSIS。
        echo 下载地址: https://nsis.sourceforge.io/Download
        
        :: 如果NSIS安装失败，仍然可以使用可执行文件
        echo.
        echo 您可以在dist文件夹中找到可执行文件ps_text_remover.exe
        echo 可以直接运行此文件，但不会创建安装包。
        pause
        exit /b
    )
    
    echo 正在安装NSIS...
    start /wait nsis-installer.exe /S
    
    :: 添加NSIS到PATH
    setx PATH "%PATH%;C:\Program Files (x86)\NSIS" /M
    
    :: 删除安装程序
    del nsis-installer.exe
    
    :: 检查安装是否成功
    where makensis >nul 2>&1
    if %errorLevel% neq 0 (
        echo NSIS安装失败或未添加到PATH！
        echo 您可以在dist文件夹中找到可执行文件ps_text_remover.exe
        echo 可以直接运行此文件，但不会创建安装包。
        pause
        exit /b
    )
    
    echo NSIS安装成功！
) else (
    echo NSIS已安装。
)

:: 构建安装包
echo 正在构建安装包...
makensis installer_script.nsi

if not exist "PS绘本文本处理工具_安装包.exe" (
    echo 构建安装包失败！
    echo 您可以在dist文件夹中找到可执行文件ps_text_remover.exe
    echo 可以直接运行此文件。
    pause
    exit /b
)

echo 安装包构建成功！

:: 完成
echo.
echo ====================================
echo 一键安装脚本执行完成！
echo.
echo 您可以找到以下文件：
echo 1. 可执行文件: dist\ps_text_remover.exe
echo 2. 安装包: PS绘本文本处理工具_安装包.exe
echo.
echo 建议使用安装包进行安装。
echo ====================================

pause
