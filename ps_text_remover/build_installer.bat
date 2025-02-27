@echo off
echo 正在构建安装包...

echo 检查NSIS是否已安装...
where makensis >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo NSIS未安装，请先安装NSIS后再运行此脚本
    echo 下载地址: https://nsis.sourceforge.io/Download
    pause
    exit /b
)

echo 检查可执行文件是否已构建...
if not exist dist\ps_text_remover.exe (
    echo 可执行文件不存在，请先运行build_exe.bat构建可执行文件
    pause
    exit /b
)

echo 构建安装包...
makensis installer_script.nsi

echo 完成！
echo 安装包已生成: PS绘本文本处理工具_安装包.exe
pause
