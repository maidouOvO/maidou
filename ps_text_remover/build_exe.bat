@echo off
echo 正在安装依赖...
pip install -r requirements.txt

echo 正在创建可执行文件...
pyinstaller ps_text_remover.spec

echo 完成！
echo 可执行文件位于 dist 文件夹中
pause
