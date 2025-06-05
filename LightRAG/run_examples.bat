@echo off
chcp 65001
echo 正在激活LightRAG环境...
call conda activate LightRAG

echo 正在执行lightrag_openai_compatible_demo.py...
python examples/lightrag_openai_compatible_demo.py

echo 正在执行graph_visual_with_html.py...
python examples/graph_visual_with_html.py

echo 所有脚本执行完成！
pause 