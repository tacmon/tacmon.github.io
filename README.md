# tacmon 的学习笔记

博客地址：https://tacmon.github.io/

第一篇文章：[题解：让小车学会送货——从同一道题理解 PPO 与 SAC](https://tacmon.github.io/posts/sac-ppo/)。

## 更新文章

1. 编辑 `content/sac-ppo.md`。
2. 安装构建依赖：`python -m pip install -r requirements.txt`。
3. 执行 `python scripts/build.py`。
4. 将 Markdown 和生成的 HTML 一起提交到 `main` 分支。

GitHub Pages 从 `main` 分支根目录发布。`.nojekyll` 保留纯静态文件；无需服务器。KaTeX 资源在 `assets/katex/` 本地托管，其许可证随资源保留。

本地预览：`python -m http.server 8765`，然后访问 `http://localhost:8765/`。

样例核验：`python examples/delivery_checks.py`。此脚本检查题目环境及文章数值，不运行 PPO/SAC 训练。
