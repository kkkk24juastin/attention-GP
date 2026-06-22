# AGENTS.md

本文件记录本项目内后续协作时必须遵守的本地约束。除非用户明确更新，本文件优先作为项目级操作规范。

## 通用沟通与检索

- 始终使用简体中文与用户沟通，技术名词可保留英文。
- 在回答具体问题或编辑文件前，必须先检索并阅读相关文件，优先使用 `rg` / `rg --files` 获取当前磁盘状态，不要凭记忆判断。
- 若当前环境提供 ace-tool 的 `search_context`，应优先用英文向该工具检索项目上下文；若不可用，需说明并用本地检索替代。
- 需要广域网搜索时，不使用模型自带联网搜索；优先使用 `mcp__exa`。涉及第三方库/API/框架文档时，优先使用 `mcp__context7` 获取官方资料。
- 若需求、数据口径或章节范围不确定，必须先询问用户，不要自行决定。

## 安全与文件操作

- 不要回退或覆盖用户已有改动，除非用户明确要求。
- 不执行破坏性命令，例如 `git reset --hard`、`git checkout --`、`rm -rf` 等，除非用户明确授权。
- 安装、更新、卸载依赖或修改系统/环境配置前，必须先列出命令或 diff 并等待用户同意。
- 手工编辑文件时使用 `apply_patch`，不要用 shell 重定向或脚本随意覆盖源文件。

## 论文修订专用约束

- 默认只修改中文稿：`InteractCADLaTeX/interactcadsample_cn.tex`。不要修改英文稿，除非用户明确要求。
- 用户已备份时，可以直接修改中文稿并使用 `InteractCADLaTeX/build_latex.sh` 编译；编译后检查生成的中文 PDF。
- 所有可见论文修改必须标红并带回应编号，沿用：
  `\reviewadd{修改内容}{AE-1, R1-5, R2-6}`。
- 表格新增或修改的可见表题、表头、表体和正文解释，均应保持红色回应格式；图注和子图注也要标注回应编号。
- 语言风格必须与论文一致：严谨、简洁、避免口语化和夸大表达。
- 图表样式应尽量与英文原稿一致；表格使用 booktabs 风格，避免脚本变量名出现在论文中。
- 图内部不要额外添加图名/title；只保留必要轴标签。轴标签应使用论文术语，例如 `RMSE in the target region`、`RMSE over the full surface`、`True quality loss`，不要使用 `rmse_all`、`ga_true_quality_loss` 等代码列名。
- 需要重新生成图表时，先修改对应脚本，再运行脚本生成 PDF/PNG，并重新编译论文。
- 不清理未引用的生成文件，除非用户明确要求。

## 当前中文稿章节定位

- 4.1：二维数值算例及新增消融、敏感性分析。
- 4.2：锂电池荷电状态预测。
- 4.3：航空机翼重量工程案例（Wing Weight 10D）。
- 4.4：微型火箭稳健参数设计。
- 不要把 4.2 或 4.3 误认为微型火箭；微型火箭是 4.4。

## 当前图表与编译注意事项

- 4.3 Wing Weight 图表脚本位于 `InteractCADLaTeX/sub/wing/make_assets.py`。
- 4.3 Wing Weight 主结果图使用：
  - `InteractCADLaTeX/sub/wing/wing_rmse_target.pdf`
  - `InteractCADLaTeX/sub/wing/wing_true_quality_loss.pdf`
- 4.3 Wing Weight 当前结果口径：初始 LHS 训练点 80 个，新增设计点 30 个，总训练样本数 110；目标值 300 lb，目标区域 `|y-300|\le20` lb；每种方法执行 20 次 GA 优化评价。
- 编译命令：
  `cd InteractCADLaTeX && ./build_latex.sh`
- 编译成功后，中文 PDF 输出为：
  `InteractCADLaTeX/build/interactcadsample_cn.pdf`
