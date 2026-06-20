# sub

这里放按案例拆分的图表生成脚本和可直接复制进论文的 `tex` 表格片段。

## 目录

- `2d/`: 二维算例
- `wing/`: wing weight 10d 算例，替换原来的 5d 算例
- `battery/`: 电池 SoC 算例

## 生成

```bash
python InteractCADLaTeX/sub/2d/make_assets.py
python InteractCADLaTeX/sub/wing/make_assets.py
python InteractCADLaTeX/sub/battery/make_assets.py
```

输出会保存在各自目录下，包含：

- `*.pdf` 和 `*.png` 图
- `tables.tex` 表格片段

