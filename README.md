# 刘宇 Yu Liu · 个人主页

网站地址：https://aquareo.github.io/

中文版：https://aquareo.github.io/

English: https://aquareo.github.io/en/

两版为独立静态页面，通过顶部语言链接切换，共用 `dist/styles.css`。维护时请同步更新 `dist/index.html` 和 `dist/en/index.html`。

教育经历中的成绩摘录自本人提供的官方成绩单：本科展示专业均分与总均分的官方百分制折算值；硕士展示原始 4 分制 GPA。课程表可展开查看。站点不包含成绩单原件或个人陈述，公开简历以 `documents/academic-cv.html` 为可编辑源文件，研究范围按完整硕士论文整理，预印本引用按 arXiv 当前书目记录核对。

个人介绍、工作与教学经历、教育背景、科研论文和项目作品位于 `dist/`。GitHub Pages 使用 GitHub Actions 发布该目录，无需前端构建步骤。

- `.github/workflows/deploy-pages.yml`：个人网站更新后自动部署。
- `dist/bonds/index.html`：原可转债排名页面，保留在 `/bonds/`。
- `.github/workflows/python_update.yml`：保留原定时更新计划，更新排名后直接发布网站。
- `update_website.py`：输出已迁移至 `dist/bonds/index.html`，避免覆盖个人主页。

本地查看：`python3 -m http.server 8000 --directory dist`。

页面采用单栏文字排版，依次为简介、科研、教育、教学、工作及补充项目。主研究项目覆盖完整硕士论文；定价预印本作为相关成果列出，排名脚本作为补充工程项目列出。没有前端依赖或 JavaScript。设计参考 [Jon Barron](https://jonbarron.info/) 与 [Andrej Karpathy](https://karpathy.ai/) 的个人主页，页面内容按本人资料整理。

## 原可转债项目说明

# 项目说明

## 主要功能

### 1. 创建可转债排序网站
- 使用 **GitHub Actions** 自动运行 Python 脚本。
- 利用 **GitHub Pages** 托管静态网页。
- 通过 Python 脚本，每天自动更新网页内容，基于某种策略实现自动化网站更新。

**项目链接**: [Aquareo's Convertible Bonds Sorting](https://aquareo.github.io/bonds/)

### 2. 实现日内交易模拟
- 根据 **移动平均交叉策略** 进行模拟交易，也支持自定义策略编写。

## 程序说明

- **update_website.py**：更新网站内容的脚本，每天运行并自动刷新网页数据。
- **day_trading.py**：日内自动交易模拟脚本，使用内置的移动平均交叉策略进行交易模拟，并支持用户自定义策略。

