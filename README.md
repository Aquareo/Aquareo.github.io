# 项目说明

## 主要功能

### 1. 创建可转债排序网站
- 使用 **GitHub Actions** 自动运行 Python 脚本。
- 利用 **GitHub Pages** 托管静态网页。
- 通过 Python 脚本，每天自动更新网页内容，基于某种策略实现自动化网站更新。

**网站链接**: [Aquareo's Convertible Bonds Sorting](https://aquareo.github.io/)

### 2. 实现日内交易模拟
- 根据 **移动平均交叉策略** 进行模拟交易，也支持自定义策略编写。

## 程序说明

- **update_website.py**：更新网站内容的脚本，每天运行并自动刷新网页数据。
- **day_trading.py**：日内自动交易模拟脚本，使用内置的移动平均交叉策略进行交易模拟，并支持用户自定义策略。

