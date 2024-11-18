import pandas as pd
from jinja2 import Template

# 创建示例 DataFrame
def load_ranking():
    data = {
        'Rank': [1, 2, 3, 4, 5],
        'Name': ['可转债A', '可转债B', '可转债C', '可转债D', '可转债E'],
        'Price': [101.5, 99.2, 98.7, 102.3, 105.0],
        'Yield': [3.5, 4.0, 3.8, 2.5, 3.2]
    }
    ranking_df = pd.DataFrame(data)
    return ranking_df

# 将 DataFrame 转换为 HTML 表格
def dataframe_to_html(ranking_df):
    return ranking_df.to_html(classes='data', header=True, index=False)

# 渲染 HTML 模板
def generate_html(content):
    # 读取 HTML 模板
    with open("index.html", "r", encoding="utf-8") as f:
        template = Template(f.read())
    
    # 使用 Jinja2 渲染模板，插入表格内容
    return template.render(content=content)

# 主程序
def main():
    ranking_df = load_ranking()

    # 转换 DataFrame 为 HTML 表格
    table_html = dataframe_to_html(ranking_df)

    # 使用 Jinja2 渲染模板并生成完整的 HTML 页面
    full_html = generate_html(table_html)

    # 将生成的 HTML 内容写入 index.html 文件
    with open("index.html", "w", encoding="utf-8") as file:
        file.write(full_html)

if __name__ == "__main__":
    main()
