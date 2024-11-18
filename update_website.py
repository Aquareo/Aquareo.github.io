import pandas as pd

# 创建示例 DataFrame
def load_ranking():
    # 示例数据
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

# 生成完整的 HTML 页面
def generate_html(content):
    html_content = f"""
    <html>
    <head>
        <title>Updated DataFrame Page</title>
        <style>
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            table, th, td {{
                border: 1px solid black;
            }}
            th, td {{
                padding: 8px;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <h1>Updated DataFrame</h1>
        {content}
    </body>
    </html>
    """
    return html_content

# 主程序
def main():
    # 获取数据
    ranking_df = load_ranking()

    # 转换 DataFrame 为 HTML 表格
    table_html = dataframe_to_html(ranking_df)

    # 生成完整的 HTML 页面
    full_html = generate_html(table_html)

    # 将 HTML 内容写入到 index.html 文件
    with open("index.html", "w") as file:
        file.write(full_html)

if __name__ == "__main__":
    main()
