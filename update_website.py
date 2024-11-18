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
    return pd.DataFrame(data)

# 将 DataFrame 转换为 HTML 表格
def dataframe_to_html(ranking_df):
    return ranking_df.to_html(classes='data', header=True, index=False)

# 主程序
def main():
    ranking_df = load_ranking()
    # 转换 DataFrame 为 HTML 表格
    table_html = dataframe_to_html(ranking_df)
    # 构建完整的 HTML 页面
    full_html = f"""
def generate_html(content):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>可转债投资排名</title>
        <style>
            /* 全局设置 */
            body {{
                font-family: 'Arial', sans-serif;
                background-color: #f4f7fa;
                margin: 0;
                padding: 20px;
            }}
            h1 {{
                text-align: center;
                color: #333;
                font-size: 2rem;
                padding: 20px;
                background-color: #007bff;
                color: white;
                border-radius: 10px;
            }}
            /* 表格样式 */
            .data {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* 添加阴影 */
                border-radius: 8px;
                overflow: hidden; /* 圆角效果 */
            }}
            .data th, .data td {{
                padding: 12px;
                text-align: left;
                border: 1px solid #ddd;
            }}
            .data th {{
                background-color: #007bff;
                color: white;
                font-size: 1.1rem;
            }}
            /* 表格行交替颜色 */
            .data tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            /* 悬浮效果 */
            .data tr:hover {{
                background-color: #f1f1f1;
                cursor: pointer;
            }}
            /* 表格单元格字体样式 */
            .data td {{
                font-size: 1rem;
                color: #555;
            }}
            /* 适配移动设备 */
            @media (max-width: 768px) {{
                .data th, .data td {{
                    padding: 8px;
                    font-size: 0.9rem;
                }}
            }}
        </style>
        <link rel="stylesheet" href="styles.css">
    </head>
    <body>
        <h1>可转债投资排名</h1>
        <div>
            {table_html}  <!-- 插入生成的 HTML 表格 -->
            {content}  <!-- 表格内容 -->
        </div>
    </body>
    </html>
    """
    return html_content
def main():
    # 加载数据
    ranking_df = load_ranking()
    # 转换为 HTML 表格
    table_html = dataframe_to_html(ranking_df)
    # 生成最终的 HTML 页面
    full_html = generate_html(table_html)

    # 将生成的 HTML 内容写入 index.html 文件
    with open("index.html", "w", encoding="utf-8") as file:
    # 写入文件
    with open("index.html", "w") as file:
        file.write(full_html)

    print("index.html 文件已生成！")
if __name__ == "__main__":
    main()
