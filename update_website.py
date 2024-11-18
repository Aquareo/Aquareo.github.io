import pandas as pd

def load_ranking():
    # 示例数据
    data = {
        'Rank': [1, 2, 3, 4, 5],
        'Name': ['可转债A', '可转债B', '可转债C', '可转债D', '可转债E'],
        'Price': [101.5, 99.2, 98.7, 102.3, 105.0],
        'Yield': [3.5, 4.0, 3.8, 2.5, 3.2]
    }
    return pd.DataFrame(data)

def dataframe_to_html(ranking_df):
    return ranking_df.to_html(classes='data', header=True, index=False)

def generate_html(content):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>可转债投资排名</title>
        <link rel="stylesheet" href="styles.css">
    </head>
    <body>
        <h1>可转债投资排名</h1>
        <div>
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

    # 写入文件
    with open("index.html", "w") as file:
        file.write(full_html)

    print("index.html 文件已生成！")

if __name__ == "__main__":
    main()
