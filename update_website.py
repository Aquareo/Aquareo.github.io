import pandas as pd
import akshare as ak

# 创建示例 DataFrame
def load_ranking():
    # 获取数据
    bond_zh_hs_cov_spot_df = ak.bond_zh_hs_cov_spot()
    
    # 根据 'trade' 列进行升序排序
    sorted_df = bond_zh_hs_cov_spot_df.sort_values(by='trade', ascending=True)
    
    # 添加 'rank' 列，使用 DataFrame 的 reset_index 来生成一个基于排序后的行号的新的索引
    sorted_df['rank'] = sorted_df.reset_index().index + 1  # rank从1开始
    
    # 选择需要的列，并重命名
    sorted_df = sorted_df[['rank', 'symbol', 'name']]
    
    # 重命名列
    #sorted_df = sorted_df.rename(columns={'trade': 'price', 'symbol': 'ticker'})
    
    sorted_df = sorted_df.rename(columns={'symbol': 'ticker'})
    return ranking_df

# 将 DataFrame 转换为 HTML 表格
def dataframe_to_html(ranking_df):
    return ranking_df.to_html(classes='data', header=True, index=False)

# 生成完整的 HTML 页面
def generate_html(content):
    html_content = f"""
    <html>
    <head>
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
    </head>
    <body>
        <h1>可转债投资排名</h1>
        <div>
            {content}  <!-- 直接插入 DataFrame 转换后的 HTML 表格 -->
        </div>
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
