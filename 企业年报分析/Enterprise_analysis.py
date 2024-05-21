import pdfplumber
import re
import pandas as pd
import os

def clean_filename(name):
    # 将非字母数字字符替换为下划线
    return re.sub(r'\W+', '_', name)

def extract_value_after_keyword(text, keyword):
    # 查找关键字后的第一个数字
    match = re.search(f'{keyword}[:：\\s“”]*([\\d,.]+)', text)
    if match:
        return match.group(1).replace(',', '')
    
    # 如果未找到，继续向后查找
    index = text.find(keyword)
    if index != -1:
        text = text[index + len(keyword):]

    # 使用循环查找数字
    for _ in range(5):  # 假设最多查找5次
        match = re.search(r'([\\d,.]+)', text)
        if match:
            return match.group(1).replace(',', '')
        # 如果未找到，继续向后查找
        index = text.find(keyword)
        if index != -1:
            text = text[index + len(keyword):]

    return None

def extract_company_info(pdf_path):
    # 从文件名中提取年报时间
    report_time_match = re.search(r'(\d{4})年年度报告', pdf_path)
    report_time = report_time_match.group(1) if report_time_match else None

    # 初始化变量
    company_name = None
    revenue = None
    cost = None
    management_expense = None
    r_d_cost = None

    with pdfplumber.open(pdf_path) as pdf:
        # 读取公司的中文名称和年报时间
        for page in pdf.pages:
            text = page.extract_text()

            if "公司信息" in text:
                # 使用正则表达式提取公司中文名称
                company_name_match = re.search(r'公司的中文名称[:：\s“”]*([^“”\n]+)', text)
                company_name = company_name_match.group(1).strip() if company_name_match else None
                print(company_name, report_time)
                print('----------------------------------------------------------------------')
                break  # 如果找到表格，停止搜索其他页面

        # 读取公司的营业总收入
        for page in pdf.pages:
            text = page.extract_text()

            if "营业总收入" in text:
                # 使用正则表达式提取第一个营业收入数字
                revenue = extract_value_after_keyword(text, "营业总收入")
                break  # 如果找到表格，停止搜索其他页面

        # 读取公司的营业总成本
        for page in pdf.pages:
            text = page.extract_text()

            if "营业总成本" in text:
                # 使用正则表达式提取第一个营业总成本数字
                cost = extract_value_after_keyword(text, "营业总成本")
                break  # 如果找到表格，停止搜索其他页面

        # 读取公司的管理费用
        for page in pdf.pages:
            text = page.extract_text()

            if "管理费用" in text:
                # 使用正则表达式提取管理费用数字
                management_expense = extract_value_after_keyword(text, "管理费用")
                if management_expense:
                    break
                # 如果未找到，继续向后查找
                text = text[text.find("管理费用") + len("管理费用"):]

        # 读取公司的研发费用
        for page in pdf.pages:
            text = page.extract_text()

            if "研发费用" in text:
                # 使用正则表达式提取研发费用数字
                r_d_cost = extract_value_after_keyword(text, "研发费用")

                break  # 如果找到表格，停止搜索其他页面

    # 构建DataFrame
    data = {
        '公司中文名称': [company_name],
        '年报时间': [report_time],
        '营业总收入（元）': [revenue],
        '营业总成本（元）': [cost],
        '管理费用（元）': [management_expense],
        '研发费用（元）': [r_d_cost],
    }
    df = pd.DataFrame(data)

    return df

def analyze_folder(input_folder, output_folder):
    # 初始化结果DataFrame
    result_df = pd.DataFrame()

    # 遍历指定文件夹下的所有PDF文件
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                pdf_data = extract_company_info(pdf_path)
                result_df = pd.concat([result_df, pdf_data], ignore_index=True)

    # 生成输出路径
    output_file_path = os.path.join(output_folder, "Corporate_Finance_All.xlsx")

    # 如果输出目录不存在，创建目录
    os.makedirs(output_folder, exist_ok=True)

    # 保存DataFrame到Excel文件
    result_df.to_excel(output_file_path, index=False)
    print(f"提取的信息已保存到: {output_file_path}")

def main():
    # 替换为实际的输入文件夹路径和输出文件夹路径
    input_folder = 'new'
    output_folder = 'output_folder'

    # 分析文件夹中的所有PDF文件并保存到Excel文件
    analyze_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()
