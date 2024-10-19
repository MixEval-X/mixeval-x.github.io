import pandas as pd
import re

meta_dicts = {
    'Claude 3.5 Sonnet': {'url': 'https://www.anthropic.com/news/claude-3-5-sonnet'},
    'GPT-4o': {'url': 'https://openai.com/index/hello-gpt-4o/'},
    'Gemini 1.5 Pro': {'url': 'https://arxiv.org/abs/2403.05530'},
    'GPT-4V': {'url': 'https://arxiv.org/abs/2303.08774'},
    'Qwen2-VL-72B': {'url': 'https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct'},
    'Gemini 1.5 Flash': {'url': 'https://arxiv.org/abs/2403.05530'},
    'LLaVA-OneVision-72B-OV': {'url': 'https://huggingface.co/lmms-lab/llava-onevision-qwen2-72b-ov-sft'},
    'Qwen2-VL-7B': {'url': 'https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct'},
    'LLaVA-Next-Video-34B': {'url': 'https://huggingface.co/lmms-lab/LLaVA-NeXT-Video-34B'},
    'Claude 3 Haiku': {'url': 'https://www.anthropic.com/news/claude-3-family'},
    'LLaVA-Next-Video-7B': {'url': 'https://huggingface.co/lmms-lab/LLaVA-NeXT-Video-7B'},
    'Reka-edge': {'url': 'https://arxiv.org/abs/2404.12387'},
    'LLaMA-VID': {'url': 'https://arxiv.org/abs/2311.17043'},
    'VideoLLaVA': {'url': 'https://arxiv.org/abs/2311.10122'},
    'Video-ChatGPT': {'url': 'https://arxiv.org/abs/2306.05424'},
    'mPLUG-video': {'url': 'https://arxiv.org/abs/2306.04362'}
}



proprietary_models = [
    'Claude 3.5 Sonnet',
    'GPT-4o',
    'Gemini 1.5 Pro',
    'GPT-4V',
    'Gemini 1.5 Flash',
    'Claude 3 Haiku',
    'Reka-edge'
]


def generate_html_table(input_file, output_file, table_id='table1'):
    # Read the Excel file
    df = pd.read_excel(input_file, header=None)
    # print(df)
    for col_idx in df.columns[1:]:
    # Scores are in the rows below the first one
        scores = pd.to_numeric(df.iloc[1:, col_idx], errors='coerce')
        # Check if there are enough scores to proceed
        if scores.dropna().empty:
            continue
        # print(scores)
        # Find the index of the highest score
        first_max_idx = scores.idxmax()  # Adjust index for header row
        # Apply bold formatting
        df.iloc[first_max_idx, col_idx] = f"<b>{df.iloc[first_max_idx, col_idx]}</b>"
        
        # Nullify the highest score to find the second highest
        scores[first_max_idx] = pd.NA  # Adjust index back for zero-based indexing
        # Check again for remaining valid scores
        if scores.dropna().empty:
            continue
        
        second_max_idx = scores.idxmax()  # Adjust index for header row
        # Apply underline formatting
        if pd.notna(second_max_idx):
            df.iloc[second_max_idx, col_idx] = f"<u>{df.iloc[second_max_idx, col_idx]}</u>"

        
    
    # for t in df.iloc[1:, 0]:
    #     print(f"'{t}': {{'url': ''}},")
    
    # Update the first column to include the hyperlink and formatting
    df.iloc[1:, 0] = df.iloc[1:, 0].apply(lambda x: f'''<td style="text-align: center;width: 200px;"><a href="{meta_dicts[x]['url']}" target="_blank"><b>{x}</b></a></td>''')
    
    df.iloc[0] = df.iloc[0].apply(
        lambda x: f'<td class="js-sort-number" style="background-color:#b3b3b3ff;"><strong><a  style="color:#000000ff;"><b>{x}</b></a></strong></td>' if '2Text' not in str(x)
        else f'<td class="js-sort-number" style="background-color:#b3b3b3ff;"><strong><a  style="color:#000000ff;"><b>{f"{x}<br>🥇"}</b></a></strong></td>'
        )
    
    # Convert the DataFrame to an HTML string, without headers and index
    html_table = df.to_html(index=False, escape=False, header=False)
    # print(type(html_table))
    # Add the table with specific class and ID
    html_table = html_table.replace('<table border="1" class="dataframe">', f'<table class="js-sort-table" id="{table_id}" style="border: 2px solid #999999ff;">').replace('<td><td ', '<td ').replace('</td></td>', '</td>')
    
    # Remove thead and tbody tags
    html_table = html_table.replace('<thead>', '').replace('</thead>', '').replace('nan', '').replace(' (Mixed)', '<br>(Mixed)').replace("Arena Elo (0527)", "Arena Elo<br> (0527)")
    html_table = html_table.replace('<tbody>', '').replace('</tbody>', '')

    # Replace the default <tr>, <th>, and <td> to include style
    pattern = r'(<tr>\n\s+<td style="text-align: center;width: 200px;"><a href="([^"]+)" target="_blank"><b>(.*?)</b></a></td>)'
    matches = re.findall(pattern, html_table)
    # print(matches)
    for match in matches:
        for p_m in proprietary_models:
            is_pm = False
            if p_m in match:
                is_pm = True
                break
        if is_pm:
            html_table = html_table.replace(match[0], match[0].replace('<tr>\n', '<tr style="background-color: #ecececff;">\n'))
        else:
            html_table = html_table.replace(match[0], match[0].replace('<tr>\n', '<tr style="background-color: #fcfcfcff;">\n'))
        
    # html_table = html_table.replace('<tr>', '<tr style="background-color: rgba(255, 208, 80, 0.15);">')
    # html_table = html_table.replace('<td>', '<td style="padding: 8px; border: 1px solid #ddd; text-align: center;">')
    # html_table = html_table.replace('<th>', '<th style="background-color: #f4f4f4; padding: 8px; border: 1px solid #ddd; text-align: center;">')
    
    # Write the HTML Table to a file
    with open(output_file, "w") as file:
        file.write(html_table)

generate_html_table('data/video2text.xlsx', 'static/utils/output_html_tab_video2text.html', 'tab_video2text')
