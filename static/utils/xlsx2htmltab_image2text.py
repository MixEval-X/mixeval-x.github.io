import pandas as pd
import re

meta_dicts = {
    'Claude 3.5 Sonnet': {'url': 'https://www.anthropic.com/news/claude-3-5-sonnet'},
    'GPT-4o': {'url': 'https://openai.com/index/hello-gpt-4o/'},
    'GPT-4V': {'url': 'https://arxiv.org/abs/2303.08774'},
    'Qwen2-VL-72B': {'url': 'https://qwen2.org/vl/'},
    'Gemini 1.5 Pro': {'url': 'https://arxiv.org/abs/2403.05530'},
    'Llama 3.2 90B': {'url': 'https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/'},
    'InternVL2-26B': {'url': 'https://internvl.github.io/blog/2024-07-02-InternVL-2.0/'},
    'InternVL-Chat-V1.5': {'url': 'https://arxiv.org/abs/2404.16821'},
    'Claude 3 Opus': {'url': 'https://www.anthropic.com/news/claude-3-family'},
    'Qwen-VL-MAX': {'url': 'https://huggingface.co/spaces/Qwen/Qwen-VL-Max'},
    'LLaVA-1.6-34B': {'url': 'https://huggingface.co/liuhaotian/llava-v1.6-34b'},
    'Claude 3 Sonnet': {'url': 'https://www.anthropic.com/news/claude-3-family'},
    'Reka Core': {'url': 'https://arxiv.org/abs/2404.12387'},
    'Reka Flash': {'url': 'https://arxiv.org/abs/2404.12387'},
    'InternVL-Chat-V1.2': {'url': 'https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2'},
    'Qwen-VL-PLUS': {'url': 'https://huggingface.co/spaces/Qwen/Qwen-VL-Plus'},
    'Claude 3 Haiku': {'url': 'https://www.anthropic.com/news/claude-3-family'},
    'Gemini 1.0 Pro': {'url': 'https://arxiv.org/abs/2312.11805'},
    'InternLM-XComposer2-VL': {'url': 'https://huggingface.co/internlm/internlm-xcomposer2-vl-7b'},
    'InternVL-Chat-V1.1': {'url': 'https://huggingface.co/OpenGVLab/InternVL-Chat-V1-1'},
    'Yi-VL-34B': {'url': 'https://huggingface.co/01-ai/Yi-VL-34B'},
    'OmniLMM-12B': {'url': 'https://huggingface.co/openbmb/OmniLMM-12B'},
    'DeepSeek-VL-7B-Chat': {'url': 'https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat'},
    'Yi-VL-6B': {'url': 'https://huggingface.co/01-ai/Yi-VL-6B'},
    'InfiMM-Zephyr-7B': {'url': 'https://huggingface.co/Infi-MM/infimm-zephyr'},
    'CogVLM': {'url': 'https://huggingface.co/THUDM/cogvlm-chat-hf'},
    'MiniCPM-V': {'url': 'https://huggingface.co/openbmb/MiniCPM-V'},
    'Marco-VL': {'url': ''},
    'LLaVA-1.5-13B': {'url': 'https://huggingface.co/liuhaotian/llava-v1.5-13b'},
    'SVIT': {'url': 'https://arxiv.org/abs/2307.04087'},
    'mPLUG-OWL2': {'url': 'https://arxiv.org/abs/2311.04257'},
    'SPHINX': {'url': 'https://arxiv.org/abs/2311.07575'},
    'InstructBLIP-T5-XXL': {'url': 'https://huggingface.co/Salesforce/instructblip-flan-t5-xxl'},
    'InstructBLIP-T5-XL': {'url': 'https://huggingface.co/Salesforce/instructblip-flan-t5-xl'},
    'BLIP-2 FLAN-T5-XXL': {'url': 'https://huggingface.co/Salesforce/blip2-flan-t5-xxl'},
    'BLIP-2 FLAN-T5-XL': {'url': 'https://huggingface.co/Salesforce/blip2-flan-t5-xl'},
    'Adept Fuyu-Heavy': {'url': 'https://www.adept.ai/blog/adept-fuyu-heavy'},
    'LLaMA-Adapter2-7B': {'url': 'https://arxiv.org/abs/2304.15010'},
    'Otter': {'url': 'https://arxiv.org/abs/2305.03726'},
    'MiniGPT4-Vicuna-13B': {'url': 'https://github.com/nelsonjchen/cog-MiniGPT-4-vicuna'}
}


proprietary_models = [
    'Claude 3.5 Sonnet',
    'GPT-4o',
    'GPT-4V',
    'Gemini 1.5 Pro',
    'Claude 3 Opus',
    'Qwen-VL-MAX',
    'Claude 3 Sonnet',
    'Reka Core',
    'Reka Flash',
    'Qwen-VL-PLUS',
    'Claude 3 Haiku',
    'Gemini 1.0 Pro',
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

generate_html_table('data/image2text.xlsx', 'static/utils/output_html_tab_image2text.html', 'tab_image2text')
