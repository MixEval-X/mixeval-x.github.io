import pandas as pd
import re

meta_dicts = {
    'HotShot-XL': {'url': 'https://github.com/hotshotco/Hotshot-XL'},
    'CogVideoX-5B': {'url': 'https://github.com/THUDM/CogVideo'},
    'LaVie': {'url': 'https://arxiv.org/abs/2309.15103'},
    'VideoCrafter2': {'url': 'https://ailab-cvc.github.io/videocrafter2/'},
    'ModelScope': {'url': 'https://arxiv.org/abs/2308.06571'},
    'ZeroScope V2': {'url': 'https://www.plugger.ai/models/zeroscope-v2'},
    'Show-1': {'url': 'https://arxiv.org/abs/2309.15818'}
}




proprietary_models = [
    'dfjqoehjfosdjoca;ndoanid'
]


def generate_html_table(input_file, output_file, table_id='table1'):
    # Read the Excel file
    df = pd.read_excel(input_file, header=None)
    # print(df)
    for col_idx in df.columns[1:]:
        if col_idx != 1 and col_idx != 3 and col_idx != 5:
            continue
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
        lambda x: f'<td class="js-sort-number" style="background-color:#b3b3b3ff;"><strong><a  style="color:#000000ff;"><b>{x}</b></a></strong></td>' if 'Elo' not in str(x)
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

generate_html_table('data/text2video.xlsx', 'static/utils/output_html_tab_text2video.html', 'tab_text2video')
