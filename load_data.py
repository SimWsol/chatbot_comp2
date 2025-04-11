def load_dataset(filepath):
    input_texts, target_texts = [], []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                # Split line into input and target based on delimiters
                if '|' in line:
                    input_text, target_text = line.strip().split('|', 1)
                elif '\t' in line:
                    input_text, target_text = line.strip().split('\t', 1)
                else:
                    continue
                input_texts.append(input_text.strip())
                target_texts.append(target_text.strip())
            except ValueError:
                # Skip lines that don't match the expected format
                continue
    return input_texts, target_texts