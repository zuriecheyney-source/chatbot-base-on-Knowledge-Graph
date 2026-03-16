# coding: utf-8
import json
import os

def generate_lora_data(input_json_path, output_path):
    """
    根据已有的 medical.json 数据，生成用于 LoRA 微调的指令数据集 (Instruction Dataset)。
    格式: [{"instruction": "...", "input": "...", "output": "..."}]
    """
    if not os.path.exists(input_json_path):
        print(f"找不到输入文件: {input_json_path}")
        return

    lora_data = []
    with open(input_json_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                name = data.get('name', '')
                desc = data.get('desc', '')
                symptoms = data.get('symptom', [])
                cause = data.get('cause', '')
                
                # 场景 1: 根据疾病问描述
                if desc:
                    lora_data.append({
                        "instruction": f"请简要介绍一下{name}这种疾病。",
                        "input": "",
                        "output": desc
                    })
                
                # 场景 2: 根据症状推导疾病 (CoT 风格)
                if symptoms:
                    symptom_str = "、".join(symptoms[:5])
                    lora_data.append({
                        "instruction": "根据以下症状，请分析可能的疾病：",
                        "input": symptom_str,
                        "output": f"根据您提供的症状（{symptom_str}），初步判断可能是{name}。该疾病的典型特征是[内容待填充]。建议进一步检查..."
                    })
                
                # 场景 3: 疾病原因
                if cause:
                    lora_data.append({
                        "instruction": f"引起{name}的原因是什么？",
                        "input": "",
                        "output": cause
                    })
            except:
                continue

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(lora_data, f, ensure_ascii=False, indent=2)
    
    print(f"成功生成 LoRA 训练数据: {len(lora_data)} 条 -> {output_path}")

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(cur_dir, 'data', 'medical.json')
    output_path = os.path.join(cur_dir, 'medical_lora_train.json')
    generate_lora_data(input_path, output_path)
