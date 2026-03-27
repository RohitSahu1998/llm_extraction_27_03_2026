import torch
import json
from transformers import AutoProcessor, AutoModelForImageTextToText
from pdf2image import convert_from_path
from PIL import Image


def load_images(file_path):
    if file_path.lower().endswith(".pdf"):
        return convert_from_path(file_path)
    else:
        return [Image.open(file_path).convert("RGB")]


class QwenExtractor:
    def __init__(self, model_path="/home/rohit.sahu/Qwen_model/qwen_models/Qwen2.5-VL-3B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Qwen model on {self.device}...")

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True
        )

        print("✅ Qwen Model loaded successfully")

        self.prompt = """
        You are a highly accurate OCR data extraction system.
        RULES:
        1. Do not mistake ZIP codes for a Tax ID.
        2. The claimant number is slightly different from the address.
        3. Return ONLY valid JSON matching this structure:

        {
          "claimant_name": "",
          "claimant_number": "",
          "tax_id": "",
          "practice_address": "",
          "billing_address": "",
          "diagnosis_codes": [],
          "date_of_service": "",
          "cpt_codes": [],
          "charges": [],
          "units": [],
          "invoice_date": "",
          "invoice_number": "",
          "taxonomy": "",
          "total_amount": ""
        }
        """

    def extract_data(self, file_path: str) -> dict:
        images = load_images(file_path)
        final_data = {}

        for i, image in enumerate(images):
            print(f"Running Qwen on page {i+1}")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt"
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=800,
                    temperature=0.0,
                    do_sample=False
                )

            result_text = self.processor.decode(
                output[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )

            try:
                cleaned = result_text.replace("```json", "").replace("```", "").strip()
                data = json.loads(cleaned)
            except:
                data = {}

            final_data[f"page_{i+1}"] = data

        #print(final_data)

        return final_data


if __name__ == "__main__":
    extractor = QwenExtractor()
    result = extractor.extract_data("/home/rohit.sahu/Qwen_model/samples_nonstandard_data/Document_1.pdf")

    print(json.dumps(result, indent=2))

#this is part 2 of qwen so take it and save it for now