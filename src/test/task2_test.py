import os
import json
import base64
from dotenv import load_dotenv
from openai import OpenAI

# Tải các biến môi trường từ tệp .env
load_dotenv()
# Hàm để mã hóa hình ảnh thành base64
def encode_image(image_path):
    """Mã hóa tệp hình ảnh thành chuỗi base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp hình ảnh tại {image_path}")
        return None
    except Exception as e:
        print(f"Đã xảy ra lỗi khi mã hóa hình ảnh: {e}")
        return None

def load_dataset(dataset_path)->list:
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    return dataset

def setup_client()->OpenAI:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),  # Đảm bảo OPENROUTER_API_KEY được thiết lập trong tệp .env của bạn
    )
    return client

def get_response(client: OpenAI, q_data: dict)->str:
    question_text = q_data["question"]
    if q_data["question_type"] == "Multiple choice":
        choices_text = "\n".join([f"{key}: {value}" for key, value in q_data["choices"].items()])
        prompt_content = f"{question_text}\n{choices_text}\nChọn câu trả lời đúng nhất."
    elif q_data["question_type"] == "Yes/No":
        prompt_content = f"{question_text} Trả lời 'Đúng' hoặc 'Sai'."
    else:
        print(f"Loại câu hỏi không xác định cho ID {q_data['id']}: {q_data['question_type']}")

    base_path = "../../data/VLSP 2025 - MLQA-TSR Data Release/public_test/public_test_images/public_test_images"
    image_file = q_data["image_id"] + ".jpg"
    image_path = os.path.join(base_path, image_file)
    base64_image = encode_image(image_path)

    print(f"\n--- Đang xử lý câu hỏi ID: {q_data['id']} ---")
    print(f"Câu hỏi: {question_text}")

    # Tạo yêu cầu hoàn thành trò chuyện
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-maverick",  # Chỉ định mô hình Qwen
        messages=[
            {
                "role": "system",
                "content": "Bạn là một trợ lý AI chuyên trả lời câu hỏi trắc nghiệm và câu hỏi đúng hay sai liên quan đến pháp luật đường bộ tại Việt Nam. Khi được cung cấp một câu hỏi và các lựa chọn, bạn PHẢI trả lời bằng chữ cái của lựa chọn đúng (ví dụ: A, B, C, D) ở dùng cuối cùng của câu trả lời trong câu hỏi trắc nghiệm và PHẢI trả lời Đúng/Sai ở dòng cuối cùng trong câu hỏi đúng hay sai. Có thể tự suy luận để đưa ra câu trả lời. Hãy suy luận mối quan hệ giữa bức tranh và câu hỏi rồi đa câu trả lời chính xác nhất ở dòng cuối cùng"
            },
            {
                "role": "user",
                "content": [
                    # Phần văn bản của tin nhắn (câu hỏi và các lựa chọn nếu có)
                    {"type": "text", "text": prompt_content},
                    # Phần hình ảnh của tin nhắn
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpg;base64,{base64_image}"},
                    },
                ],
            }
        ],
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    questions_data = [
        {
            "id": "public_test_52",
            "image_id": "public_test_5_12",
            "question": "Các biển báo xuất hiện trong hình bên là loại biển báo gì?",
            "question_type": "Multiple choice",
            "choices": {
                "A": "Biển báo chỉ dẫn và biển hiệu lệnh",
                "B": "Biển báo cấm và biển chỉ dẫn",
                "C": "Biển báo hiệu lệnh và biển báo cấm",
                "D": "Cả A, B, và C đều sai."
            }
        },
        {
            "id": "public_test_53",
            "image_id": "public_test_5_11",
            "question": "Đây là biển báo cấm vượt. Đúng hay sai?",
            "question_type": "Yes/No"
        },
        # Bạn có thể thêm nhiều câu hỏi khác vào đây
    ]
    # dataset_path = "../../data/VLSP 2025 - MLQA-TSR Data Release/public_test/vlsp_2025_public_test_task2.json"
    # questions_data = load_dataset(dataset_path)
    client = setup_client()
    for q_data in questions_data:
        response = get_response(client, q_data)
        print(response)
