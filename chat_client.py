import os
import sys
import torch
import urllib.parse
from tokenizer_utils import DynamicBPETokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("[*] Đang khởi động hệ thống AI...")

MAX_HISTORY_TOKENS = 200 
TOKENIZER_CHAT_PATH = 'tokenizer_chat.pkl'

tokenizer_chat = DynamicBPETokenizer()
if os.path.exists(TOKENIZER_CHAT_PATH):
    tokenizer_chat.load(TOKENIZER_CHAT_PATH)
    vocab_size_chat = len(tokenizer_chat.encoder)
else:
    print(f"[Lỗi] Không tìm thấy file Tokenizer tại {TOKENIZER_CHAT_PATH}. Vui lòng chạy train.py trước!")
    sys.exit()

from train import AdvancedLanguageModel

model_chat = AdvancedLanguageModel(custom_vocab_size=vocab_size_chat).to(device)
if os.path.exists('chatbot_model.pth'):
    try:
        model_chat.load_state_dict(torch.load('chatbot_model.pth', map_location=device)['model_state'])
    except Exception as e:
        print(f"[-] Không thể tải file checkpoint do lệch cấu trúc ma trận: {e}")
        print("[*] Đang tự động sửa và tải lại từ ma trận nhúng mới...")
        checkpoint = torch.load('chatbot_model.pth', map_location=device)
        state_dict = checkpoint['model_state']
        saved_vocab_size = state_dict['token_embedding_table.weight'].shape[0]
        model_chat = AdvancedLanguageModel(custom_vocab_size=saved_vocab_size).to(device)
        model_chat.load_state_dict(state_dict)
model_chat.eval()

tokenizer_code = DynamicBPETokenizer()
model_code = None 

if os.path.exists('tokenizer_code.pkl') and os.path.exists('code_model.pth'):
    try:
        tokenizer_code.load('tokenizer_code.pkl')
        vocab_size_code = len(tokenizer_code.encoder)
        model_code = AdvancedLanguageModel(custom_vocab_size=vocab_size_code).to(device)
        model_code.load_state_dict(torch.load('code_model.pth', map_location=device)['model_state'])
        model_code.eval()
    except Exception:
        model_code = None
else:
    print("[*] Lưu ý: Chưa tìm thấy dữ liệu Model 2 (Code). Tính năng dịch chuyển gõ lệnh code tạm thời khóa.")

conversation_history = []

def build_context_prompt(new_user_input):
    """
    Xây dựng ngữ cảnh liên tục. Đảm bảo chèn thẻ kết thúc <|endofline|> 
    vào đúng vị trí như cấu trúc data train liên tiếp.
    """
    full_prompt = ""
    for turn in conversation_history:
        full_prompt += f"{turn['role']}: {turn['content']}<|endofline|>\n"
    full_prompt += f"User: {new_user_input}<|endofline|>\nAI:"
    tokens = tokenizer_chat.encode(full_prompt)
    
    while len(tokens) > MAX_HISTORY_TOKENS and len(conversation_history) > 0:
        conversation_history.pop(0)
        full_prompt = ""
        for turn in conversation_history:
            full_prompt += f"{turn['role']}: {turn['content']}<|endofline|>\n"
        full_prompt += f"User: {new_user_input}<|endofline|>\nAI:"
        tokens = tokenizer_chat.encode(full_prompt)
        
    return full_prompt

def execute_chat_with_memory(user_input):
    global conversation_history
    
    prompt_with_memory = build_context_prompt(user_input)
    input_ids = torch.tensor([tokenizer_chat.encode(prompt_with_memory)], dtype=torch.long, device=device)
    
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        chat_tokens = model_chat.generate(
            input_ids, 
            max_new_tokens=40, 
            tokenizer=tokenizer_chat,
            temperature=0.2,
            top_k=5,
            top_p=0.85
        )[0].tolist()
    
    full_output_text = tokenizer_chat.decode(chat_tokens)

    prompt_len_text = len(prompt_with_memory)
    chat_response = full_output_text[prompt_len_text:].strip()
    
    chat_response = chat_response.replace('<|endofline|>', '').replace('<|endoftext|>', '')
    if "User:" in chat_response:
        chat_response = chat_response.split("User:")[0].strip()
    if "AI:" in chat_response:
        chat_response = chat_response.split("AI:")[0].strip()

    if "<CONNECT_TO_CODER_M2>" in chat_response and model_code is not None:
        try:
            extracted_prompt = chat_response.split("<CONNECT_TO_CODER_M2>")[1].strip()
        except Exception:
            extracted_prompt = user_input
            
        code_prompt = f"### Instruction: {extracted_prompt}\n### Python Code:\n"
        code_input_ids = torch.tensor([tokenizer_code.encode(code_prompt)], dtype=torch.long, device=device)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            code_tokens = model_code.generate(
                code_input_ids, 
                max_new_tokens=512,
                temperature=0.2,
                top_k=10,
                top_p=0.95
            )[0].tolist()
        
        raw_code = tokenizer_code.decode(code_tokens[len(code_input_ids[0]):]).strip()
        if "```python" not in raw_code:
            raw_code = "```python\n" + raw_code
        if not raw_code.endswith("```"):
            raw_code = raw_code + "\n```"
            
        ai_final_output = f"⚡ [AI LẬP TRÌNH PHẢN HỒI]:\n{raw_code}"
        conversation_history.append({"role": "User", "content": user_input})
        conversation_history.append({"role": "AI", "content": ai_final_output})
        return ai_final_output
        
    else:
        conversation_history.append({"role": "User", "content": user_input})
        conversation_history.append({"role": "AI", "content": chat_response})
        return chat_response

# Run Client CMD
print("\n" + "="*50)
print("HỆ THỐNG CHATBOT")
print("="*50)

while True:
    try:
        user_msg = input("\nUser: ")
        if user_msg.lower() in ['exit', 'quit', 'thoát']:
            print("[*] Đang ngắt kết nối máy chủ AI...")
            break
        if not user_msg.strip():
            continue
            
        ai_output = execute_chat_with_memory(user_msg)
        print(f"\nAI: {ai_output}")
        
    except KeyboardInterrupt:
        print("\n[*] Đang đóng ứng dụng khách...")
        break