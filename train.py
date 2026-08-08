import os
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizer_utils import DynamicBPETokenizer

# ===============================
# KHU VỰC CẤU HÌNH THAM SỐ 
# ===============================
DATA_PATH = 'input_chatbot.txt'      
MODEL_SAVE_PATH = 'chatbot_model.pth' 
TOKENIZER_PATH = 'tokenizer_chat.pkl' 

batch_size = 1                   
block_size = 4096             
grad_accumulation = 64       
max_iters = 10000         
eval_interval = 500          
learning_rate = 1e-4          
eval_iters = 30                 

# kiến trúc mô hình
n_embd = 4096                  
n_head = 32                       
n_layer = 32                   
dropout = 0.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# =========================================================================

print(f"[*] Thiết bị kích hoạt: {device.upper()}")
if device == 'cuda':
    print(f"[*] Kiến trúc tối ưu hóa phần cứng: {torch.cuda.get_device_name(0)}")
    torch.set_float32_matmul_precision('high')

tokenizer = DynamicBPETokenizer(vocab_size_max=24000)

def get_current_vocab_size():
    if os.path.exists(TOKENIZER_PATH):
        temp_tok = DynamicBPETokenizer(vocab_size_max=24000)
        temp_tok.load(TOKENIZER_PATH)
        return len(temp_tok.encoder)
    return 326

vocab_size = get_current_vocab_size()

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.key_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.query_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.value_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        B, T, C = x.shape
        x_norm = self.ln1(x)      
        q = self.query_proj(x_norm).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.key_proj(x_norm).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value_proj(x_norm).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)    
        x = x + self.out_proj(attn_out)
        x = x + self.ffn(self.ln2(x))
        return x

class AdvancedLanguageModel(nn.Module):
    def __init__(self, custom_vocab_size=None):
        super().__init__()
        active_vocab = custom_vocab_size if custom_vocab_size is not None else vocab_size
        self.token_embedding_table = nn.Embedding(active_vocab, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, active_vocab)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, tokenizer=None, temperature=0.85, top_k=40, top_p=0.92):
        eolf_id = tokenizer.encoder.get('<|endofline|>', -1) if tokenizer is not None else -1  
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            if top_p > 0.0 and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                for i in range(logits.size(0)):
                    indices_to_remove = sorted_indices[i, sorted_indices_to_remove[i]]
                    logits[i, indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == eolf_id:
                break
        return idx

model = AdvancedLanguageModel().to(device)
start_iter = 0
BACKUP_DIR = 'checkpoints_backup'

if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        print(f"[Lỗi] Không tìm thấy file dữ liệu tại {DATA_PATH}")
        sys.exit()

    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    if not os.path.exists(TOKENIZER_PATH):
        tokenizer.train_from_text(text)
        tokenizer.save(TOKENIZER_PATH)
    else:
        print(f"[*] Tìm thấy tokenizer đã lưu tại {TOKENIZER_PATH}, đang tải lại...")
        tokenizer.load(TOKENIZER_PATH)

    vocab_size = len(tokenizer.encoder)
    print(f"[*] Kích thước từ vựng thực tế (Dynamic Vocab Size): {vocab_size}")

    if model.token_embedding_table.num_embeddings != vocab_size:
        print("[*] Vocab size thay đổi -> khởi tạo lại token_embedding_table và lm_head.")
        model.token_embedding_table = nn.Embedding(vocab_size, n_embd).to(device)
        model.lm_head = nn.Linear(n_embd, vocab_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"[*] Tìm thấy file trọng số chính {MODEL_SAVE_PATH}, đang đồng bộ...")
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_iter = checkpoint.get('iter', 0)

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split):
        data_set = train_data if split == 'train' else val_data
        current_block_size = block_size
        if len(data_set) <= current_block_size:
            current_block_size = len(data_set) - 2 if len(data_set) > 2 else 1
            
        ix = torch.randint(len(data_set) - current_block_size, (batch_size,))
        x = torch.stack([data_set[i:i+current_block_size] for i in ix])
        y = torch.stack([data_set[i+1:i+current_block_size+1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss(model_to_eval):
        out = {}
        model_to_eval.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _, loss = model_to_eval(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model_to_eval.train()
        return out

    print(f"[*] Bắt đầu huấn luyện mô hình hệ thống từ bước {start_iter}...")
    print("[*] Nhấn Ctrl + C bất kỳ lúc nào để dừng và lưu lại bước hiện tại!")

    try:
        for step in range(start_iter, max_iters):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(grad_accumulation):
                xb, yb = get_batch('train')
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16): 
                    logits, loss = model(xb, yb)
                    loss = loss / grad_accumulation
                loss.backward()
            optimizer.step()

            if step % eval_interval == 0 or step == max_iters - 1:
                losses = estimate_loss(model)
                print(f"\n[BƯỚC {step}] | Train Loss: {losses['train']:.4f} | Var (Val) Loss: {losses['val']:.4f}")
                print("-> Test Context (AI Dự đoán ngữ cảnh):")
                test_prompt = "User: xin chào<|endofline|>\nAI:"
                context_ids = torch.tensor([tokenizer.encode(test_prompt)], dtype=torch.long, device=device)
                generated_raw = model.generate(context_ids, max_new_tokens=40, tokenizer=tokenizer)[0].tolist()
                print(f"   Context:\n{tokenizer.decode(generated_raw)}")
                print("="*60)
                
                checkpoint_data = {
                    'iter': step + 1,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                }
                torch.save(checkpoint_data, MODEL_SAVE_PATH)
                
                step_backup_path = os.path.join(BACKUP_DIR, f"model_step_{step}_loss_{losses['val']:.4f}.pth")
                torch.save(checkpoint_data, step_backup_path)

    except KeyboardInterrupt:
        print("\n[-] Phát hiện lệnh ngắt tiến trình thủ công (Ctrl + C) từ bàn phím!")
        print(f"[*] Đang tiến hành sao lưu trọng số mô hình tại bước {step}...")
        checkpoint_data = {
            'iter': step,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(checkpoint_data, MODEL_SAVE_PATH)
        
        step_backup_path = os.path.join(BACKUP_DIR, f"model_step_{step}_interrupted.pth")
        torch.save(checkpoint_data, step_backup_path)
        print("[+] Hoàn thành.")
