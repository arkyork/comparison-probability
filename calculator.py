import torch.nn.functional as F
import torch


class calculate:
    def __init__(self,tokenizer,model,dataset):
        self.win_prob = 0
        self.tokenizer = tokenizer
        self.model = model
        self.dataset = dataset
    def log_prob(self,logits,ids):
        """
            Args:
            logits (torch.Tensor): シーケンスの logits (batch_size, seq_len, vocab_size)
            ids (torch.Tensor): 選ばれたシーケンスのラベル IDs (batch_size, seq_len)
        """
        # log probabilities を計算
        log_probs = F.log_softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
        selected_log_probs = torch.gather(
            log_probs, dim=-1, index=ids.unsqueeze(-1)  # (batch_size, seq_len, 1)
        ).squeeze(-1)  # (batch_size, seq_len)
        prob = selected_log_probs.sum(dim=-1)  # (batch_size,)
        return prob
    def calc(self):
        
        win_prob = self.win_prob
        dataset = self.dataset
        for i in range(len(dataset)):
            chosen_text=dataset[i]["prompt"]+dataset[i]["chosen"]
            rejected_text=dataset[i]["prompt"]+dataset[i]["rejected"]
            #WIN
            chosen_input_ids = self.tokenizer.encode(chosen_text,return_tensors="pt").to("cuda")
            chosen_dist_logits = self.model(chosen_input_ids).logits
            #LOSE
            rejected_input_ids = self.tokenizer.encode(rejected_text,return_tensors="pt").to("cuda")
            rejected_dist_logits = self.model(rejected_input_ids).logits
            #Logit
            chosen_prob=self.log_prob(chosen_dist_logits,chosen_input_ids)
            rejected_prob=self.log_prob(rejected_dist_logits,rejected_input_ids)
            if chosen_prob > rejected_prob:
                win_prob+=1
        print(win_prob,"/",len(dataset))
        print("percentage",win_prob/len(dataset)*100)

        return win_prob