import torch
import torch.nn as nn
import config
import math
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_pers = nn.Embedding(config.pers,config.pers_emb)
        self.emb_pos = nn.Embedding(config.pos,config.pos_emb)
        self.emb_punc = nn.Embedding(config.punc,config.punc_emb)
        self.emb_wsen = nn.Embedding(config.wsen,config.wsen_emb)
        self.line_pers = nn.Linear(config.pers_emb,2)
        self.line_pos = nn.Linear(config.pos_emb,2)
        self.line_punc = nn.Linear(config.punc_emb,2)
        self.line_wsen = nn.Linear(config.wsen_emb,2)
        self.line_freq = nn.Linear(1,2)
        self.line_wtox = nn.Linear(1,2)

        self.wordAttention = MultiHeadedAttention(config.word_hidden,config.head_num,config.dropout)
        self.wordLinear = nn.Linear(config.word_hidden,config.word_final)

        self.emb_entity = nn.Embedding(config.entity,config.entity_emb)
        self.line_entity = nn.Linear(config.entity_emb,2)
        self.line_bifreq = nn.Linear(2,2)
        self.line_trifreq = nn.Linear(3,2)

        self.phaseAttention = MultiHeadedAttention(config.phase_hidden,config.head_num,config.dropout)
        self.phaseLinear = nn.Linear(config.phase_hidden,config.phase_final)
        self.phasePool = nn.AdaptiveAvgPool1d(1)

        self.emb_ssen = nn.Embedding(config.ssen,config.ssen_emb)
        self.emb_sten = nn.Embedding(config.sten,config.sten_emb)
        self.line_ssen = nn.Linear(config.ssen_emb,2)
        self.line_sten = nn.Linear(config.sten_emb,2)
        self.line_stox = nn.Linear(1,2)

        self.sentAttention = MultiHeadedAttention(config.sent_hidden,config.head_num,config.dropout)
        self.sentLinear = nn.Linear(config.sent_hidden,config.sent_final)
        
        self.depCNN = MyCNN(config.dep, config.dep_emb, config.num_filter)
        self.depLinear = nn.Linear(config.num_filter, config.dep_final)
        self.depAttention = MultiHeadedAttention(config.sent_final+config.dep_final,config.head_num,config.dropout)

        self.fcn = nn.Linear(config.sent_final+config.dep_final,1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()

    def forward(self, batch_sent):
        padded_freq,padded_wtox,padded_pers,padded_pos,padded_punc,padded_wsen,\
            padded_entity,padded_bifreq,padded_trifreq,\
                padded_stox,padded_ssen,padded_sten,\
                    dep,label,mask = batch_sent
        
        emb_pers = self.emb_pers(padded_pers)
        emb_pos = self.emb_pos(padded_pos)
        emb_punc = self.emb_punc(padded_punc)
        emb_wsen = self.emb_wsen(padded_wsen)
        emb_freq = padded_freq.unsqueeze(-1)
        emb_wtox = padded_wtox.unsqueeze(-1)
        line_pers = self.line_pers(emb_pers)
        line_pos = self.line_pos(emb_pos)
        line_punc = self.line_punc(emb_punc)
        line_wsen  = self.line_wsen(emb_wsen)
        line_freq = self.line_freq(emb_freq)
        line_wtox = self.line_wtox(emb_wtox)

        wordF = torch.cat([line_pers,line_pos,line_punc,line_wsen,line_freq,line_wtox],dim = 2)#b*s*12
        wordOut,weight_w = self.wordAttention(wordF,wordF,wordF,mask)#b*s*12, b*12
        wordFinal = self.wordLinear(wordOut)#b*s*2

        emb_entity = self.emb_entity(padded_entity)
        emb_bifreq = padded_bifreq
        emb_trifreq = padded_trifreq
        line_entity = self.line_entity(emb_entity)
        line_bifreq = self.line_bifreq(emb_bifreq)
        line_trifreq = self.line_trifreq(emb_trifreq)

        phaseF = torch.cat([wordFinal,line_entity,line_bifreq, line_trifreq],dim = 2)
        phaseOut,weight_p = self.phaseAttention(phaseF,phaseF,phaseF,mask)#b*s*8,b*8
        phaseFinal = self.phaseLinear(phaseOut)#b*s*2

        phaseFinal = phaseFinal.permute(0,2,1)
        phaseFinal_ = self.phasePool(phaseFinal).squeeze(-1)#batch_size * phase_final

        emb_stox = padded_stox.unsqueeze(-1)#b*1
        emb_ssen = self.emb_ssen(padded_ssen)#b*4
        emb_sten = self.emb_sten(padded_sten)#b*2
        line_stox = self.line_stox(emb_stox)
        line_ssen = self.line_ssen(emb_ssen)
        line_sten = self.line_sten(emb_sten)
        sentF = torch.cat([phaseFinal_,line_stox,line_ssen,line_sten],dim = 1).unsqueeze(1) #b *[1]* (phase_final + 3)

        sentOut, weight_s = self.sentAttention(sentF,sentF,sentF,torch.ones(sentF.size(0),1, device=sentF.device))
        sentFinal = self.sentLinear(sentOut)#batch_size * seq_lenth * sent_final

        depF = torch.stack([self.depCNN(d.to(config.device)) for d in dep])
        depFinal = self.depLinear(depF).unsqueeze(1)#batch_size * dep_final
        
        totalF = torch.cat([sentFinal, depFinal],dim=2)
        totalOut, weight_d = self.depAttention(totalF, totalF,totalF,torch.ones(totalF.size(0),1, device=sentF.device))
        totalFinal = totalOut.squeeze(1)
        s = self.fcn(totalFinal).squeeze(-1)
        score = self.sigmoid(s)
        loss = self.loss(score, label)
        return score, loss, label, torch.cat([weight_w,weight_p,weight_s,weight_d],dim=1)

 