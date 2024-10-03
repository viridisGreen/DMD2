from transformers import CLIPTextModel, CLIPTextModelWithProjection
import torch 

#todo 把token id转换为text embedding, token id -> text embedding
#* 可以简单理解为自定义了一种特殊的text encoder
class SDXLTextEncoder(torch.nn.Module):
    def __init__(self, args, accelerator, dtype=torch.float32) -> None:
        super().__init__()

        self.text_encoder_one = CLIPTextModel.from_pretrained(
            args.model_id, subfolder="text_encoder", revision=args.revision
        ).to(accelerator.device).to(dtype=dtype)

        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            args.model_id, subfolder="text_encoder_2", revision=args.revision
        ).to(accelerator.device).to(dtype=dtype)

        self.accelerator = accelerator

    def forward(self, batch):
        text_input_ids_one = batch['text_input_ids_one'].to(self.accelerator.device).squeeze(1)
        text_input_ids_two = batch['text_input_ids_two'].to(self.accelerator.device).squeeze(1)
        prompt_embeds_list = []

        for text_input_ids, text_encoder in zip([text_input_ids_one, text_input_ids_two], [self.text_encoder_one, self.text_encoder_two]):
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]

            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(len(text_input_ids_one), -1) # use the second text encoder's pooled prompt embeds (overwrite in for loop) 
        
        #* prompt_embeds: 是拼接后的细粒度嵌入, 包含丰富的语义信息
        #* pooled_prompt_embeds: 是整体的语义表示, 用于高层次的条件生成控制
        return prompt_embeds, pooled_prompt_embeds
