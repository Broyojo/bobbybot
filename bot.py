import discord
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class Bobby(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super(Bobby, self).__init__(intents=intents)
        self.tokenizer = GPT2TokenizerFast.from_pretrained("./tokenizer")
        self.tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "pad_token": "<pad>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "mask_token": "<mask>",
            }
        )
        self.model = GPT2LMHeadModel.from_pretrained("./models/model3/").to("cuda")

    def evaluate_model(self, prompt: str) -> str:
        with torch.no_grad():
            print(f"Model has {self.model.num_parameters():,} parameters")

            encoding = self.tokenizer(
                prompt,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to("cuda")

            # refer to https://huggingface.co/blog/how-to-generate
            generated_ids = self.model.generate(
                **encoding,
                max_length=1024,
                # max_new_tokens=512,
                temperature=1,
                num_return_sequences=1,
                top_k=50,
                top_p=1,
                do_sample=True,
                # repetition_penalty=1.1,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )[0]

            generated_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

            print(generated_text)

            prompt = prompt.replace("<s>", "").replace("</s>", "")

            length_of_new = len(generated_text) - len(prompt)

            generated_text = generated_text[
                max(len(generated_text) - length_of_new, 0) : len(generated_text)
            ]

            print(generated_text)

            generated_text = generated_text.replace("\\n", "\n")

            if generated_text == "":
                generated_text = " "

            return generated_text[:2000]

    async def on_ready(self):
        print("Logged on as", self.user)

    async def on_message(self, message: discord.Message):
        # don't respond to ourselves
        if message.author == self.user:
            return

        # if self.user in message.mentions:
        past_messages = ""

        async for message in message.channel.history(limit=20, oldest_first=False):
            content = message.content.replace("\n", "\\n")
            past_messages = f"{message.author}:<s>{content}</s>" + past_messages
        response = self.evaluate_model(
            past_messages[max(len(past_messages) - 512, 0) : len(past_messages)]
            + "Bobby#5900:<s>"
        )
        await message.channel.send(response)


def main():
    bot = Bobby()
    with open("token", "r") as f:
        bot.run(f.read())


if __name__ == "__main__":
    main()
