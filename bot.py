import logging

import discord
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

discord.utils.setup_logging()


class Bobby(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super(Bobby, self).__init__(intents=intents)

        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained("./models/model7").to("cuda")

    async def on_ready(self):
        logging.info(f"Logged in as {self.user}")

    def generate_next_message(self, prompt, max_message_length):
        # gather message history
        # tokenize message history
        # generate next message

        encoded = self.tokenizer(
            prompt,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=1024 - max_message_length,
        ).to("cuda")

        # refer to https://huggingface.co/blog/how-to-generate
        generated_ids = self.model.generate(
            **encoded,
            # max_length=1024,
            max_new_tokens=max_message_length,
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

        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text

    async def on_message(self, message: discord.Message):
        if message.author == self.user:
            return
        if self.user.display_name.lower() not in message.clean_content.lower():
            return
        """
        when receive new message ->
            generate next message

            if next message is authored by you -> say it
            otherwise -> don't say anything
        """

        async for msg in message.channel.history(limit=1, oldest_first=False):
            c = msg.clean_content.replace("\n", "\\n")
            msg = f"{msg.author.display_name}:{c}<|endoftext|>"
            print(msg)

        response = self.generate_next_message(
            msg + f"{self.user.display_name}:", max_message_length=200
        )[len(msg.replace("<|endoftext|>", "")) :].split(":", 1)

        print(response)

        author = response[0]
        content = response[1]

        if author != self.user.display_name:
            return
        if content == "":
            content = " "
        await message.channel.send(content)


def main():
    bot = Bobby()
    with open("token", "r") as f:
        bot.run(f.read())


if __name__ == "__main__":
    main()
