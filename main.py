import os
import sys
from pathlib import Path

import discord
import torch
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    logging,
)


def fetch_data(filepath):
    if os.path.exists("./data/data.txt"):
        return

    class Scraper(discord.Client):
        async def on_ready(self):
            print(f"Logged in as {self.user}")
            with open(filepath, "w") as f:
                data = ""
                for channel in self.get_all_channels():
                    print(f"#{channel}")
                    last_author = None
                    try:
                        async for message in channel.history(
                            limit=None, oldest_first=True
                        ):
                            if message.author != self.user:
                                content = message.content.replace("\n", "\\n")
                                if message.author == last_author:
                                    data += "\\n" + content
                                else:
                                    data += f"</s>{message.author}:<s>{content}"
                                last_author = message.author
                    except Exception as e:
                        print(e)
                        continue
                    data += "</s>\n"
                f.write(data)
            await self.close()

    intents = discord.Intents.default()
    intents.message_content = True
    with open("token", "r") as f:
        Scraper(intents=intents).run(f.read())


def train_model(dataset):
    logging.set_verbosity_info()
    data_files = [str(x) for x in Path(dataset).glob("**/*.txt")]
    if data_files == []:
        print(f"Error: There are no files in {dataset}. Quitting.")
        sys.exit(1)
    if not os.path.exists("./tokenizer"):
        tokenizer = ByteLevelBPETokenizer()

        tokenizer.train(
            files=data_files,
            vocab_size=52_000,
            min_frequency=1,
            special_tokens=[
                "<s>",  # start of sequence
                "<pad>",  # pad sequence to correct length
                "</s>",  # end of sequence
                "<unk>",  # unknown token
                "<mask>",  # token to tell the model where to fill in
            ],
        )

        os.mkdir("./tokenizer")
        tokenizer.save_model("./tokenizer")

    dataset = load_dataset("text", data_files=data_files, split="train")

    tokenizer = GPT2TokenizerFast.from_pretrained("./tokenizer")
    tokenizer.add_special_tokens(
        {
            "bos_token": "<s>",
            "pad_token": "<pad>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "mask_token": "<mask>",
        }
    )

    def encode(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=1024,
            return_overflowing_tokens=True,
            return_length=True,
        )
        return {"input_ids": outputs["input_ids"]}

    dataset = dataset.map(
        encode, batched=True, remove_columns=["text"], num_proc=os.cpu_count()
    )
    dataset = dataset.train_test_split(test_size=0.1)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.bos_token_id,
            eos_token=tokenizer.eos_token_id,
            n_positions=1024,
        )
    )

    print(f"Model has {model.num_parameters():,} parameters")

    training_args = TrainingArguments(
        output_dir="./models/model3",
        overwrite_output_dir=True,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=50,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=8,
        fp16=True,
        save_steps=500,
        save_total_limit=5,
        prediction_loss_only=False,
        remove_unused_columns=False,
        evaluation_strategy="steps",
        eval_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    trainer.train()
    trainer.save_model("./models/model3")


def finetune_model(data_dir):
    dataset = load_dataset("text", data_dir=data_dir, split="train")
    print(dataset)
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def encode(elem):
        outputs = tokenizer(
            elem["text"],
            truncation=True,
            max_length=1024,
            return_overflowing_tokens=True,
            return_length=True,
        )
        return {"input_ids": outputs["input_ids"]}

    dataset = dataset.map(
        encode, batched=True, remove_columns=["text"], num_proc=os.cpu_count()
    ).train_test_split(test_size=0.1)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    print(f"Model has {model.num_parameters():,} parameters")

    training_args = TrainingArguments(
        output_dir="./models/model3",
        overwrite_output_dir=True,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=50,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=8,
        fp16=True,
        save_steps=500,
        save_total_limit=5,
        prediction_loss_only=False,
        remove_unused_columns=False,
        evaluation_strategy="steps",
        eval_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    trainer.train()
    trainer.save_model("./models/model3")


def evaluate_model() -> str:
    with torch.no_grad():
        tokenizer = GPT2TokenizerFast.from_pretrained("./tokenizer")
        tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "pad_token": "<pad>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "mask_token": "<mask>",
            }
        )

        model = GPT2LMHeadModel.from_pretrained("./models/model3/").to("cuda")

        print(f"Model has {model.num_parameters():,} parameters")

        encoding = tokenizer(
            "Broyojo#2667:Hey @SolarE#9007 how is the modpack going?<|endoftext|>",
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # refer to https://huggingface.co/blog/how-to-generate
        generated_ids = model.generate(
            **encoding,
            max_length=512,
            temperature=0.7,
            num_return_sequences=1,
            top_k=50,
            top_p=1,
            do_sample=True,
            # repetition_penalty=config.repetition_penalty,
            # bos_token_id=tokenizer.bos_token_id,
            # eos_token_id=tokenizer.eos_token_id,
            # pad_token_id=tokenizer.pad_token_id,
        )[0]

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        print("Done generating outputs")

        with open("output.txt", "w+") as f:
            f.write(generated_text)

        return generated_text


def main():
    # fetch_data("./data/data.txt")
    # finetune_model("./data/")
    # evaluate_model()
    train_model("./data/")


if __name__ == "__main__":
    main()
