import json
import multiprocessing

from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
)


def main():
    with open("./data/scraped_data.json", "r") as f:
        scraped = json.load(f)

    processed = []

    for server in scraped.values():
        for channel in server.values():
            messages = ""
            for message in channel:
                messages += (
                    f"{message['author']}:{message['clean_content']}<|endoftext|>"
                )
            processed.append({"messages": messages})

    with open("./data/processed_data.json", "w") as f:
        f.write(json.dumps(processed))

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def encode(example):
        return {
            "input_ids": tokenizer(
                example["messages"],
                truncation=True,
                max_length=1024,
                return_overflowing_tokens=True,
                return_length=True,
            )["input_ids"]
        }

    dataset = (
        load_dataset("json", data_files="./data/processed_data.json", split="train")
        .map(
            encode,
            batched=True,
            remove_columns=["messages"],
            num_proc=multiprocessing.cpu_count(),
        )
        .shuffle()
        .train_test_split(0.1)
    )

    model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./models/model7",
        overwrite_output_dir=True,
        logging_strategy="steps",
        logging_steps=50,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=8,
        fp16=True,
        save_steps=500,
        save_total_limit=5,
        prediction_loss_only=False,
        remove_unused_columns=False,
        evaluation_strategy="steps",
        eval_steps=500,
        # gradient_accumulation_steps=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    trainer.train()
    trainer.save_model("./models/model7")


if __name__ == "__main__":
    main()
