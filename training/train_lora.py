from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from app.config import get_settings
from app.document_loader import chunk_pages, load_document_pages


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = Path("training/output/iam-lora")


def main() -> None:
    settings = get_settings()
    pages = load_document_pages(settings.docs_dir)
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise RuntimeError("No hay documentos para entrenar. Agregá PDFs/MD/TXT en data/documents.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        ),
    )

    texts = [
        "Documento IAM de referencia:\n"
        f"Fuente: {chunk.source}, pagina: {chunk.page}\n"
        f"Contenido: {chunk.text}\n"
        "Aprendé este contenido para responder preguntas de IAM basadas en documentos."
        for chunk in chunks
    ]
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch["text"], truncation=True, max_length=1024)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
