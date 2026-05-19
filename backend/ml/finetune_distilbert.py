"""Script de fine-tuning DistilBERT multilingue (LoRA) sur les avis Tour Eiffel.

Ce script démontre la capacité à fine-tuner un transformer pré-entraîné.
Il n'est pas requis pour faire tourner l'application (le modèle TF-IDF est utilisé
en production pour des raisons de latence). Pour lancer :

    pip install torch transformers datasets peft accelerate
    python backend/ml/finetune_distilbert.py

Architecture :
- Backbone : distilbert-base-multilingual-cased (134M params)
- Adapter LoRA : ~0.8M params entraînables (rank=8, alpha=16)
- Tâche : classification sentiment 3 classes (positive/neutral/negative)
- Données : 5000 avis multilingues FR/EN/ES/DE/IT
- Durée estimée : ~15min sur CPU, ~3min sur GPU consumer

Métriques attendues : F1 macro ~0.91 vs ~0.85 pour TF-IDF baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
except ImportError as e:
    print(f"[INFO] Dépendances HuggingFace non installées : {e}")
    print("Installer avec : pip install torch transformers datasets peft accelerate")
    print("Ce script est optionnel : le modèle TF-IDF sklearn est utilisé en production.")
    raise SystemExit(0)


ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT / "data" / "reviews.csv"
ARTIFACTS = ROOT / "ml_artifacts" / "distilbert_lora"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


MODEL_NAME = "distilbert-base-multilingual-cased"
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def main() -> None:
    print(f"Chargement {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["label"] = df["sentiment"].map(LABEL2ID)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["sentiment"]
    )
    print(f"Train : {len(train_df)} | Test : {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df[["text", "label"]]).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Configuration LoRA - adapter léger, seulement ~0.8M params entraînables
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # On reste léger pour le démo (1 epoch + petit batch) — CPU friendly
    training_args = TrainingArguments(
        output_dir=str(ARTIFACTS),
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=5e-4,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=100,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    # transformers 5.x : `tokenizer` renommé en `processing_class`
    import inspect
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    sig = inspect.signature(Trainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    eval_results = trainer.evaluate()
    print("\nRésultats finaux :", eval_results)

    trainer.save_model(str(ARTIFACTS))
    tokenizer.save_pretrained(str(ARTIFACTS))

    with open(ARTIFACTS / "metrics.json", "w") as fp:
        json.dump(eval_results, fp, indent=2)


if __name__ == "__main__":
    main()
