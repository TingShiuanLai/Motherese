import argparse
import json
import os
import random
import torch
from datasets import load_dataset, get_dataset_config_names
from transformers import WhisperConfig, WhisperTokenizerFast
from WhisperWrapperFromScratch import WhisperProsodyModel
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm


def load_whisper_from_scratch(save_directory):
    with open(os.path.join(save_directory, "prosody_config.json")) as f:
        prosody_dim = json.load(f)["prosody_dim"]

    config = WhisperConfig.from_pretrained(save_directory)
    model = WhisperProsodyModel(config=config, prosody_dim=prosody_dim)
    model.load_state_dict(torch.load(os.path.join(save_directory, "pytorch_model.bin"), map_location="cpu"))
    model.eval()
    return model, prosody_dim


def get_surprisal(sentence, model, tokenizer, prosody_dim):
    tokens = tokenizer(
        sentence.split(),
        is_split_into_words=True,
        return_tensors="pt",
        add_special_tokens=False
    )

    input_ids = tokens.input_ids.to(model.device)
    labels = input_ids.clone()
    seq_len = input_ids.size(1)

    dummy_prosody = torch.zeros((1, seq_len, prosody_dim)).to(model.device)
    positions = torch.arange(seq_len).unsqueeze(0).to(model.device)
    dummy_input_features = torch.zeros((1, 80, 3000)).to(model.device)

    with torch.no_grad():
        outputs = model(
            input_features=dummy_input_features,
            labels=labels,
            prosody=dummy_prosody,
            positions=positions
        )
    return outputs.loss.item()


def evaluate_blimp_task(task_name, model, tokenizer, prosody_dim, num_samples=100):
    ds = load_dataset("nyu-mll/blimp", task_name)['train']
    random.seed(42)
    samples = random.sample(list(ds), min(num_samples, len(ds)))

    y_true = []
    y_pred = []

    for ex in samples:
        good_loss = get_surprisal(ex["sentence_good"], model, tokenizer, prosody_dim)
        bad_loss = get_surprisal(ex["sentence_bad"], model, tokenizer, prosody_dim)
        y_true.append(1)
        y_pred.append(1 if good_loss < bad_loss else 0)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    
    # return acc, precision, recall, f1
    return acc, precision, recall, f1, y_true, y_pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model folder")
    parser.add_argument("--samples", type=int, default=100, help="Number of sentence pairs per subtask")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with 3 subtasks")

    args = parser.parse_args()

    tokenizer = WhisperTokenizerFast(
        vocab_file=r"whisper_positional_embedding/whisper_tokenizer/vocab.json",
        merges_file=r"whisper_positional_embedding/whisper_tokenizer/merges.txt",
        add_prefix_space=True
    )

    model, prosody_dim = load_whisper_from_scratch(args.model_dir)
    model.to(next(model.parameters()).device)

    if args.debug:
        subtasks = ["npi_present_1", "anaphor_gender_agreement", "determiner_noun_agreement_1"]
        print(f"\n🔧 DEBUG MODE: Evaluating 3 subtasks with {args.samples} pairs each\n")
    else:
        subtasks = get_dataset_config_names("nyu-mll/blimp")
        print(f"\n🔍 FULL EVALUATION: {len(subtasks)} BLiMP subtasks with {args.samples} pairs each\n")

    
    results = {}
    all_y_true = []   # 👈 for overall accuracy
    all_y_pred = []

    for task in tqdm(subtasks, desc="Evaluating", unit="task"):
        try:
            # acc, prec, recall, f1, y_true, y_pred = evaluate_blimp_task(task, model, tokenizer, num_samples=args.samples)
            acc, prec, recall, f1, y_true, y_pred = evaluate_blimp_task(task, model, tokenizer, prosody_dim, num_samples=args.samples)

            results[task] = {
                "accuracy": acc,
                "precision": prec,
                "recall": recall,
                "f1": f1
            }

            # 👇 Accumulate for overall score
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)

            tqdm.write(f"✅ {task:<35} Acc: {acc:.2%} | F1: {f1:.2f} | Prec: {prec:.2f} | Recall: {recall:.2f}")
        except Exception as e:
            tqdm.write(f"❌ {task:<35} Error: {e}")

    # Final aggregate metrics
    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_prec, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, average="binary", zero_division=0
    )

    # Print and save
    print(f"\n🌐 OVERALL RESULTS across all subtasks ({len(all_y_true)} pairs total):")
    print(f"   Accuracy: {overall_acc:.2%}")
    print(f"   Precision: {overall_prec:.2%}")
    print(f"   Recall: {overall_recall:.2%}")
    print(f"   F1 Score: {overall_f1:.2%}")

    results["overall"] = {
        "accuracy": overall_acc,
        "precision": overall_prec,
        "recall": overall_recall,
        "f1": overall_f1
    }



    with open("blimp_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    # results = {}
    # for task in tqdm(subtasks, desc="Evaluating", unit="task"):
    #     try:
    #         acc, prec, recall, f1 = evaluate_blimp_task(task, model, tokenizer, prosody_dim, num_samples=args.samples)
    #         results[task] = {
    #             "accuracy": acc,
    #             "precision": prec,
    #             "recall": recall,
    #             "f1": f1
    #         }
    #         tqdm.write(f"✅ {task:<35} Acc: {acc:.2%} | F1: {f1:.2f} | Prec: {prec:.2f} | Recall: {recall:.2f}")

    #     except Exception as e:
    #         tqdm.write(f"❌ {task:<35} Error: {e}")


    # with open("blimp_eval_results.json", "w") as f:
    #     json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()


# from datasets import load_dataset
# from transformers import WhisperConfig, WhisperProcessor, WhisperTokenizerFast
# from WhisperWrapperFromScratch import WhisperProsodyModel
# import torch
# import json
# import os

# # Load model and processor
# def load_whisper_from_scratch(save_directory):
#     with open(os.path.join(save_directory, "prosody_config.json")) as f:
#         prosody_dim = json.load(f)["prosody_dim"]

#     config = WhisperConfig.from_pretrained(save_directory)
#     model = WhisperProsodyModel(config=config, prosody_dim=prosody_dim)

#     model.load_state_dict(torch.load(os.path.join(save_directory, "pytorch_model.bin"), map_location="cpu"))
#     model.eval()
#     return model, prosody_dim

# # processor = WhisperProcessor.from_pretrained("trained_whisper_with_prosody")
# # if processor.tokenizer.pad_token is None:
# #     processor.tokenizer.pad_token = processor.tokenizer.eos_token

# tokenizer = WhisperTokenizerFast(
#     vocab_file="./whisper_tokenizer/vocab.json",
#     merges_file="./whisper_tokenizer/merges.txt",
#     add_prefix_space=True
# )

# model, prosody_dim = load_whisper_from_scratch("trained_whisper_with_prosody")
# device = next(model.parameters()).device
# model.to(device)

# # Updated get_surprisal
# def get_surprisal(sentence, model, tokenizer, prosody_dim):
#     tokens = tokenizer(
#         sentence.split(),
#         is_split_into_words=True,
#         return_tensors="pt",
#         add_special_tokens=False
#     )

#     input_ids = tokens.input_ids.to(model.device)
#     labels = input_ids.clone()
#     seq_len = input_ids.size(1)

#     dummy_prosody = torch.zeros((1, seq_len, prosody_dim)).to(model.device)
#     positions = torch.arange(seq_len).unsqueeze(0).to(model.device)
#     dummy_input_features = torch.zeros((1, 80, 3000)).to(model.device)

#     with torch.no_grad():
#         outputs = model(
#             input_features=dummy_input_features,
#             labels=labels,
#             prosody=dummy_prosody,
#             positions=positions
#         )
#     return outputs.loss.item()


# # BLiMP Evaluation
# def evaluate_blimp_task_custom(task_name):
#     ds = load_dataset("nyu-mll/blimp", task_name)
#     correct = 0
#     total = 0

#     for ex in ds['train']:
#         good_loss = get_surprisal(ex["sentence_good"], model, tokenizer, prosody_dim)
#         bad_loss = get_surprisal(ex["sentence_bad"], model, tokenizer, prosody_dim)
#         if good_loss < bad_loss:
#             correct += 1
#         total += 1

#     return correct / total

# # Evaluate
# subtasks = ["npi_present_1", "principle_A_case_1", "adjunct_island", "anaphor_number_agreement", "determiner_noun_agreement_1"]
# results_sub = {}

# for task in subtasks:
#     acc = evaluate_blimp_task_custom(task)
#     results_sub[task] = acc
#     print(f"{task}: {acc:.2%}")
