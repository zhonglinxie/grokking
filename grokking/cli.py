from argparse import ArgumentParser

from data import ALL_OPERATIONS
from training import main

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--operation", type=str, choices=ALL_OPERATIONS.keys(), default="x+y")
    parser.add_argument("--training_fraction", type=float, default=0.5)
    parser.add_argument("--prime", type=int, default=97)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=.01)
    parser.add_argument("--num_steps", type=int, default=1e5)
    parser.add_argument("--device", type=str, default="cuda")
    # parser.add_argument("--model", type=str, default="MLP")
    parser.add_argument("--model", type=str, default="Transformer")
    # parser.add_argument("--model", type=str, default="LSTM")
    # parser.add_argument("--optimizer", type=str, default="AdamW")
    # parser.add_argument("--optimizer", type=str, default="Sophia")
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--sharp_penalty", type=float, default=0)
    args = parser.parse_args()

    main(args)
