import json
import argparse
import numpy as np
from pathlib import Path

from llm_command_helper import _compress_world_state


def main():
    parser = argparse.ArgumentParser(description="Compare raw vs compressed world_state lengths.")
    parser.add_argument("input", help="Path to JSON file produced by dump_world_state")
    parser.add_argument("--quantize", action="store_true", help="Use integer quantisation (×10)")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    ws = np.array(data, dtype=np.float32)

    # Raw JSON string length (pretty similar to token count for numbers)
    raw_json_str = json.dumps(data)
    raw_len = len(raw_json_str)

    # Compressed string length
    comp_str = _compress_world_state(ws, quantize=args.quantize)
    comp_len = len(comp_str)

    reduction = 100 * (1 - comp_len / raw_len)
    print("Raw JSON chars :", raw_len)
    print("Compressed chars:", comp_len)
    print(f"Reduction      : {reduction:.1f}%")


if __name__ == "__main__":
    main() 