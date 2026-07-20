import yaml, sys
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser

parser = make_arg_parser(FlexibleArgumentParser())
ok = 0
for f in sys.argv[1:]:
    try:
        cmd = yaml.safe_load(open(f))["services"]["model"]["command"]
        a = parser.parse_args(cmd)
        spec = getattr(a, "speculative_config", None)
        kv = getattr(a, "kv_cache_dtype", None)
        quant = getattr(a, "quantization", None)
        spec_str = f" spec={spec.get('method','')}" if isinstance(spec, dict) and spec else ""
        kv_str = f" kv_dtype={kv}" if kv else ""
        quant_str = f" quant={quant}" if quant else ""
        print(f"OK {f} ({len(cmd)} args){quant_str}{kv_str}{spec_str}")
        ok += 1
    except Exception as e:
        print(f"FAIL {f}: {e}")
print(f"\n{ok}/{len(sys.argv)-1} passed")
