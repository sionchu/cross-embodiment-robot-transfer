"""Local stdio worker keeps ML packages outside Isaac Sim Python."""
import argparse
import base64
import contextlib
import json
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    args=parser.parse_args()
    # Library messages go to stderr; stdout is a JSON-lines protocol only.
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        from act import ACTPolicy
        torch.set_num_threads(4)
        policy=ACTPolicy(args.checkpoint)
    print(json.dumps({'ready':True}),flush=True)
    for line in sys.stdin:
        try:
            message=json.loads(line)
            if message['op']=='reset':
                policy.reset(); result={'reset':True}
            elif message['op']=='predict':
                rgb=np.frombuffer(base64.b64decode(message['front']),dtype=np.uint8).reshape(480,640,3)
                start=time.perf_counter()
                with contextlib.redirect_stdout(sys.stderr):
                    chunk=policy.predict({'state':message['state'],'front':rgb})
                result={'chunk':chunk.tolist(),'inference_ms':(time.perf_counter()-start)*1000}
            else:
                raise ValueError('UNKNOWN_OPERATION')
            print(json.dumps(result,allow_nan=False),flush=True)
        except Exception as exc:
            print(json.dumps({'error':type(exc).__name__+': '+str(exc)}),flush=True)
            return 1
    return 0


if __name__=='__main__':
    sys.exit(main())
