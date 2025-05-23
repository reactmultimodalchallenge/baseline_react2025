"""
runid util.
Taken from wandb.sdk.lib.runid
"""

from datetime import datetime
import shortuuid  # type: ignore


def generate_id() -> str:
    # ~3t run ids (36**8)
    run_gen = shortuuid.ShortUUID(alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz"))

    # prepend time string to 'run_gen'
    prefix = datetime.now().strftime("%y%m%d%H%M%S")

    return prefix + '_' + run_gen.random(8)
    # return run_gen.random(8)
