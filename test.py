import dataclasses

import jax

from openpi.models import model as _model
from openpi.policies import libero_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import time

config = _config.get_config("pi0_fast_libero")
checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_libero")

start = time.time()
# Create a trained policy.
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
example = libero_policy.make_libero_example()
result = policy.infer(example)
end = time.time()

# Delete the policy to free up memory.
del policy

print("Actions shape:", result["actions"].shape)
print("Time:", end - start)