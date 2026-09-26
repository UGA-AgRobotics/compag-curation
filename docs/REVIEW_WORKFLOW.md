# Review workflow - code evidence only in this candidate

The source tree retains review-interface and state-contract code, but this
JR09 model-free package does not expose an accepted live review, model-assist,
GPU, resume, training, or active-learning route. The r92 resources are
reviewer-only and are not a public dependency.

The only current executable review-related evidence is bounded synthetic
contract testing inside `tools/public_safe_acceptance.py`. It does not launch a
GUI, load a model, score proposals, or claim scientific/GPU acceptance.

Historical workflow semantics remain inspectable in the application source and
the explicitly historical upstream README. Do not copy its commands as current
instructions. A deployable review workflow requires a new public-only
distribution identity and separate asset, dependency, rights, hardware, and
release approval.
