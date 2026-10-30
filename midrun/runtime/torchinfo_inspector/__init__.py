"""Optional torchinfo-backed model summaries.

`torchinfo` is an optional dependency (`midrun[torchinfo]`); every entry point here
degrades gracefully when it isn't installed. `torchinfo.summary` runs the model in eval
mode internally and restores the original mode afterward, so the forward pass of
`torchinfo_rich_summary` does not perturb BatchNorm running statistics or leak into the
live autograd graph.
"""
