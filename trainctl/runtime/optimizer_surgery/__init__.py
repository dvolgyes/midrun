"""Typed, conservative optimizer mutations: parameter selection and momentum reset.

No arbitrary optimizer-dict mutation and no arbitrary Python object paths -- only the
operations named here, against a small registry of known optimizer types. An unknown
optimizer type is rejected rather than guessed at.
"""
