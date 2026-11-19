# Trainctl: three-slide visual explanation

This file specifies exactly three images for a text-to-image generator. The deck explains the project's central message:

> Add one mixin to an ordinary PyTorch Lightning module, then inspect and control the live training run from the
> outside.

The lifecycle shell-hook feature is included as one capability within the larger system. Generate exactly three slides.

Reference: [Lifecycle hooks and unified logging handover](LIFECYCLE_HOOKS_PLAN.md).

## Shared visual direction

- Format: 16:9, presentation-ready technical infographic, clean vector style, very light background, generous
  whitespace, rounded rectangles, thin orthogonal arrows, and large legible sans-serif labels.
- Color language: navy = Lightning training; teal = live observation; amber = control/hold; orange = tensor or graph
  capture; purple = persisted artifacts and Loguru diagnostics; red = failure or high-risk operation; gray = read-only,
  disabled, or deferred.
- Use one consistent central icon for the training process: a navy Lightning loop around a small neural network and
  batch stream.
- Render exact labels as written. Avoid decorative pseudo-code, dense source-code screenshots, tiny endpoint lists, and
  invented product features.
- Do not imply cloud hosting, an asynchronous worker queue, automatic retries, arbitrary remote shell execution, a
  sandbox, or native HTTP lifecycle hooks. An optional shell hook may use `curl`, but native HTTP hooks are deferred.
- Use the footer `Trainctl — live Lightning inspection and control` on all three slides.

## Slide 1 — Add one mixin, gain a live control plane

### Purpose

Make the adoption story unmistakable: the user keeps a normal Lightning module and enables Trainctl by inheriting from
`TrainctlMixin`.

### Image-generation prompt

```text
Create a 16:9 clean vector technical infographic titled "Add one mixin. Gain a live Lightning control plane.".

Place a large code-shaped card on the left, with only these two readable lines:

class MyModel(TrainctlMixin, LightningModule):
    ...

Put a large teal arrow from this card to a central navy rounded rectangle labelled "ordinary Lightning training".
Inside the central rectangle show a small neural network, batches, optimizer steps, and a looping arrow labelled
"training continues normally".

Around the central rectangle place five compact capability cards, all visibly connected to one teal outline labelled
"Trainctl". The cards must say exactly:

"observe live state"
"pause at safe points"
"capture evidence"
"apply targeted controls"
"run lifecycle hooks"

Under "capture evidence", add: "stacks • graphs • tensors • CUDA memory • profiles".
Under "run lifecycle hooks", add: "light metadata • heavy temporary files".

On the far right show a simple operator terminal and a small local REST client labelled "operator". Connect them to
Trainctl with teal arrows. Add a bottom banner with this exact text:
"Inspect, diagnose, and act while the run is alive".

Use navy for Lightning, teal for the mixin and observation, amber for holds and controls, orange for captures, purple for
artifacts, and red only for failures. Make all text legible at 1920×1080. Do not show cloud services, a second training
engine, a web-scale dashboard, an async queue, or automatic model redesign.
```

### Required visual facts

- The code card must be the most memorable element after the title.
- The exact phrase `TrainctlMixin` must be legible and visually connected to the existing `LightningModule`.
- The central training process must look unchanged after integration.
- The five cards are capabilities of one system, not five independent products.
- “Run lifecycle hooks” is one peer capability, not the entire project.

### Caption / speaker note

The killer feature is the integration surface: compose `TrainctlMixin` before the Lightning module base class. The model
remains a normal Lightning model, while Trainctl adds live observation, safe control, debugging, artifacts, and hooks.

## Slide 2 — One mixin, several access paths, one safe boundary

### Purpose

Show the architecture without making the operator learn implementation details. The key idea is that the mixin is thin;
the runtime owns the control and inspection plane, and stateful work returns to the training thread at safe points.

### Image-generation prompt

```text
Create a 16:9 vector architecture diagram titled "The mixin is thin; the live run stays in control".
Use three vertical zones from left to right.

LEFT zone title: "Operator and external tools".
Show four small cards:
"REST client / curl"
"read-only FUSE mount"
"py-spy — read-only attach"
"GDB — high-risk Python injection"

CENTER zone title: "Trainctl runtime added by TrainctlMixin".
Show a small top card labelled "thin Lightning integration seam" connected to a stacked runtime with these blocks:
"immutable live state"
"REST + FUSE access surfaces"
"typed command queue"
"cooperative hold gate"
"artifact store"
"lifecycle hook dispatcher"
"Loguru diagnostics"

RIGHT zone title: "ordinary Lightning module and Trainer".
Show a large block containing:
"LightningModule"
"Trainer"
"training thread"
"callback points"
"model • batch • gradients • optimizer"

Draw one clear composition arrow from `TrainctlMixin` into `LightningModule`.
Draw solid arrows from REST/FUSE reads to "immutable live state".
Draw solid arrows from REST commands to "typed command queue" to "cooperative hold gate" to "training thread".
Draw a separate solid arrow from "callback points" to "lifecycle hook dispatcher".
Draw arrows from capture operations to "artifact store" and from "artifact store" back to REST/FUSE reads.
Draw dashed external-attach arrows from py-spy and GDB to the Lightning process. Label the GDB arrow "confirm hold first".
Draw a purple arrow from diagnostic records to "Loguru diagnostics".

Add this exact bottom boundary statement in a wide gray banner:
"REST and FUSE read published state or enqueue commands; the training thread touches live Trainer/model state."

Use clean orthogonal arrows and strong visual separation between zones. No crossed arrows, cloud architecture, async worker
queue, direct REST-to-model mutation, or native HTTP hook server.
```

### Required visual facts

- `TrainctlMixin` is the single, simple integration seam between a normal Lightning module and the runtime.
- REST/FUSE are two access surfaces over shared state; FUSE is read-only.
- Commands and holds are queued externally and applied by the training thread at safe points.
- Persisted captures are artifacts; transient live views are not silently persisted.
- py-spy observes from outside; GDB can execute Python in the target and is explicitly higher risk.
- Lifecycle hooks are triggered by callback points and are separate from the existing command safe-point path.
- Loguru is the diagnostic sink; native Lightning metric logging is not depicted as replaced.

### Caption / speaker note

The user does not need to rewrite the training loop or expose model internals manually. The mixin connects the normal
Lightning lifecycle to a runtime that separates read access, queued control, safe-point execution, and external
debugging.

## Slide 3 — From symptom to evidence to a deliberate next step

### Purpose

Show the practical debugging loop and how the major features fit together: observe a symptom, hold, collect the smallest
useful evidence, optionally act, and leave a durable diagnosis.

### Image-generation prompt

```text
Create a 16:9 left-to-right workflow infographic titled "From symptom to evidence to a deliberate next step".
Use six numbered stages connected by solid arrows:

1. "observe" — show `/run`, metrics, ranks, learning rates, pipeline timing, and a read-only FUSE tree;
2. "pause" — show `POST /holds` and a barrier labelled "safe point"; status changes to "held";
3. "choose the least invasive capture" — show three compact branches:
   "py-spy: stack only", "Trainctl capture: graph / profile / CUDA memory / batch", "GDB: live Python call, high risk";
4. "inspect artifacts" — show `manifest.json`, `graph.svg`, `profile`, `memory snapshot`, and `batch/loss/gradients`;
5. "targeted action" — show `set learning rate`, `reset momentum`, `set declared hparam`, and `finite-value check`;
6. "resume or preserve failure" — show `POST /control/continue` returning to training, and a separate red arrow labelled
   "backward exception → exception-held → diagnose → re-raise original exception".

Add a narrow bottom lane spanning stages 1–4 labelled "lifecycle hooks":
"baseline" → "logs/<run>/hooks/ copy" → "chmod +x light/<hook>.sh" →
"light: bounded metadata" or "heavy: temporary .npy + manifest" → "Loguru success/error".
Show disabled examples as gray files and enabled scripts as green files.

Add a final bottom statement exactly:
"The run keeps training semantics; every intervention is explicit, bounded, and visible."

Use teal for observation, amber for holds/actions, orange for captures, purple for artifacts/logs, red for GDB risk and
preserved failures, and gray for disabled/deferred features. Make all labels legible. Do not show automatic retries,
hidden background execution, native HTTP hooks, or a script aborting training merely because it exits nonzero.
```

### Required visual facts

- The workflow starts with read-only observation, not immediate mutation.
- A hold engages at a safe point before stateful inspection or GDB use.
- py-spy is the least invasive stack option; Trainctl captures produce artifacts; GDB is reserved for live Python calls
  and is explicitly high risk.
- Targeted controls are narrow and typed, not arbitrary optimizer-dict mutation or remote eval.
- The hook lane shows that a baseline becomes a run-local copy, examples start disabled, and scripts run synchronously.
- A failed hook is logged; it does not automatically retry or replace the original training exception.
- The backward-exception branch ends in re-raising the original error after diagnosis.

### Caption / speaker note

The user journey is: see the run, stop it cooperatively, gather evidence, make a narrowly defined change if justified,
then continue—or preserve the original failure with enough context to fix the code. Lifecycle hooks add an editable,
callback-local debugging lane without changing this larger control loop.

## Final image review checklist

- Exactly three slides are generated, in the order above.
- Slide 1 makes `TrainctlMixin` and “easy adoption” the first memorable message.
- The central message remains understandable without reading endpoint or module names.
- GDB is never presented as safe by default; py-spy is never shown mutating the target.
- REST/FUSE never appear to directly manipulate a live model.
- The run-local hook copy, disabled-by-default examples, `chmod +x`, light/heavy distinction, and Loguru errors appear
  as supporting details, not as the project's headline.
- No image implies async hook execution, retries, native HTTP lifecycle hooks, unrestricted code injection, or automatic
  recovery from a training exception.
- Text is large enough for a presentation and exact path names remain readable after 16:9 cropping.
