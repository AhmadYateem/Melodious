# Meeting Ready

Use this file during Hasan's meeting.

## Show These First

1. `sample_detections/FORMAT.md`
2. `sample_detections/class_mapping.json`
3. `sample_detections/HASAN_HANDOFF.md`
4. `sample_detections/reference/`

## What Hasan Can Use Immediately

- 5 contract-valid JSON payloads from real DeepScores test pages
- Stable class mapping for the 15 detector classes
- Stable field names and bbox conventions
- Canonical `image_path` per payload for page lookup

## Exact Folders

- Reference payloads: `sample_detections/reference/`
- Quick real-model payloads: `sample_detections/model_outputs_quick/`
- Quick checkpoint: `outputs/meeting_run/yolo_scratch_best.pth`

## What To Say Clearly

- The contract is frozen for integration.
- The reference payloads are from labeled real pages and are safe for Hassan to use immediately.
- The quick real-model payloads prove the export pipeline works from an actual checkpoint.
- The quick model is undertrained and should not be used to discuss detector quality.

## Do Not Claim

- Do not claim the quick checkpoint is a meaningful baseline.
- Do not claim the current detector is ready for evaluation or comparison.
- Do not reuse old detector F1 numbers without labeling them as pre-fix and unreliable.

## Next Step After The Meeting

1. Train a stronger checkpoint.
2. Replace `sample_detections/model_outputs_quick/` with 2-5 non-empty real detector outputs.
3. Start YOLOv8 fine-tuning and compare against the fixed custom detector.