"""Validation utilities for Anton's pipeline."""

def validate_stage_transition(prev_stage_result, next_stage):
    """Validate that the transition between pipeline stages is consistent."""
    if prev_stage_result is None:
        raise ValueError(f"Previous stage result missing for transition to {next_stage}")
    
    # Validate stage-specific requirements
    if next_stage == "stage_2" and "description" not in prev_stage_result:
        raise ValueError("Stage 1 must provide description for Stage 2 transition")
    
    if next_stage == "stage_3":
        if "detected_objects" not in prev_stage_result:
            raise ValueError("Stage 2 must provide detected_objects for Stage 3 transition")
    
    if next_stage == "stage_4":
        if "object_analyses" not in prev_stage_result:
            raise ValueError("Stage 3 must provide object_analyses for Stage 4 transition")
    
    return True 