"""N1.7 NEW_EMBODIMENT contract for already-encoded task-frame deltas."""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

MODALITIES = {
    'video': ModalityConfig(delta_indices=[0], modality_keys=['front']),
    'state': ModalityConfig(delta_indices=[0], modality_keys=['eef_relative_pose']),
    'action': ModalityConfig(delta_indices=list(range(16)), modality_keys=['eef_delta'],
        # ABSOLUTE disables the upstream absolute->relative preprocessing. Stored
        # actions are already deltas; this is not a world-absolute pose claim.
        action_configs=[ActionConfig(rep=ActionRepresentation.ABSOLUTE,
                                     type=ActionType.EEF, format=ActionFormat.XYZ_ROTVEC)]),
    'language': ModalityConfig(delta_indices=[0], modality_keys=['annotation.human.task_description']),
}
register_modality_config(MODALITIES, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
