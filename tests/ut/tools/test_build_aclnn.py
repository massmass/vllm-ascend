from pathlib import Path

BUILD_ACLNN_SCRIPT = Path(__file__).parents[3] / "csrc" / "build_aclnn.sh"
REPOSITORY_ROOT = BUILD_ACLNN_SCRIPT.parents[1]
RMS_NORM_DYNAMIC_QUANT_CALLERS = (
    REPOSITORY_ROOT / "vllm_ascend" / "attention" / "dsa_v1.py",
    REPOSITORY_ROOT / "vllm_ascend" / "attention" / "context_parallel" / "dsa_cp.py",
)
RMS_NORM_DYNAMIC_QUANT_CUSTOM_SOURCE = REPOSITORY_ROOT / "csrc" / "attention" / "rms_norm_dynamic_quant"
RMS_NORM_DYNAMIC_QUANT_TORCH_BINDINGS = (
    REPOSITORY_ROOT / "csrc" / "torch_binding.cpp",
    REPOSITORY_ROOT / "csrc" / "torch_binding_meta.cpp",
)


def _custom_ops_for_soc(script: str, soc_pattern: str) -> set[str]:
    branch_marker = f'elif [[ "$SOC_VERSION" =~ ^{soc_pattern} ]]; then'
    assert branch_marker in script, f"Missing {soc_pattern} build_aclnn.sh branch"

    section = script.split(branch_marker, maxsplit=1)[1]
    for next_branch_marker in ("\nelif ", "\nelse"):
        section = section.split(next_branch_marker, maxsplit=1)[0]

    return {
        line.strip().strip('"')
        for line in section.splitlines()
        if line.strip().startswith('"') and line.strip().endswith('"')
    }


def test_a2_and_a3_use_official_rms_norm_dynamic_quant() -> None:
    script = BUILD_ACLNN_SCRIPT.read_text()

    a2_ops = _custom_ops_for_soc(script, "ascend910b")
    a3_ops = _custom_ops_for_soc(script, "ascend910_93")

    assert "rms_norm_dynamic_quant" not in a2_ops
    assert "rms_norm_dynamic_quant" not in a3_ops

    for caller in RMS_NORM_DYNAMIC_QUANT_CALLERS:
        caller_source = caller.read_text()
        assert "torch.ops._C_ascend.npu_rms_norm_dynamic_quant" not in caller_source
        assert "torch_npu.npu_rms_norm_dynamic_quant" in caller_source

    for torch_binding in RMS_NORM_DYNAMIC_QUANT_TORCH_BINDINGS:
        assert "npu_rms_norm_dynamic_quant" not in torch_binding.read_text()

    assert not RMS_NORM_DYNAMIC_QUANT_CUSTOM_SOURCE.exists()
